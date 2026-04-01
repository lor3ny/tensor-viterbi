#include "kernels.cuh"

#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifdef __HIP_PLATFORM_AMD__
    #define WARP_SHFL_UP(val, stride)   __shfl_up(val, stride)
    #define WARP_SHFL_DOWN(val, stride) __shfl_down(val, stride)
    #define FULL_MASK 0xffffffffffffffff
#else
    #define WARP_SHFL_UP(val, stride)   __shfl_up_sync(0xffffffff, val, stride)
    #define WARP_SHFL_DOWN(val, stride) __shfl_down_sync(0xffffffff, val, stride)
    #define FULL_MASK 0xffffffff
#endif


__global__ void kernel_initialization(
        const double* __restrict__ start_probs,
        const double* __restrict__ duration_probs,         // N×D log-space
        const double* __restrict__ duration_probs_linear,  // N×D linear-space
        const double* __restrict__ emission_probs,
        const int*    __restrict__ obs_seq,
        double* delta, int* psi_dur,
        const double* __restrict__ trans_mat,
        double* AP,
        double* AP_tail,
        int N, int D, int T)
{
    const int j       = blockIdx.x;
    const int i       = blockIdx.y;
    const int d       = threadIdx.x;

    if (i >= N || j >= N) return;

    const int warp_id   = d / WARP_SIZE;
    const int lane      = d % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // shared layout: [ sh_surv: blockDim.x | sh_em: blockDim.x | sh_warp_sums: num_warps ]
    // sh_warp_sums è riusata per entrambi i scan (survival e emissions)
    extern __shared__ double sh[];
    double* sh_surv      = sh;
    double* sh_em        = sh + blockDim.x;
    double* sh_warp_sums = sh + 2 * blockDim.x;

    //* ── Step 1: Survival probs — reverse prefix sum in linear space ───────── //
    // thread d calcola S[D-1-d] = sum_{k=D-1-d}^{D-1} duration_probs_linear[j*D+k]
    double surv_val = (d < D) ? duration_probs_linear[j*D + (D-1-d)] : 0.0;


    for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
        double v = WARP_SHFL_UP(surv_val, stride);
        if (lane >= stride) surv_val += v;
    }
    sh_surv[d] = surv_val;
    if (lane == WARP_SIZE - 1)
        sh_warp_sums[warp_id] = surv_val;
    __syncthreads();  // sync 1

    if (d < WARP_SIZE) {
        double ws = (d < num_warps) ? sh_warp_sums[d] : 0.0;
        for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
            double v = WARP_SHFL_UP(ws, stride);
            if (d >= stride) ws += v;
        }
        if (d < num_warps)
            sh_warp_sums[d] = ws;
    }
    __syncthreads();  // sync 2

    surv_val = (warp_id > 0)
             ? sh_surv[d] + sh_warp_sums[warp_id - 1]
             : sh_surv[d];


    //* ── Step 2: AP, AP_tail, survival_probs global ────────────────────────── //
    if (d < D) {
        double log_surv = log(surv_val);
        AP     [j*N*D + i*D + d] = trans_mat[j*N + i] + duration_probs[j*D + d];
        AP_tail[j*N*D + i*D + (D-1-d)] = trans_mat[j*N + i] + log_surv;
    }

    //* ── Step 3: Phase 1 — solo i==0 ──────────────────────────────────────── //
    if (i != 0) return;

    // riusa sh_warp_sums per il prefix sum delle emissions
    double val = (d < D) ? emission_probs[obs_seq[d] * N + j] : 0.0;

    for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
        double v = WARP_SHFL_UP(val, stride);
        if (lane >= stride) val += v;
    }
    sh_em[d] = val;
    if (lane == WARP_SIZE - 1)
        sh_warp_sums[warp_id] = val;
    __syncthreads();  // sync 3

    if (d < WARP_SIZE) {
        double ws = (d < num_warps) ? sh_warp_sums[d] : 0.0;
        for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
            double v = WARP_SHFL_UP(ws, stride);
            if (d >= stride) ws += v;
        }
        if (d < num_warps)
            sh_warp_sums[d] = ws;
    }
    __syncthreads();  // sync 4

    val = (warp_id > 0) ? sh_em[d] + sh_warp_sums[warp_id - 1] : sh_em[d];

    if (d < D) {
        delta  [j*T + d] = duration_probs[j*D + d] + start_probs[j] + val;
        psi_dur[j*T + d] = d + 1;
    }
}



__global__ void kernel_induction(
    int obs_t,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    const double* __restrict__ em_cur,
    double*                    em_nxt,
    double* psi_state_ji,
    int*    psi_dur_ji,
    int N, int D, int T, int tau, int t)
{
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;

    if (i >= N || j >= N) return;

    const int warp_id   = d / WARP_SIZE;
    const int lane      = d % WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // smem: solo num_warps entry per la riduzione cross-warp.
    // Quando num_warps == 1 (bs <= WARP_SIZE) non viene usata.
    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + num_warps);

    //* Broadcast emission — tutti i thread leggono lo stesso indirizzo (L1 broadcast) *//
    const double new_em = emission_probs[obs_t * N + j];

    //* Cached Emissions *//
    double em_val = -1e300;
    if (d < tau) {
        em_val = (d == 0) ? new_em : new_em + em_cur[j * D + d - 1];
        if (i == 0)
            em_nxt[j * D + d] = em_val;
    }

    //* Brick — in registers, niente smem *//
    double val = (d < tau) ? (em_val + delta[i*T + (t-1-d)] + AP[j*N*D + i*D + d])
                           : -1e300;
    int    dd  = (d < tau) ? d : 0;

    //* Intra-warp argmax (shuffle, zero syncthreads) *//
    for (int s = min(WARP_SIZE >> 1, (int)(blockDim.x >> 1)); s > 0; s >>= 1) {
        double ov = WARP_SHFL_DOWN(val, s);
        int    od = WARP_SHFL_DOWN(dd,  s);
        if (ov > val || (ov == val && od < dd)) {
            val = ov;
            dd  = od;
        }
    }

    //* Fast path: blocco a warp singolo — risultato già in lane 0 *//
    if (num_warps == 1) {
        if (lane == 0) {
            psi_state_ji[j * N + i] = val;
            psi_dur_ji  [j * N + i] = dd;
        }
        return;
    }

    //* Cross-warp: lane 0 di ogni warp scrive in smem, poi warp 0 riduce *//
    if (lane == 0) {
        sh_val[warp_id] = val;
        sh_d  [warp_id] = dd;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? sh_val[lane] : -1e300;
        dd  = (lane < num_warps) ? sh_d  [lane] : 0;

        for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
            double ov = WARP_SHFL_DOWN(val, s);
            int    od = WARP_SHFL_DOWN(dd,  s);
            if (ov > val || (ov == val && od < dd)) {
                val = ov;
                dd  = od;
            }
        }
        if (lane == 0) {
            psi_state_ji[j * N + i] = val;
            psi_dur_ji  [j * N + i] = dd;
        }
    }
}


__global__ void kernel_reduce_i(
    const double* __restrict__ psi_state_ji,
    const int*    __restrict__ psi_dur_ji,
    double*                    delta,
    int*                       psi_state,
    int*                       psi_dur,
    int N, int D, int T, int t)
{
    const int j = blockIdx.x;
    const int i = threadIdx.x;

    if (j >= N) return;

    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_i   = reinterpret_cast<int*>(sh_val + blockDim.x);
    int*    sh_d   = sh_i + blockDim.x;

    if (i < N) {
        sh_val[i] = psi_state_ji[j * N + i];
        sh_i  [i] = i;
        sh_d  [i] = psi_dur_ji[j * N + i];
    } else {
        sh_val[i] = -1e300;
        sh_i  [i] = 0;
        sh_d  [i] = 0;
    }
    __syncthreads();

    // ── cross-warp ───────────────────────────────────────────────────────── //
    for (int stride = blockDim.x >> 1; stride >= WARP_SIZE; stride >>= 1) {
        if (i < stride) {
            if (sh_val[i + stride] > sh_val[i]) {
                sh_val[i] = sh_val[i + stride];
                sh_i  [i] = sh_i  [i + stride];
                sh_d  [i] = sh_d  [i + stride];
            }
        }
        __syncthreads();
    }

    // ── intra-warp ───────────────────────────────────────────────────────── //
    if (i < WARP_SIZE) {
        double reg_val = sh_val[i];
        int    reg_i   = sh_i[i];
        int    reg_d   = sh_d[i];

        for (int stride = min(WARP_SIZE >> 1, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
            double other_val = WARP_SHFL_DOWN(reg_val, stride);
            int    other_i   = WARP_SHFL_DOWN(reg_i,   stride);
            int    other_d   = WARP_SHFL_DOWN(reg_d,   stride);
            if (other_val > reg_val) {
                reg_val = other_val;
                reg_i   = other_i;
                reg_d   = other_d;
            }
        }
        if (i == 0) {
            sh_val[0] = reg_val;
            sh_i  [0] = reg_i;
            sh_d  [0] = reg_d;
        }
    }

    if (i == 0) {
        const bool update = (t >= D) || (sh_val[0] > delta[j*T + t]);
        if (update) {
            delta    [j*T + t] = sh_val[0];
            psi_state[j*T + t] = sh_i[0];
            psi_dur  [j*T + t] = sh_d[0] + 1;
        }
    }
}


__global__ void kernel_persistent(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    double*                    delta,
    const double* __restrict__ AP,
    double*                    d_em0,
    double*                    d_em1,
    double*                    psi_state_ji,
    int*                       psi_dur_ji,
    int*                       psi_state,
    int*                       psi_dur,
    int N, int D, int T)
{
    cg::grid_group grid = cg::this_grid();

    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;

    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + blockDim.x);

    int cur = 0;

    for (int t = 1; t < T; ++t) {
        const int tau = min(t, D);
        const int nxt = 1 - cur;
        double* em_cur = (cur == 0) ? d_em0 : d_em1;
        double* em_nxt = (cur == 0) ? d_em1 : d_em0;

        //* Cached Emissions *//
        double em_val = -1e300;
        if (d < tau) {
            const double new_em = emission_probs[obs_seq[t] * N + j];
            em_val = (d == 0) ? new_em : new_em + em_cur[j * D + d - 1];
            if (i == 0)
                em_nxt[j * D + d] = em_val;
        }

        //* Brick *//
        if (d < tau) {
            sh_val[d] = em_val
                      + delta[i*T + (t-1-d)]
                      + AP[j*N*D + i*D + d];
            sh_d[d] = d;
        } else {
            sh_val[d] = -1e300;
            sh_d[d]   = 0;
        }
        __syncthreads();

        //* Argmax *//
        // ── cross-warp ───────────────────────────────────────────────────── //
        for (int stride = blockDim.x >> 1; stride >= WARP_SIZE; stride >>= 1) {
            if (d < stride) {
                double other_val = sh_val[d + stride];
                if (other_val > sh_val[d] ||
                   (other_val == sh_val[d] && sh_d[d + stride] < sh_d[d])) {
                    sh_val[d] = other_val;
                    sh_d[d]   = sh_d[d + stride];
                }
            }
            __syncthreads();
        }

        // ── intra-warp ───────────────────────────────────────────────────── //
        if (d < WARP_SIZE) {
            double reg_val = sh_val[d];
            int    reg_d   = sh_d[d];

            for (int stride = min(WARP_SIZE >> 1, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
                double other_val = WARP_SHFL_DOWN(reg_val, stride);
                int    other_d   = WARP_SHFL_DOWN(reg_d,   stride);
                if (other_val > reg_val ||
                   (other_val == reg_val && other_d < reg_d)) {
                    reg_val = other_val;
                    reg_d   = other_d;
                }
            }
            if (d == 0) {
                sh_val[0] = reg_val;
                sh_d[0]   = reg_d;
            }
        }

        if (d == 0) {
            psi_state_ji[j * N + i] = sh_val[0];
            psi_dur_ji  [j * N + i] = sh_d[0];
        }

        grid.sync();

        if (i == 0 && d == 0) {
            double best_val = psi_state_ji[j * N + 0];
            int    best_d   = psi_dur_ji  [j * N + 0];
            int    best_i   = 0;

            for (int k = 1; k < N; ++k) {
                double v = psi_state_ji[j * N + k];
                if (v > best_val) {
                    best_val = v;
                    best_d   = psi_dur_ji[j * N + k];
                    best_i   = k;
                }
            }

            const bool update = (t >= D) || (best_val >= delta[j*T + t]);
            if (update) {
                delta    [j*T + t] = best_val;
                psi_state[j*T + t] = best_i;
                psi_dur  [j*T + t] = best_d + 1;
            }
        }

        grid.sync();
        cur = nxt;
    }
}


__global__ void kernel_tail_adjustment(
    const double* __restrict__ AP_tail,
    const double* __restrict__ d_em_last,
    double*                    delta,
    double*                    psi_state_ji,
    int*                       psi_dur_ji,
    int N, int D, int T)
{
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;

    if (i >= N || j >= N) return;

    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + blockDim.x);

    const int tau = min(T - 1, D);

    // ── Brick ─────────────────────────────────────────────────────────────── //
    if (d < tau) {
        sh_val[d] = d_em_last[j * D + d]
                  + delta[i * T + (T - 2 - d)]
                  + AP_tail[j * N*D + i*D + d];
        sh_d[d] = d;
    } else {
        sh_val[d] = -1e300;
        sh_d[d]   = 0;
    }
    __syncthreads();

    // ── cross-warp ───────────────────────────────────────────────────────── //
    for (int stride = blockDim.x >> 1; stride >= WARP_SIZE; stride >>= 1) {
        if (d < stride) {
            double other_val = sh_val[d + stride];
            if (other_val > sh_val[d] ||
               (other_val == sh_val[d] && sh_d[d + stride] < sh_d[d])) {
                sh_val[d] = other_val;
                sh_d[d]   = sh_d[d + stride];
            }
        }
        __syncthreads();
    }

    // ── intra-warp ───────────────────────────────────────────────────────── //
    if (d < WARP_SIZE) {
        double reg_val = sh_val[d];
        int    reg_d   = sh_d[d];

        for (int stride = min(WARP_SIZE >> 1, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
            double other_val = WARP_SHFL_DOWN(reg_val, stride);
            int    other_d   = WARP_SHFL_DOWN(reg_d,   stride);
            if (other_val > reg_val ||
               (other_val == reg_val && other_d < reg_d)) {
                reg_val = other_val;
                reg_d   = other_d;
            }
        }
        if (d == 0) {
            sh_val[0] = reg_val;
            sh_d[0]   = reg_d;
        }
    }

    if (d == 0) {
        psi_state_ji[j * N + i] = sh_val[0];
        psi_dur_ji  [j * N + i] = sh_d[0];
    }
}

__global__ void kernel_tail_reduce_i(
    const double* __restrict__ psi_state_ji,
    const int*    __restrict__ psi_dur_ji,
    double*                    delta,
    int*                       psi_state,
    int*                       psi_dur,
    int N, int D, int T, int t)
{
    const int j = blockIdx.x;
    const int i = threadIdx.x;

    if (j >= N) return;

    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_i   = reinterpret_cast<int*>(sh_val + blockDim.x);
    int*    sh_d   = sh_i + blockDim.x;

    if (i < N) {
        sh_val[i] = psi_state_ji[j * N + i];
        sh_i  [i] = i;
        sh_d  [i] = psi_dur_ji[j * N + i];
    } else {
        sh_val[i] = -1e300;
        sh_i  [i] = 0;
        sh_d  [i] = 0;
    }
    __syncthreads();

    // ── cross-warp ───────────────────────────────────────────────────────── //
    for (int stride = blockDim.x >> 1; stride >= WARP_SIZE; stride >>= 1) {
        if (i < stride) {
            if (sh_val[i + stride] > sh_val[i]) {
                sh_val[i] = sh_val[i + stride];
                sh_i  [i] = sh_i  [i + stride];
                sh_d  [i] = sh_d  [i + stride];
            }
        }
        __syncthreads();
    }

    // ── intra-warp ───────────────────────────────────────────────────────── //
    if (i < WARP_SIZE) {
        double reg_val = sh_val[i];
        int    reg_i   = sh_i[i];
        int    reg_d   = sh_d[i];

        for (int stride = min(WARP_SIZE >> 1, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
            double other_val = WARP_SHFL_DOWN(reg_val, stride);
            int    other_i   = WARP_SHFL_DOWN(reg_i,   stride);
            int    other_d   = WARP_SHFL_DOWN(reg_d,   stride);
            if (other_val > reg_val) {
                reg_val = other_val;
                reg_i   = other_i;
                reg_d   = other_d;
            }
        }
        if (i == 0) {
            sh_val[0] = reg_val;
            sh_i  [0] = reg_i;
            sh_d  [0] = reg_d;
        }
    }

    if (i == 0) {
        delta    [j*T + t] = sh_val[0];
        psi_state[j*T + t] = sh_i[0];
        psi_dur  [j*T + t] = sh_d[0] + 1;
    }
}

