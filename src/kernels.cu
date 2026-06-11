#include "kernels.cuh"

#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__constant__ int N, D, T;

void set_kernel_constants(int n, int d, int t) {
    cudaMemcpyToSymbol(N, &n, sizeof(int));
    cudaMemcpyToSymbol(D, &d, sizeof(int));
    cudaMemcpyToSymbol(T, &t, sizeof(int));
}

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
    const double* __restrict__ duration_probs,
    const double* __restrict__ duration_probs_linear,
    const double* __restrict__ emission_probs,
    const int*    __restrict__ obs_seq,
    double* delta, int* psi_dur,
    double* survival_probs)
{
    const int j = blockIdx.x;
    const int d = threadIdx.x;
    if (j >= N) return;

    const int warp_id   = d / WARP_SIZE;
    const int lane      = d % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // smem: [ sh_scan: blockDim.x | sh_warp_sums: num_warps ]
    extern __shared__ double sh[];
    double* sh_scan      = sh;
    double* sh_warp_sums = sh + blockDim.x;

    // ── Step 1: survival prefix sum (chunked) → survival_probs ───────────
    {
        double running_offset = 0.0;

        for (int base = 0; base < D; base += blockDim.x) {
            const int dg = d + base;

            double v = (dg < D) ? duration_probs_linear[j*D + (D-1-dg)] : 0.0;
            for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
                double u = WARP_SHFL_UP(v, stride);
                if (lane >= stride) v += u;
            }
            sh_scan[d] = v;
            if (lane == WARP_SIZE-1) sh_warp_sums[warp_id] = v;
            __syncthreads();

            if (d < WARP_SIZE) {
                double ws = (d < num_warps) ? sh_warp_sums[d] : 0.0;
                for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
                    double u = WARP_SHFL_UP(ws, stride);
                    if (d >= stride) ws += u;
                }
                if (d < num_warps) sh_warp_sums[d] = ws;
            }
            __syncthreads();

            v = (warp_id > 0 ? sh_scan[d] + sh_warp_sums[warp_id-1] : sh_scan[d])
                + running_offset;

            if (dg < D)
                survival_probs[j*D + (D-1-dg)] = log(v);

            running_offset += sh_warp_sums[num_warps-1];
            __syncthreads();
        }
    }

    // ── Step 2: emission prefix sum (chunked) + delta / psi_dur ───────────
    {
        double running_offset = 0.0;

        for (int base = 0; base < D; base += blockDim.x) {
            const int dg = d + base;

            double v = (dg < D) ? emission_probs[obs_seq[dg] * N + j] : 0.0;
            for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
                double u = WARP_SHFL_UP(v, stride);
                if (lane >= stride) v += u;
            }
            sh_scan[d] = v;
            if (lane == WARP_SIZE-1) sh_warp_sums[warp_id] = v;
            __syncthreads();

            if (d < WARP_SIZE) {
                double ws = (d < num_warps) ? sh_warp_sums[d] : 0.0;
                for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
                    double u = WARP_SHFL_UP(ws, stride);
                    if (d >= stride) ws += u;
                }
                if (d < num_warps) sh_warp_sums[d] = ws;
            }
            __syncthreads();

            v = (warp_id > 0 ? sh_scan[d] + sh_warp_sums[warp_id-1] : sh_scan[d])
                + running_offset;

            if (dg < D) {
                delta  [j*T + dg] = duration_probs[j*D + dg] + start_probs[j] + v;
                psi_dur[j*T + dg] = dg + 1;
            }

            running_offset += sh_warp_sums[num_warps-1];
            __syncthreads();
        }
    }
}


__global__ void kernel_induction(
    const double* __restrict__ em_t,      // emission_probs + obs_seq[t]*N (host-offset)
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    const double* __restrict__ delta,
    const double* __restrict__ em_cur,
    double*                    em_nxt,
    double* psi_state_ji,
    int*    psi_dur_ji,
    int tau, int t)
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
    const double new_em   = em_t[j];
    const double trans_ji = trans_mat[j * N + i];

    // ── Thread coarsening ──────────────────────────────────────────────────────
    // Ogni thread copre dg = d, d+blockDim.x, d+2*blockDim.x, …
    // Accumula argmax locale; dd è già l'indice globale (0..tau-1).
    double val = -1e300;
    int    dd  = 0;

    for (int base = 0; base < tau; base += blockDim.x) {
        const int dg = threadIdx.x + base;
        if (dg < tau) {
            const double em_val_dg = (dg == 0) ? new_em
                                                : new_em + em_cur[j * D + dg - 1];

            // em_nxt: scritto una volta per dg, solo dal blocco i==0
            if (i == 0)
                em_nxt[j * D + dg] = em_val_dg;

            const double v = em_val_dg + delta[i*T + (t-1-dg)] + trans_ji + duration_probs[j*D + dg];
            if (v > val || (v == val && dg < dd)) { val = v; dd = dg; }
        }
    }

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
    int t)
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
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    d_em0,
    double*                    d_em1,
    double*                    psi_state_ji,
    int*                       psi_dur_ji,
    int*                       psi_state,
    int*                       psi_dur)
{
    cg::grid_group grid = cg::this_grid();

    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;

    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + blockDim.x);

    const double trans_ji = trans_mat[j * N + i];
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
                      + trans_ji + duration_probs[j*D + d];
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
    const double* __restrict__ trans_mat,
    const double* __restrict__ survival_probs,
    const double* __restrict__ d_em_last,
    double*                    delta,
    double*                    psi_state_ji,
    int*                       psi_dur_ji)
{
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;
    if (i >= N || j >= N) return;

    const int warp_id   = d / WARP_SIZE;
    const int lane      = d % WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const int tau       = min(T - 1, D);

    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + num_warps);

    const double trans_ji = trans_mat[j * N + i];

    // ── Thread coarsening: accumulo argmax locale in registro ─────────────
    double val = -1e300;
    int    dd  = 0;

    for (int base = 0; base < tau; base += blockDim.x) {
        const int dg = threadIdx.x + base;
        if (dg < tau) {
            const double v = d_em_last[j * D + dg]
                           + delta[i * T + (T - 2 - dg)]
                           + trans_ji + survival_probs[j*D + dg];
            if (v > val || (v == val && dg < dd)) { val = v; dd = dg; }
        }
    }

    // ── Intra-warp argmax (shuffle) ───────────────────────────────────────
    for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
        double ov = WARP_SHFL_DOWN(val, s);
        int    od = WARP_SHFL_DOWN(dd,  s);
        if (ov > val || (ov == val && od < dd)) { val = ov; dd = od; }
    }

    // ── Fast path: blocco a warp singolo ─────────────────────────────────
    if (num_warps == 1) {
        if (lane == 0) {
            psi_state_ji[j * N + i] = val;
            psi_dur_ji  [j * N + i] = dd;
        }
        return;
    }

    // ── Cross-warp: lane 0 scrive in smem, warp 0 riduce ─────────────────
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
            if (ov > val || (ov == val && od < dd)) { val = ov; dd = od; }
        }

        if (lane == 0) {
            psi_state_ji[j * N + i] = val;
            psi_dur_ji  [j * N + i] = dd;
        }
    }
}

__global__ void kernel_tail_reduce_i(
    const double* __restrict__ psi_state_ji,
    const int*    __restrict__ psi_dur_ji,
    double*                    delta,
    int*                       psi_state,
    int*                       psi_dur,
    int t)
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

