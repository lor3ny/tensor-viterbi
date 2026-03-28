#include "kernels.cuh"

#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__global__ void kernel_initialization(
        const double* __restrict__ start_probs,
        const double* __restrict__ duration_probs,
        const double* __restrict__ emission_probs, 
        const int*    __restrict__ obs_seq,
        double* delta, int* delta_dur,
        const double* __restrict__ trans_mat,
        double* AP, int N, int D, int T)
{
    const int j = blockIdx.x;    // stato corrente
    const int i = blockIdx.y;    // stato precedente
    const int d = threadIdx.x;   // durata 

    if (i >= N || j >= N || d >= D) return;

    //* AP *//
    AP[j *N*D + i*D + d] = trans_mat[j*N + i] + duration_probs[j*D + d];
    
    // ── Phase 1 — solo un blocco per stato (i==0 arbitrario) ─────────── //
    if (i != 0) return;

    //* Emissions *//
    extern __shared__ double sh_em[];
    sh_em[d] = emission_probs[obs_seq[d] * N + j];
    __syncthreads();

    // Prefix sum sequenziale — thread d somma sh_em[0..d]
    double emissions = 0.0;
    for (int k = 0; k <= d; ++k)
        emissions += sh_em[k];

    delta    [j*T + d] = duration_probs[j*D + d] + start_probs[j] + emissions;
    delta_dur[j*T + d] = d + 1;
}


__global__ void kernel_induction(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    const double* __restrict__ d_em_cur,   // D×N — emissions iterazione precedente
    double*                    d_em_nxt,   // D×N — emissions iterazione corrente
    double* best_state_ji,   // N×N output
    int*    best_d_ji,     // N×N output
    int N, int D, int T, int tau, int t)
{
    const int j = blockIdx.x;    // stato corrente
    const int i = blockIdx.y;    // stato precedente
    const int d = threadIdx.x;   // 0 ... blockDim.x-1 (potenza di 2)

    // [ sh_val: tau doubles | sh_d: tau ints ]
    if (i >= N || j >= N) return;
    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + blockDim.x);  

    
    //* Cached Emissions *//
    // solo i blocchi i=0 calcolano e scrivono — ridondante su i altrimenti
    double em_val = -1e300;
    if (d < tau) {
        const double new_em = emission_probs[obs_seq[t] * N + j];
        if (d == 0) {
            em_val = new_em;
        } else {
            em_val = new_em + d_em_cur[j * D + d - 1];
        }
        // scrive per t+1 — solo un blocco per j (i=0 è arbitrario, tutti scrivono lo stesso)
        if (i == 0)
            d_em_nxt[j * D + d] = em_val;
    }

    //* Brick *//
    if (d < tau) {
        sh_val[d] = em_val
                  + delta[i*T + (t-1-d)]
                  + AP[j * N*D + i*D + d];
        sh_d[d] = d;
    } else {
        sh_val[d] = -1e300;
        sh_d[d]   = 0;
    }
    __syncthreads();

    //* Intra-Block Argmax *//
    // ── cross-warp ───────────────────── //
    for (int stride = blockDim.x >> 1; stride >= 32; stride >>= 1) {
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

    // intra-warp //
    // if (d < 32) {
    // for (int stride = min(16, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
    //         if (d < stride) {
    //             double other_val = sh_val[d + stride];
    //             if (other_val > sh_val[d] ||
    //             (other_val == sh_val[d] && sh_d[d + stride] < sh_d[d])) {
    //                 sh_val[d] = other_val;
    //                 sh_d[d]   = sh_d[d + stride];
    //             }
    //         }
    //         __syncwarp();
    //     }
    // }

    if (d < 32) {
        double reg_val = sh_val[d];
        int    reg_d   = sh_d[d];

        for (int stride = min(16, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
            double other_val = __shfl_down_sync(0xffffffff, reg_val, stride);
            int    other_d   = __shfl_down_sync(0xffffffff, reg_d,   stride);
            if (other_val > reg_val ||
            (other_val == reg_val && other_d < reg_d)) {
                reg_val = other_val;
                reg_d   = other_d;
            }
        }
        // solo thread 0 scrive il risultato finale in shared
        if (d == 0) {
            sh_val[0] = reg_val;
            sh_d[0]   = reg_d;
        }
    }

    if (d == 0) {
        best_state_ji[j * N + i] = sh_val[0];
        best_d_ji  [j * N + i] = sh_d[0];
    }

}


__global__ void kernel_reduce_i(
    const double* __restrict__ best_state_ji,   // N×N
    const int*    __restrict__ best_d_ji,     // N×N
    double*                    delta,       // T×N
    int*                       delta_state, // T×N
    int*                       delta_dur,   // T×N
    int N, int D, int T, int t)
{
    const int j = threadIdx.x;   // un thread per j

    if (j >= N) return;

    // ── argmax su i — loop sequenziale (N<=20, trascurabile) ─────────────── //
    double best_val = best_state_ji[j * N + 0];
    int    best_d   = best_d_ji  [j * N + 0];
    int    best_i   = 0;

    for (int i = 1; i < N; ++i) {
        double v = best_state_ji[j * N + i];
        if (v > best_val) {
            best_val = v;
            best_d   = best_d_ji[j * N + i];
            best_i   = i;
        }
    }

    // ── aggiornamento condizionale su delta ───────────────────────────────── //
    const bool update = (t >= D) || (best_val > delta[j*T + t]);
    if (update) {
        delta      [j*T + t] = best_val;
        delta_state[j*T + t] = best_i;
        delta_dur  [j*T + t] = best_d + 1;
    }

}



__global__ void kernel_persistent(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    double*                    delta,        // T×N — lettura e scrittura
    const double* __restrict__ AP,
    double*                    d_em0,        // doppio buffer emissions
    double*                    d_em1,
    double*                    best_state_ji,  // N×N — buffer intermedio
    int*                       best_d_ji,    // N×N — buffer intermedio
    int*                       delta_state,  // T×N
    int*                       delta_dur,    // T×N
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
        double* d_em_cur = (cur == 0) ? d_em0 : d_em1;
        double* d_em_nxt = (cur == 0) ? d_em1 : d_em0;

        //* Cached Emissions *//
        double em_val = -1e300;
        if (d < tau) {
            const double new_em = emission_probs[obs_seq[t] * N + j];
            em_val = (d == 0) ? new_em : new_em + d_em_cur[j * D + d - 1];
            if (i == 0)
                d_em_nxt[j * D + d] = em_val;
        }

        //* Brick *//
        if (d < tau) {
            sh_val[d] = em_val
                      + delta[i*T + (t-1-d)]
                      + AP[j * N*D + i*D + d];
            sh_d[d] = d;
        } else {
            sh_val[d] = -1e300;
            sh_d[d]   = 0;
        }
        __syncthreads();

        //* Argmax *//
 
        // ── cross-warp reduction ──────────────────────────────────────────── //
        for (int stride = blockDim.x >> 1; stride >= 32; stride >>= 1) {
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

        // ── intra-warp reduction ──────────────────────────────────────────── //
        // if (d < 32) {
        //     for (int stride = min(16, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
        //         if (d < stride) {
        //             double other_val = sh_val[d + stride];
        //             if (other_val > sh_val[d] ||
        //                (other_val == sh_val[d] && sh_d[d + stride] < sh_d[d])) {
        //                 sh_val[d] = other_val;
        //                 sh_d[d]   = sh_d[d + stride];
        //             }
        //         }
        //         __syncwarp();
        //     }
        // }

        if (d < 32) {
            double reg_val = sh_val[d];
            int    reg_d   = sh_d[d];

            for (int stride = min(16, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
                double other_val = __shfl_down_sync(0xffffffff, reg_val, stride);
                int    other_d   = __shfl_down_sync(0xffffffff, reg_d,   stride);
                if (other_val > reg_val ||
                (other_val == reg_val && other_d < reg_d)) {
                    reg_val = other_val;
                    reg_d   = other_d;
                }
            }
            // solo thread 0 scrive il risultato finale in shared
            if (d == 0) {
                sh_val[0] = reg_val;
                sh_d[0]   = reg_d;
            }
        }
            
        if (d == 0) {
            best_state_ji[j * N + i] = sh_val[0];
            best_d_ji  [j * N + i] = sh_d[0];
        }

        grid.sync();
        // ── tutti i blocchi hanno scritto best_state_ji ─────────────────────── //

        // ── 2. Reduce su i — solo blocchi con i=0 ────────────────────────── //
        if (i == 0 && d == 0) {
            double best_val = best_state_ji[j * N + 0];
            int    best_d   = best_d_ji  [j * N + 0];
            int    best_i   = 0;

            for (int k = 1; k < N; ++k) {
                double v = best_state_ji[j * N + k];
                if (v > best_val) {
                    best_val = v;
                    best_d   = best_d_ji[j * N + k];
                    best_i   = k;
                }
            }

            const bool update = (t >= D) || (best_val >= delta[j*T + t]);
            if (update) {
                delta      [j*T + t] = best_val;
                delta_state[j*T + t] = best_i;
                delta_dur  [j*T + t] = best_d + 1;
            }
        }

        grid.sync();
        // ── delta[t] scritto — tutti possono procedere a t+1 ─────────────── //

        cur = nxt;
    }
}