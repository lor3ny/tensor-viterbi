#include "kernels.cuh"

#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void kernel_compute_AP(
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    AP,
    int N, int D)
{
    const int j = blockIdx.x;    // stato corrente
    const int i = blockIdx.y;    // stato precedente
    const int d = threadIdx.x;   // durata | [V2] d va da 0 a blockDim.x-1 (potenza di 2)

    if (i >= N || j >= N || d >= D) return;

    AP[d*N*N + i*N + j] = trans_mat[i*N + j] + duration_probs[i*D + d];
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
    int N, int D, int tau, int t)
{
    const int j = blockIdx.x;    // stato corrente
    const int i = blockIdx.y;    // stato precedente
    const int d = threadIdx.x;   // [V1] 0 ... tau-1 | [V2] 0 ... blockDim.x-1 (potenza di 2)


    // ── [V1] shared memory: solo riduzione ────────────────────────────────────── //
    // [ sh_val: tau doubles | sh_d: tau ints ]
    // if (d >= tau || i >= N || j >= N) return;
    // extern __shared__ char shmem[];
    // double* sh_val = reinterpret_cast<double*>(shmem);
    // int*    sh_d   = reinterpret_cast<int*>(sh_val + tau);

    // ── [V2] shared memory: solo riduzione ────────────────────────────────────── //
    // [ sh_val: tau doubles | sh_d: tau ints ]
    if (i >= N || j >= N) return;
    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + blockDim.x);  

    long long t0 = clock64();

    // [V1] Emission in registro + score diretto in shared ────────────────── //
    // double cum = 0.0;
    // for (int k = 0; k <= d; ++k)
    //     cum += emission_probs[obs_seq[t - k] * N + j];
    
    // sh_val[d] = cum
    //           + delta[(t - 1 - d) * N + i]
    //           + AP[d * N*N + j*N + i];
    // sh_d[d]   = d;
    // __syncthreads();

    // [V2.1] Same as V1, but extra threads initialize shared ────────────────── //
    // if (d < tau) {
    //     double cum = 0.0;
    //     for (int k = 0; k <= d; ++k)
    //         cum += emission_probs[obs_seq[t - k] * N + j];
    //     sh_val[d] = cum
    //               + delta[(t - 1 - d) * N + i]
    //               + AP[d * N*N + j*N + i];
    //     sh_d[d]   = d;
    // } else {
    //     sh_val[d] = -1e300;
    //     sh_d[d]   = 0;
    // }
    // __syncthreads();

    // [V2.2] Emission: shift O(1) ───────────────────────────────────────────── //
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

    long long t1 = clock64();

    if (d < tau) {
        sh_val[d] = em_val
                  + delta[(t - 1 - d) * N + i]
                  + AP[d * N*N + j*N + i];
        sh_d[d] = d;
    } else {
        sh_val[d] = -1e300;
        sh_d[d]   = 0;
    }
    __syncthreads();

    long long t2 = clock64();

    // [V1] Reduction — thread 0 fa l'argmax su d
    // if (d == 0) {
    //     double best = sh_val[0];
    //     int    bd   = sh_d[0];
    //     for (int k = 1; k < tau; ++k) {
    //         if (sh_val[k] > best) {
    //             best = sh_val[k];
    //             bd   = k;
    //         }
    //     }
    //     best_state_ji[j * N + i] = best;
    //     best_d_ji  [j * N + i] = bd;
    // }


    // [V2] Riduzione parallela su d ──────────────────────────────────────────── //
    // for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    //     if (d < stride) {
    //         double other_val = sh_val[d + stride];
    //         double curr_val  = sh_val[d];
    //         // aggiorna se strettamente maggiore, oppure uguale ma indice minore
    //         if (other_val > curr_val ||
    //         (other_val == curr_val && sh_d[d + stride] < sh_d[d])) {
    //             sh_val[d] = other_val;
    //             sh_d[d]   = sh_d[d + stride];
    //         }
    //     }
    //     __syncthreads();
    // }

    // [V3] Same as V2, but with intra-warp optimization //
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

    // ── intra-warp ───────────────────────────── //
    if (d < 32) {
    for (int stride = min(16, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
            if (d < stride) {
                double other_val = sh_val[d + stride];
                if (other_val > sh_val[d] ||
                (other_val == sh_val[d] && sh_d[d + stride] < sh_d[d])) {
                    sh_val[d] = other_val;
                    sh_d[d]   = sh_d[d + stride];
                }
            }
            __syncwarp();
        }
    }

    // [V2] [V3] Write result
    if (d == 0) {
        best_state_ji[j * N + i] = sh_val[0];
        best_d_ji  [j * N + i] = sh_d[0];
    }


    long long t3 = clock64();

    // [DEBUG]
    // if (d == 0 && j == 0 && i == 0 && (t >= 200 && t < 205)) {
    //     printf("emission: %lld, score: %lld, reduction: %lld (cycles)\n", t1-t0, t2-t1, t3-t2);
    // }
}


__global__ void kernel_reduce_i(
    const double* __restrict__ best_state_ji,   // N×N
    const int*    __restrict__ best_d_ji,     // N×N
    double*                    delta,       // T×N
    int*                       delta_state, // T×N
    int*                       delta_dur,   // T×N
    int N, int D, int t)
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
    const bool update = (t >= D) || (best_val > delta[t * N + j]);
    if (update) {
        delta      [t * N + j] = best_val;
        delta_state[t * N + j] = best_i;
        delta_dur  [t * N + j] = best_d + 1;
    }

    // [DEBUG]
    // if (t == 50) {
    // printf("F t=50 j=%d delta=%.6f state=%d dur=%d\n",
    //        j, delta[50*N+j], delta_state[50*N+j], delta_dur[50*N+j]);
    // }
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

        // ── 1. Emission shift O(1) ────────────────────────────────────────── //
        double em_val = -1e300;
        if (d < tau) {
            const double new_em = emission_probs[obs_seq[t] * N + j];
            em_val = (d == 0) ? new_em : new_em + d_em_cur[j * D + d - 1];
            if (i == 0)
                d_em_nxt[j * D + d] = em_val;
        }

        // ── 2. Score in shared memory ─────────────────────────────────────── //
        if (d < tau) {
            sh_val[d] = em_val
                      + delta[(t - 1 - d) * N + i]
                      + AP[d * N*N + j*N + i];
            sh_d[d] = d;
        } else {
            sh_val[d] = -1e300;
            sh_d[d]   = 0;
        }
        __syncthreads();

        // ── 3. Reduction ─────────────────────────────────────── //
 
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
        if (d < 32) {
            for (int stride = min(16, (int)(blockDim.x >> 1)); stride > 0; stride >>= 1) {
                if (d < stride) {
                    double other_val = sh_val[d + stride];
                    if (other_val > sh_val[d] ||
                       (other_val == sh_val[d] && sh_d[d + stride] < sh_d[d])) {
                        sh_val[d] = other_val;
                        sh_d[d]   = sh_d[d + stride];
                    }
                }
                __syncwarp();
            }
        }

        if (d == 0) {
            best_state_ji[j * N + i] = sh_val[0];
            best_d_ji  [j * N + i] = sh_d[0];
        }

        // ── tutti i blocchi hanno scritto best_state_ji ─────────────────────── //
        long long s0 = clock64();
        grid.sync();
        long long s1 = clock64();

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

            const bool update = (t >= D) || (best_val >= delta[t * N + j]);
            if (update) {
                delta      [t * N + j] = best_val;
                delta_state[t * N + j] = best_i;
                delta_dur  [t * N + j] = best_d + 1;
            }
        }

        // ── delta[t] scritto — tutti possono procedere a t+1 ─────────────── //
        grid.sync();

        cur = nxt;

        long long s2 = clock64();

        // [DEBUG]
        // if ((t >= 200 && t < 205) && j == 0 && i == 0 && d == 0)
        //     printf("grid.sync 1: %lld  grid.sync 2: %lld\n", s1-s0, s2-s1);
    }
}