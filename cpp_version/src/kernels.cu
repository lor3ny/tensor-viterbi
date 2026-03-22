#include "kernels.cuh"
#include <cstdio>

__global__ void kernel_compute_AP(
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    AP,
    int N, int D)
{
    int j = threadIdx.x;   // stato destinazione
    int i = threadIdx.y;   // stato sorgente
    int d = blockIdx.x;    // durata

    if (i >= N || j >= N || d >= D) return;

    AP[d*N*N + i*N + j] = trans_mat[i*N + j] + duration_probs[i*D + d];
}

__global__ void kernel_induction(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    double* best_val_ji,   // N×N output
    int*    best_d_ji,     // N×N output
    int N, int tau, int t)
{
    const int j = blockIdx.x;    // stato corrente
    const int i = blockIdx.y;    // stato precedente
    const int d = threadIdx.x;   // durata | [V2] d va da 0 a blockDim.x-1 (potenza di 2)


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

    // [V2] Same as V1, but extra threads initialize shared ────────────────── //
    if (d < tau) {
        double cum = 0.0;
        for (int k = 0; k <= d; ++k)
            cum += emission_probs[obs_seq[t - k] * N + j];
        sh_val[d] = cum
                  + delta[(t - 1 - d) * N + i]
                  + AP[d * N*N + j*N + i];
        sh_d[d]   = d;
    } else {
        sh_val[d] = -1e300;
        sh_d[d]   = 0;
    }
    __syncthreads();

    long long t1 = clock64();

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
    //     best_val_ji[j * N + i] = best;
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

    // [V3] Same as V2, but with intra-warp optimization (no __syncthreads() when stride<32) //
    // ── cross-warp: serve __syncthreads() ───────────────────── //
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

    // ── intra-warp: __syncwarp() ───────────────────────────── //
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
        best_val_ji[j * N + i] = sh_val[0];
        best_d_ji  [j * N + i] = sh_d[0];
    }


    long long t2 = clock64();

    if (d == 0 && j == 0 && i == 0 && (t >= 200 && t < 205)) {
        printf("emission + score cycles: %lld, reduction cycles: %lld\n", t1-t0, t2-t1);
    }
}


__global__ void kernel_reduce_i(
    const double* __restrict__ best_val_ji,   // N×N
    const int*    __restrict__ best_d_ji,     // N×N
    double*                    d_delta,       // T×N
    int*                       d_delta_state, // T×N
    int*                       d_delta_dur,   // T×N
    int N, int D, int t)
{
    const int j = threadIdx.x;   // un thread per j

    if (j >= N) return;

    // ── argmax su i — loop sequenziale (N<=20, trascurabile) ─────────────── //
    double best_val = best_val_ji[j * N + 0];
    int    best_d   = best_d_ji  [j * N + 0];
    int    best_i   = 0;

    for (int i = 1; i < N; ++i) {
        double v = best_val_ji[j * N + i];
        if (v > best_val) {
            best_val = v;
            best_d   = best_d_ji[j * N + i];
            best_i   = i;
        }
    }

    // ── aggiornamento condizionale su delta ───────────────────────────────── //
    const bool update = (t >= D) || (best_val > d_delta[t * N + j]);
    if (update) {
        d_delta      [t * N + j] = best_val;
        d_delta_state[t * N + j] = best_i;
        d_delta_dur  [t * N + j] = best_d + 1;
    }
}