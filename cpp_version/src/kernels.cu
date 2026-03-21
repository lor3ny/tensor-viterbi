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
    const int d = threadIdx.x;   // durata

    if (d >= tau || i >= N || j >= N) return;
    // if (i >= N || j >= N) return;

    // ── shared memory: solo riduzione ────────────────────────────────────── //
    // [ sh_val: tau doubles | sh_d: tau ints ]
    extern __shared__ char shmem[];
    double* sh_val = reinterpret_cast<double*>(shmem);
    int*    sh_d   = reinterpret_cast<int*>(sh_val + tau);

    // ── 1. Emission in registro + score diretto in shared ────────────────── //
    double cum = 0.0;
    for (int k = 0; k <= d; ++k)
        cum += emission_probs[obs_seq[t - k] * N + j];

    sh_val[d] = cum
              + delta[(t - 1 - d) * N + i]
              + AP[d * N*N + j*N + i];
    sh_d[d]   = d;
    __syncthreads();

    // if (d < tau) {
    //     double cum = 0.0;
    //     for (int k = 0; k <= d; ++k)
    //         cum += emission_probs[obs_seq[t - k] * N + j];
    //     sh_val[d] = cum
    //               + delta[(t - 1 - d) * N + i]
    //               + AP[d * N*N + j*N + i];
    //     sh_d[d]   = d;
    // } else {
    //     sh_val[d] = -1e300;   // mai il massimo
    //     sh_d[d]   = 0;
    // }


    // [V1] kernel_induction — thread 0 fa l'argmax su d
    if (d == 0) {
        double best = sh_val[0];
        int    bd   = sh_d[0];
        for (int k = 1; k < tau; ++k) {
            if (sh_val[k] > best) {
                best = sh_val[k];
                bd   = k;
            }
        }
        best_val_ji[j * N + i] = best;
        best_d_ji  [j * N + i] = bd;
    }

    // [V2] Riduzione parallela su d ──────────────────────────────────────────── //
    // pad alla prossima potenza di 2 per riduzione corretta
    // int stride = 1;
    // while (stride < tau) stride <<= 1;
    // stride >>= 1;

    // for (; stride > 0; stride >>= 1) {
    //     if (d < stride) {
    //         int other = d + stride;
    //         if (other < tau && sh_val[other] > sh_val[d]) {
    //             sh_val[d] = sh_val[other];
    //             sh_d  [d] = sh_d  [other];
    //         }
    //     }
    //     __syncthreads();
    // }

    // if (d == 0) {
    //     best_val_ji[j * N + i] = sh_val[0];
    //     best_d_ji  [j * N + i] = sh_d  [0];
    // }
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