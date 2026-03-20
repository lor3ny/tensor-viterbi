#include "kernels.cuh"

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
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    const double* __restrict__ emissions,
    double*                    score,
    int N, int D, int tau, int t)
{
    const int d = blockIdx.x;    // [0, D)
    const int j = threadIdx.y;   // stato corrente
    const int i = threadIdx.x;   // stato precedente

    if (d >= tau || i >= N || j >= N) return;

//    int idx = (start + (window_size - 1 - i)) * n_features + j;
    score[d * N*N + j * N + i] =
        emissions [d * N   + j         ] +
        delta     [(t - 1 - d) * N + i ] +
        AP        [d * N*N + j*N + i   ];
}