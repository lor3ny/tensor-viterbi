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
    const double* __restrict__ d_past_delta,
    const double* __restrict__ d_AP,
    const double* __restrict__ d_emissions,
    double*                    d_result,
    int t, int N, int D)
{

}