#include "kernels.cuh"

__global__ void kernel_compute_AP(
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    AP,
    int N, int D)
{
    // un thread per elemento AP[d, i, j]
    int j = threadIdx.x + blockIdx.x * blockDim.x;  // colonna (stato dest)
    int i = threadIdx.y + blockIdx.y * blockDim.y;  // riga    (stato src)
    int d = blockIdx.z;                              // durata

    if (i >= N || j >= N || d >= D) return;

    AP[d*N*N + i*N + j] = trans_mat[i*N + j] + duration_probs[i*D + d];
}