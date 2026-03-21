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
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    double*                    score,
    int N, int D, int tau, int t)
{
    const int d = blockIdx.x;    // [0, D)
    const int j = threadIdx.y;   // stato corrente
    const int i = threadIdx.x;   // stato precedente

    if (d >= tau || i >= N || j >= N) return;

    // ── Emission: solo thread i=0 calcola, poi condivide su i ── //
    // sh_em[j] = Σ_{k=0}^{d} emission_probs[obs[t-k], j]
    extern __shared__ double sh_em[];   // shape: N  (un valore per j)

    if (i == 0) {
        double em = 0.0;
        for (int k = 0; k <= d; ++k)
            em += emission_probs[obs_seq[t - k] * N + j];
        sh_em[j] = em;
    }
    __syncthreads();


    // ── Induction: ogni thread (i,j) calcola un contributo per score[d,i,j] ── //
    score[d * N*N + j * N + i] =
        sh_em     [j] + //emissions [d * N   + j         ] +
        delta     [(t - 1 - d) * N + i ] +
        AP        [d * N*N + j*N + i   ];
}