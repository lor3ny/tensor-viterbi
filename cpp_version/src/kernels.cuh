#pragma once

#include <cuda_runtime.h>

// ── AP kernel ────────────────────────────────────────────────────────────── //

__global__ void kernel_compute_AP(
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    AP,
    int N, int D);

    
// ── Induction kernel ─────────────────────────────────────────────────────── //
__global__ void kernel_induction(
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    const double* __restrict__ emissions,
    double*                    score,
    int N, int D, int tau, int t);