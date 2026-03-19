#pragma once

#include <cuda_runtime.h>

// ── AP kernel ────────────────────────────────────────────────────────────── //

__global__ void kernel_compute_AP(
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    AP,
    int N, int D);