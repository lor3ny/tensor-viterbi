#pragma once

#include <cuda_runtime.h>

// ── AP kernel ────────────────────────────────────────────────────────────── //

__global__ void kernel_compute_AP(
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    AP,
    int N, int D);

__global__ void kernel_induction(
    const double* __restrict__ d_past_delta,
    const double* __restrict__ d_AP,
    const double* __restrict__ d_emissions,
    double*                    d_result,
    int t, int N, int D);