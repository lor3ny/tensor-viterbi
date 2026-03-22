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
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    double* best_val_ji,   // N×N output
    int*    best_d_ji,     // N×N output
    int N, int tau, int t);


// ── Reduction kernel su i (argmax) ─────────────────────────────────────── //
__global__ void kernel_reduce_i(
    const double* __restrict__ best_val_ji,   // N×N
    const int*    __restrict__ best_d_ji,     // N×N
    double*                    d_delta,       // T×N
    int*                       d_delta_state, // T×N
    int*                       d_delta_dur,   // T×N
    int N, int D, int t);


// ── Persistent kernel (induction + reduction) ───────────────────────────── //
__global__ void kernel_persistent(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    double*                    delta,        // T×N — lettura e scrittura
    const double* __restrict__ AP,
    double*                    best_state_ji,  // N×N — buffer intermedio
    int*                       best_d_ji,    // N×N — buffer intermedio
    int*                       delta_state,  // T×N
    int*                       delta_dur,    // T×N
    int N, int D, int T);