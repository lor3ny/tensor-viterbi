#pragma once

#include <cuda_runtime.h>

// ── Initialization kernel ────────────────────────────────────────────────────────────── //

__global__ void kernel_initialization(
        const double* __restrict__ d_start_probs,
        const double* __restrict__ d_duration_probs,
        const double* __restrict__ d_emission_probs, 
        const int*    __restrict__ d_obs_seq,
        double* d_delta, int* d_delta_dur,
        const double* __restrict__ d_trans_mat,
        double* d_AP,
        int N, int D, int T);

    
// ── Induction kernel ─────────────────────────────────────────────────────── //
__global__ void kernel_induction(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    const double* __restrict__ d_em_cur,   // D×N — emissions iterazione precedente
    double*                    d_em_nxt,   // D×N — emissions iterazione corrente
    double* best_val_ji,   // N×N output
    int*    best_d_ji,     // N×N output
    int N, int D, int T, int tau, int t);


// ── Reduction kernel su i (argmax) ─────────────────────────────────────── //
__global__ void kernel_reduce_i(
    const double* __restrict__ best_val_ji,   // N×N
    const int*    __restrict__ best_d_ji,     // N×N
    double*                    d_delta,       // T×N
    int*                       d_delta_state, // T×N
    int*                       d_delta_dur,   // T×N
    int N, int D, int T, int t);


// ── Persistent kernel (induction + reduction) ───────────────────────────── //
__global__ void kernel_persistent(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    double*                    delta,        // T×N — lettura e scrittura
    const double* __restrict__ AP,
    double*                    d_em0,        // doppio buffer emissions
    double*                    d_em1,
    double*                    best_state_ji,  // N×N — buffer intermedio
    int*                       best_d_ji,    // N×N — buffer intermedio
    int*                       delta_state,  // T×N
    int*                       delta_dur,    // T×N
    int N, int D, int T);