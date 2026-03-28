#pragma once

#include <cuda_runtime.h>

// ── Initialization kernel ────────────────────────────────────────────────────────────── //

__global__ void kernel_initialization(
        const double* __restrict__ start_probs,
        const double* __restrict__ duration_probs,
        const double* __restrict__ emission_probs, 
        const int*    __restrict__ obs_seq,
        double* delta, int* psi_dur,
        const double* __restrict__ trans_mat,
        double* AP,
        int N, int D, int T);

    
// ── Induction kernel ─────────────────────────────────────────────────────── //
__global__ void kernel_induction(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    const double* __restrict__ delta,
    const double* __restrict__ AP,
    const double* __restrict__ em_cur,   // D×N — emissions iterazione precedente
    double*                    em_nxt,   // D×N — emissions iterazione corrente
    double* best_val_ji,   // N×N output
    int*    best_d_ji,     // N×N output
    int N, int D, int T, int tau, int t);


// ── Reduction kernel su i (argmax) ─────────────────────────────────────── //
__global__ void kernel_reduce_i(
    const double* __restrict__ best_val_ji,   // N×N
    const int*    __restrict__ best_d_ji,     // N×N
    double*                    delta,       // T×N
    int*                       psi_state, // T×N
    int*                       psi_dur,   // T×N
    int N, int D, int T, int t);


// ── Persistent kernel (induction + reduction) ───────────────────────────── //
__global__ void kernel_persistent(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    double*                    delta,        // T×N — lettura e scrittura
    const double* __restrict__ AP,
    double*                    em0,        // doppio buffer emissions
    double*                    em1,
    double*                    best_state_ji,  // N×N — buffer intermedio
    int*                       best_d_ji,    // N×N — buffer intermedio
    int*                       psi_state,  // T×N
    int*                       psi_dur,    // T×N
    int N, int D, int T);