#pragma once

#include <cuda_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
    #define WARP_SIZE 64
#else
    #define WARP_SIZE 32
#endif

// ── Initialization kernel ────────────────────────────────────────────────────────────── //

__global__ void kernel_initialization(
        const double* __restrict__ start_probs,
        const double* __restrict__ duration_probs,         // N×D log-space
        const double* __restrict__ duration_probs_linear,  // N×D linear-space
        const double* __restrict__ emission_probs,
        const int*    __restrict__ obs_seq,
        double* delta, int* psi_dur,
        const double* __restrict__ trans_mat,
        double* AP,
        double* AP_tail,
        int N, int D, int T);

    
// ── Induction kernel ─────────────────────────────────────────────────────── //
__global__ void kernel_induction(
    int                           obs_t,         // obs_seq[t] — valore diretto
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

// ── Tail Adjustment kernel ─────────────────────────────────────────────── //
__global__ void kernel_tail_adjustment(
    const double* __restrict__ AP_tail,         // N×N×D — trans_mat + log_surv
    const double* __restrict__ d_em_last,       // D×N — emissions t=T-1
    double*                    delta,           // N×T — lettura past delta
    double*                    psi_state_ji,    // N×N — output per kernel_reduce_i
    int*                       psi_dur_ji,      // N×N — output per kernel_reduce_i
    int N, int D, int T);

__global__ void kernel_tail_reduce_i(
    const double* __restrict__ psi_state_ji,
    const int*    __restrict__ psi_dur_ji,
    double*                    delta,
    int*                       psi_state,
    int*                       psi_dur,
    int N, int D, int T, int t);