#pragma once

#include <cuda_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
    #define WARP_SIZE 64
#else
    #define WARP_SIZE 32
#endif

// ── Constant-memory dimensions — set once before any kernel launch ────────
void set_kernel_constants(int N, int D, int T);

// ── Initialization kernel ─────────────────────────────────────────────────
__global__ void kernel_initialization(
        const double* __restrict__ start_probs,
        const double* __restrict__ duration_probs,
        const double* __restrict__ duration_probs_linear,
        const double* __restrict__ emission_probs,
        const int*    __restrict__ obs_seq,
        double* delta, int* psi_dur,
        double* survival_probs);

// ── Induction kernel ──────────────────────────────────────────────────────
__global__ void kernel_induction(
    const double* __restrict__ em_t,      // emission_probs + obs_seq[t]*N (host-offset)
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    const double* __restrict__ delta,
    const double* __restrict__ em_cur,
    double*                    em_nxt,
    double* psi_state_ji,
    int*    psi_dur_ji,
    int tau, int t);

// ── Reduction kernel su i (argmax) ────────────────────────────────────────
__global__ void kernel_reduce_i(
    const double* __restrict__ psi_state_ji,
    const int*    __restrict__ psi_dur_ji,
    double*                    delta,
    int*                       psi_state,
    int*                       psi_dur,
    int t);

// ── Persistent kernel (induction + reduction) ─────────────────────────────
__global__ void kernel_persistent(
    const int*    __restrict__ obs_seq,
    const double* __restrict__ emission_probs,
    double*                    delta,
    const double* __restrict__ trans_mat,
    const double* __restrict__ duration_probs,
    double*                    em0,
    double*                    em1,
    double*                    psi_state_ji,
    int*                       psi_dur_ji,
    int*                       psi_state,
    int*                       psi_dur);

// ── Tail Adjustment kernel ────────────────────────────────────────────────
__global__ void kernel_tail_adjustment(
    const double* __restrict__ trans_mat,
    const double* __restrict__ survival_probs,
    const double* __restrict__ d_em_last,
    double*                    delta,
    double*                    psi_state_ji,
    int*                       psi_dur_ji);

__global__ void kernel_tail_reduce_i(
    const double* __restrict__ psi_state_ji,
    const int*    __restrict__ psi_dur_ji,
    double*                    delta,
    int*                       psi_state,
    int*                       psi_dur,
    int t);