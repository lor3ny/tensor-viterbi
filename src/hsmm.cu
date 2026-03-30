#include "hsmm.hpp"
#include "kernels.cuh"

#include <iostream>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define SMOOTHNESS 1e-30

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s: %s\n",                    \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ============================================================
// File-scope helpers (not part of the public API)
// ============================================================

static bool check_cooperative_launch(const void* kernel, int block_size, size_t shmem, int required_blocks)
{
    int supports_coop;
    cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, 0);

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);

    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, shmem);

    if (!supports_coop || required_blocks > max_blocks_per_sm * num_sm)
        return false;

    return true;
}

// forward declaration — defined after decode_tensor_viterbi_cuda
static void run_induction(
    double* d_delta, const double* d_AP,
    const double* d_emission_probs, const int* d_obs_seq,
    double** d_em,
    double* d_best_val_ji, int* d_best_d_ji,
    int* d_psi_state, int* d_psi_dur,
    int T, int N, int D);


static std::vector<double> compute_survival_probs(
    int N, int D,
    const std::vector<double>& duration_probs)
{
    // layout D×N: survival_probs[d*N + j]
    std::vector<double> survival_probs(D * N, -std::numeric_limits<double>::infinity());
    for (int j = 0; j < N; ++j) {
        double cum = -std::numeric_limits<double>::infinity();
        for (int d = D - 1; d >= 0; --d) {
            double lp = duration_probs[j * D + d];
            double m  = std::max(cum, lp);
            cum = (m == -std::numeric_limits<double>::infinity())
                ? lp
                : m + std::log(std::exp(cum - m) + std::exp(lp - m));
            survival_probs[d * N + j] = cum;
        }
    }
    return survival_probs;
}

static void tail_adjustment(
    std::vector<double>& delta,
    std::vector<int>&    psi_state,
    std::vector<int>&    psi_dur,
    const std::vector<double>& EMISSION_PROBS,
    const std::vector<double>& PAST_DELTA,
    const std::vector<double>& survival_probs,
    const std::vector<double>& trans_mat,
    int N, int D, int T)
{
    const int t   = T - 1;
    const int tau = std::min(t, D);

    for (int j = 0; j < N; ++j) {
        double best_val = -std::numeric_limits<double>::infinity();
        int    best_d   = 0;
        int    best_i   = 0;

        for (int d = 0; d < tau; ++d) {
            const double ep = EMISSION_PROBS[d * N + j];
            for (int i = 0; i < N; ++i) {
                const double ap_tail = trans_mat[j * N + i] + survival_probs[d * N + j];
                const double val = ep + PAST_DELTA[d * N + i] + ap_tail;
                if (val > best_val) {
                    best_val = val;
                    best_d   = d;
                    best_i   = i;
                }
            }
        }

        delta    [j * T + t] = best_val;
        psi_state[j * T + t] = best_i;
        psi_dur  [j * T + t] = best_d + 1;
    }
}

static std::vector<int> backtracking_termination(
    const std::vector<double>& delta,
    const std::vector<int>&    psi_state,
    const std::vector<int>&    psi_dur,
    int N, int T)
{
    std::vector<int> path(T, 0);

    int t = T - 1;
    int curr_state = 0;
    double best_val = delta[0 * T + t];
    for (int n = 1; n < N; ++n) {
        if (delta[n * T + t] > best_val) {
            best_val   = delta[n * T + t];
            curr_state = n;
        }
    }

    while (t >= 0) {
        int d       = psi_dur  [curr_state * T + t];
        int prev_s  = psi_state[curr_state * T + t];
        int start_t = t - d + 1;

        for (int k = start_t; k <= t; ++k)
            path[k] = curr_state;

        t          = t - d;
        curr_state = prev_s;
    }

    return path;
}

static void hsmm_to_gpu(
    int N, int O, int D, int T,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq,
    double*& d_trans_mat,
    double*& d_emission_probs,
    double*& d_start_probs,
    double*& d_duration_probs,
    int*&    d_obs_seq)
{
    CUDA_CHECK(cudaMalloc(&d_trans_mat,      N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_emission_probs, O * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_start_probs,    N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_duration_probs, N * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_obs_seq,        T     * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_trans_mat,      trans_mat.data(),      N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emission_probs, emission_probs.data(), O * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start_probs,    start_probs.data(),    N     * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_duration_probs, duration_probs.data(), N * D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_seq,        obs_seq.data(),        T     * sizeof(int),    cudaMemcpyHostToDevice));
}

static void hsmm_free_gpu(
    double*& d_trans_mat,
    double*& d_emission_probs,
    double*& d_start_probs,
    double*& d_duration_probs,
    int*&    d_obs_seq)
{
    CUDA_CHECK(cudaFree(d_trans_mat));
    CUDA_CHECK(cudaFree(d_emission_probs));
    CUDA_CHECK(cudaFree(d_start_probs));
    CUDA_CHECK(cudaFree(d_duration_probs));
    CUDA_CHECK(cudaFree(d_obs_seq));

    d_trans_mat      = nullptr;
    d_emission_probs = nullptr;
    d_start_probs    = nullptr;
    d_duration_probs = nullptr;
    d_obs_seq        = nullptr;
}


// ============================================================
// Public API — namespace hsmm (declared in hsmm.hpp)
// ============================================================
namespace hsmm {

std::vector<int> decode_tensor_viterbi(
    const int                  n_states,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq)
{
    const int N = n_states;
    const int T = static_cast<int>(obs_seq.size());
    const int D = static_cast<int>(duration_probs.size()) / N;

    std::vector<double> delta    (T * N, SMOOTHNESS);
    std::vector<int>    psi_state(T * N, 0);
    std::vector<int>    psi_dur  (T * N, 1);
    std::vector<double> EMISSION_PROBS(D * N, 0.0);

    const std::vector<double> survival_probs = compute_survival_probs(N, D, duration_probs);

    //* ── PHASE 1 — Initialization (0 <= t < D) ────────────────────────────── *//
    std::vector<double> PAST_DELTA(D * N);
    for (int d = 0; d < D; ++d)
        for (int n = 0; n < N; ++n)
            PAST_DELTA[d * N + n] = duration_probs[n * D + d] + start_probs[n];

    std::vector<double> CUM_EMISSION(D * N, 0.0);
    for (int d = 0; d < D; ++d) {
        int obs = obs_seq[d];
        for (int n = 0; n < N; ++n) {
            double prev = (d > 0) ? CUM_EMISSION[(d - 1) * N + n] : 0.0;
            CUM_EMISSION[d * N + n] = prev + emission_probs[obs * N + n];
        }
    }

    for (int t = 0; t < D; ++t)
        for (int n = 0; n < N; ++n) {
            delta  [n * T + t] = PAST_DELTA[t * N + n] + CUM_EMISSION[t * N + n];
            psi_dur[n * T + t] = t + 1;
        }

    // ── AP: (N x N x D) ──────────────────────────────────────────────────── //
    std::vector<double> AP(N * N * D);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < D; ++d)
                AP[j * N * D + i * D + d] = trans_mat[j * N + i] + duration_probs[j * D + d];

    //* ── PHASE 2 — Induction (t >= 1) ──────────────────────────────────────── *//
    for (int t = 1; t < T; ++t) {
        const int tau = std::min(t, D);

        for (int n = 0; n < N; ++n) {
            double new_em = emission_probs[obs_seq[t] * N + n];
            for (int d = tau - 1; d >= 1; --d)
                EMISSION_PROBS[d * N + n] = new_em + EMISSION_PROBS[(d - 1) * N + n];
            EMISSION_PROBS[0 * N + n] = new_em;
        }

        for (int d = 0; d < tau; ++d)
            for (int n = 0; n < N; ++n)
                PAST_DELTA[d * N + n] = delta[n * T + (t - 1 - d)];

        for (int j = 0; j < N; ++j) {
            double best_val = -std::numeric_limits<double>::infinity();
            int    best_d   = 0;
            int    best_i   = 0;

            for (int d = 0; d < tau; ++d) {
                const double ep = EMISSION_PROBS[d * N + j];
                for (int i = 0; i < N; ++i) {
                    const double val = ep + PAST_DELTA[d * N + i] + AP[j * N * D + i * D + d];
                    if (val > best_val) {
                        best_val = val;
                        best_d   = d;
                        best_i   = i;
                    }
                }
            }

            const bool update = (t >= D) || (best_val > delta[j * T + t]);
            if (update) {
                delta    [j * T + t] = best_val;
                psi_state[j * T + t] = best_i;
                psi_dur  [j * T + t] = best_d + 1;
            }
        }
    }

    //* ── TAIL ADJUSTMENT — t = T-1 ──────────────────────────────────────────── *//
    tail_adjustment(delta, psi_state, psi_dur,
                    EMISSION_PROBS, PAST_DELTA, survival_probs,
                    trans_mat, N, D, T);

    return backtracking_termination(delta, psi_state, psi_dur, N, T);
}


std::vector<int> decode_tensor_viterbi_cuda(
    const int                  n_states,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq)
{
    const int N = n_states;
    const int T = static_cast<int>(obs_seq.size());
    const int O = static_cast<int>(emission_probs.size()) / N;
    const int D = static_cast<int>(duration_probs.size()) / N;

    std::vector<double> delta    (T * N, -std::numeric_limits<double>::infinity());
    std::vector<int>    psi_state(T * N, 0);
    std::vector<int>    psi_dur  (T * N, 1);

    double* d_trans_mat      = nullptr;
    double* d_emission_probs = nullptr;
    double* d_start_probs    = nullptr;
    double* d_duration_probs = nullptr;
    int*    d_obs_seq        = nullptr;
    hsmm_to_gpu(N, O, D, T,
                trans_mat, emission_probs, start_probs, duration_probs, obs_seq,
                d_trans_mat, d_emission_probs, d_start_probs, d_duration_probs, d_obs_seq);

    double* d_delta          = nullptr;
    double* d_AP             = nullptr;
    double* d_emissions      = nullptr;
    double* d_best_state_ji  = nullptr;
    int*    d_best_d_ji      = nullptr;
    int*    d_psi_state      = nullptr;
    int*    d_psi_dur        = nullptr;
    CUDA_CHECK(cudaMalloc(&d_delta,         N * T     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AP,            D * N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_emissions,     D * N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_best_state_ji, N * N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_best_d_ji,     N * N     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_psi_state,     N * T     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_psi_dur,       N * T     * sizeof(int)));

    double* d_em[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_em[0], D * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_em[1], D * N * sizeof(double)));

    //* ── PHASE 1 — Initialization & AP ──────────────────────────────────────── *//
    int bs_init = 1;
    while (bs_init < D) bs_init <<= 1;
    const int    num_warps = bs_init / 32;
    const size_t sm_init   = (bs_init + num_warps) * sizeof(double);
    kernel_initialization<<<dim3(N, N), dim3(bs_init), sm_init>>>(
        d_start_probs, d_duration_probs,
        d_emission_probs, d_obs_seq,
        d_delta, d_psi_dur,
        d_trans_mat, d_AP,
        N, D, T);
    CUDA_CHECK(cudaGetLastError());

    //* ── PHASE 2 — Induction ─────────────────────────────────────────────────── *//
    run_induction(
        d_delta, d_AP,
        d_emission_probs, d_obs_seq,
        d_em,
        d_best_state_ji, d_best_d_ji,
        d_psi_state, d_psi_dur,
        T, N, D);

    CUDA_CHECK(cudaMemcpy(delta.data(),     d_delta,     N * T * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(psi_state.data(), d_psi_state, N * T * sizeof(int),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(psi_dur.data(),   d_psi_dur,   N * T * sizeof(int),    cudaMemcpyDeviceToHost));

    hsmm_free_gpu(d_trans_mat, d_emission_probs, d_start_probs, d_duration_probs, d_obs_seq);
    CUDA_CHECK(cudaFree(d_delta));
    CUDA_CHECK(cudaFree(d_AP));
    CUDA_CHECK(cudaFree(d_emissions));
    CUDA_CHECK(cudaFree(d_best_state_ji));
    CUDA_CHECK(cudaFree(d_best_d_ji));
    CUDA_CHECK(cudaFree(d_psi_state));
    CUDA_CHECK(cudaFree(d_psi_dur));
    CUDA_CHECK(cudaFree(d_em[0]));
    CUDA_CHECK(cudaFree(d_em[1]));

    return backtracking_termination(delta, psi_state, psi_dur, N, T);
}

} // namespace hsmm


// ============================================================
// run_induction (defined here — after decode_tensor_viterbi_cuda
// because it is used only by that function)
// ============================================================
static void run_induction(
    double* d_delta, const double* d_AP,
    const double* d_emission_probs, const int* d_obs_seq,
    double** d_em,
    double* d_best_state_ji, int* d_best_d_ji,
    int* d_psi_state, int* d_psi_dur,
    int T, int N, int D)
{
    int block_size = 1;
    while (block_size < D) block_size <<= 1;
    const size_t shmem = block_size * (sizeof(double) + sizeof(int));

    bool use_persistent = check_cooperative_launch(
        (void*)kernel_persistent, block_size, shmem, N * N);

    int cur = 0;

    if (use_persistent) {
        void* args[] = {
            &d_obs_seq, &d_emission_probs, &d_delta, &d_AP,
            &d_em[0], &d_em[1],
            &d_best_state_ji, &d_best_d_ji,
            &d_psi_state, &d_psi_dur,
            &N, &D, &T
        };
        cudaLaunchCooperativeKernel(
            (void*)kernel_persistent,
            dim3(N, N), dim3(block_size),
            args, shmem);
        CUDA_CHECK(cudaGetLastError());
    } else {
        for (int t = 1; t < T; ++t) {
            const int tau = std::min(t, D);
            const int nxt = 1 - cur;

            int bs = 1;
            while (bs < tau) bs <<= 1;
            const size_t sm = bs * (sizeof(double) + sizeof(int));

            kernel_induction<<<dim3(N, N), dim3(bs), sm>>>(
                d_obs_seq, d_emission_probs, d_delta, d_AP,
                d_em[cur], d_em[nxt],
                d_best_state_ji, d_best_d_ji, N, D, T, tau, t);
            CUDA_CHECK(cudaGetLastError());

            kernel_reduce_i<<<1, N>>>(
                d_best_state_ji, d_best_d_ji,
                d_delta, d_psi_state, d_psi_dur,
                N, D, T, t);
            CUDA_CHECK(cudaGetLastError());

            cur = nxt;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}
