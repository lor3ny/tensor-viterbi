#include "hsmm.hpp"
#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <omp.h>

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
    const double* d_emission_probs, const double* duration_probs_linear,
    const int* d_obs_seq,
    double** d_em,
    double* d_best_val_ji, int* d_best_d_ji,
    int* d_psi_state, int* d_psi_dur,
    int T, int N, int D);


static std::vector<double> compute_survival_probs(
    int N, int D,
    const std::vector<double>& duration_probs)
{
    // TODO: compute survival in linear space using duration_probs_linear
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
    const std::vector<double>& delta_t,
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
                const double val = ep + delta_t[(t - 1 - d) * N + i] + ap_tail;
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
    const std::vector<double>& duration_probs_linear,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq,
    double*& d_trans_mat,
    double*& d_emission_probs,
    double*& d_duration_probs_linear,
    double*& d_start_probs,
    double*& d_duration_probs,
    int*&    d_obs_seq)
{
    CUDA_CHECK(cudaMalloc(&d_trans_mat,      N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_emission_probs, O * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_duration_probs_linear, O * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_start_probs,    N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_duration_probs, N * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_obs_seq,        T     * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_trans_mat,      trans_mat.data(),      N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emission_probs, emission_probs.data(), O * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_duration_probs_linear, duration_probs_linear.data(), O * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start_probs,    start_probs.data(),    N     * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_duration_probs, duration_probs.data(), N * D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_seq,        obs_seq.data(),        T     * sizeof(int),    cudaMemcpyHostToDevice));
}

static void hsmm_free_gpu(
    double*& d_trans_mat,
    double*& d_emission_probs,
    double*& d_duration_probs_linear,
    double*& d_start_probs,
    double*& d_duration_probs,
    int*&    d_obs_seq)
{
    CUDA_CHECK(cudaFree(d_trans_mat));
    CUDA_CHECK(cudaFree(d_emission_probs));
    CUDA_CHECK(cudaFree(d_duration_probs_linear));
    CUDA_CHECK(cudaFree(d_start_probs));
    CUDA_CHECK(cudaFree(d_duration_probs));
    CUDA_CHECK(cudaFree(d_obs_seq));

    d_trans_mat      = nullptr;
    d_emission_probs = nullptr;
    d_duration_probs_linear = nullptr;
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
    const std::vector<double>& duration_probs_linear,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq)
{
    const int N = n_states;
    const int T = static_cast<int>(obs_seq.size());
    const int D = static_cast<int>(duration_probs.size()) / N;
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    // State-major layout (n*T+t) — required by backtracking_termination
    std::vector<double> delta    (N * T, NEG_INF);
    std::vector<int>    psi_state(N * T, 0);
    std::vector<int>    psi_dur  (N * T, 1);
    // Time-major layout (t*N+n) — used by BRICK UPDATE reads and condition guard
    std::vector<double> delta_t  (T * N, NEG_INF);


    //* Brick Compuring: AP = TRANS_MAT + DURATION_PROBS
    std::vector<double> AP(D * N * N);
    for (int d = 0; d < D; ++d)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                AP[d * N * N + j * N + i] = trans_mat[j * N + i] + duration_probs[j * D + d];

    const std::vector<double> survival_probs = compute_survival_probs(N, D, duration_probs);


    std::vector<double> EMISSION_PROBS(D * N, 0.0);
    std::vector<double> EMISSION_CACHE(D * N, 0.0);
    std::vector<double> RESULT_B      (D * N * N);

    //* ── PHASE 1 — Initialization 0 <= t < D ──────────────────────────────── *//
    // EMISSION_PROBS[d, j] = cumsum(emission_probs[obs[0:d+1], :], axis=0)[d, j]
    for (int d = 0; d < D; ++d) {
        const double* em_d = &emission_probs[obs_seq[d] * N];
        for (int j = 0; j < N; ++j) {
            const double prev = (d > 0) ? EMISSION_PROBS[(d - 1) * N + j] : 0.0;
            EMISSION_PROBS[d * N + j] = prev + em_d[j];
        }
    }
    // delta[t, j] = duration_probs[j, t] + start_probs[j] + EMISSION_PROBS[t, j]
    for (int d = 0; d < D; ++d) {
        for (int j = 0; j < N; ++j) {
            const double v = duration_probs[j * D + d] + start_probs[j] + EMISSION_PROBS[d * N + j];
            delta_t[d * N + j] = v;
            psi_dur[j * T + d] = d + 1;
        }
    }

    //* ── PHASE 2 — Induction t >= 1 ────────────────────────────────────────── *//
    for (int t = 1; t < T; ++t) {
        const int tau      = std::min(t, D);
        const double* em_t = &emission_probs[obs_seq[t] * N];

        //* EMISSION PROBS COMPUTATION
        if (t > D) {
            // Rolling cache: EMISSION_PROBS[d, j] = EMISSION_CACHE[d, j] + em_t[j]
            // Matches Python: np.add(EMISSION_CACHE, _probs_t, out=EMISSION_PROBS)
            for (int d = 0; d < D; ++d) {
                const double* c = &EMISSION_CACHE[d * N];
                double*       p = &EMISSION_PROBS[d * N];
                for (int j = 0; j < N; ++j)
                    p[j] = c[j] + em_t[j];
            }
            // Shift cache: EMISSION_CACHE[1:] = EMISSION_PROBS[0:D-1]
            // Matches Python: EMISSION_CACHE[1:, :] = EMISSION_PROBS[: D - 1, :]
            std::memcpy(EMISSION_CACHE.data() + N, EMISSION_PROBS.data(),
                        (D - 1) * N * sizeof(double));
        } else {
            // Recompute from scratch for t <= D (reversed cumsum over obs[t..t-tau+1])
            // Matches Python: cumsum(flip(emission_probs[segment, :], axis=0), axis=0)
            for (int j = 0; j < N; ++j)
                EMISSION_PROBS[j] = em_t[j];
            for (int d = 1; d < tau; ++d) {
                const double* em_back = &emission_probs[obs_seq[t - d] * N];
                const double* prev_p  = &EMISSION_PROBS[(d - 1) * N];
                double*       cur_p   = &EMISSION_PROBS[d * N];
                for (int j = 0; j < N; ++j)
                    cur_p[j] = prev_p[j] + em_back[j];
            }
            // Initialise rolling cache once t == D
            // Matches Python: EMISSION_CACHE[1:, :] = cum_emission[: D - 1, :]
            if (t == D)
                std::memcpy(EMISSION_CACHE.data() + N, EMISSION_PROBS.data(),
                            (D - 1) * N * sizeof(double));
        }

        //* PAST DELTA IS EXTRACTED DURING UPDATE TO AVOID REDUNDANT MEMCPY
        //* BRICK UPDATE: RESULT_B[d, j, i] = EMISSION_PROBS + PAST_DELTA + AP
        for (int d = 0; d < tau; ++d) {
            const double* pd    = &delta_t[(t - 1 - d) * N];
            const double* ap_d  = &AP[d * N * N];
            double*       rb_d  = &RESULT_B[d * N * N];
            for (int j = 0; j < N; ++j) {
                const double  ep    = EMISSION_PROBS[d * N + j];
                const double* ap_dj = &ap_d[j * N];
                double*       rb_dj = &rb_d[j * N];
                for (int i = 0; i < N; ++i)
                    rb_dj[i] = ep + pd[i] + ap_dj[i];
            }
        }

        //* ARGMAX
        for (int j = 0; j < N; ++j) {
            double best_val = NEG_INF;
            int    best_d   = 0;
            int    best_i   = 0;

            for (int d = 0; d < tau; ++d) {
                const double* rb_dj = &RESULT_B[d * N * N + j * N];
                for (int i = 0; i < N; ++i) {
                    if (rb_dj[i] > best_val) {
                        best_val = rb_dj[i];
                        best_d   = d;
                        best_i   = i;
                    }
                }
            }

            const bool update = (t >= D) || (best_val > delta_t[t * N + j]);
            if (update) {
                delta_t  [t * N + j] = best_val;
                psi_state[j * T + t] = best_i;
                psi_dur  [j * T + t] = best_d + 1;
            }
        }
    }

    //* ── TAIL ADJUSTMENT — t = T-1 ──────────────────────────────────────────── *//
    tail_adjustment(delta, psi_state, psi_dur,
                    EMISSION_PROBS, delta_t, survival_probs,
                    trans_mat, N, D, T);

    return backtracking_termination(delta, psi_state, psi_dur, N, T);
}


std::vector<int> decode_tensor_viterbi_omp(
    const int                  n_states,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& duration_probs_linear,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq)
{
    const int N = n_states;
    const int T = static_cast<int>(obs_seq.size());
    const int D = static_cast<int>(duration_probs.size()) / N;
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    std::vector<double> delta    (N * T, NEG_INF);
    std::vector<int>    psi_state(N * T, 0);
    std::vector<int>    psi_dur  (N * T, 1);
    std::vector<double> delta_t  (T * N, NEG_INF);

    //* ── AP = TRANS_MAT + DURATION_PROBS ───────────────────────────────────── *//
    // d, j, i are fully independent — collapse all three loops.
    std::vector<double> AP(D * N * N);
    #pragma omp parallel for collapse(3) schedule(static)
    for (int d = 0; d < D; ++d)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                AP[d * N * N + j * N + i] = trans_mat[j * N + i] + duration_probs[j * D + d];

    //* ── Survival Probs ─────────────────────────────────────────────────────── *//
    // The d loop carries a running sum per state j, but j iterations are
    // independent of each other — parallelize over j only.
    std::vector<double> survival_probs(D * N, NEG_INF);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) {
        double cum = NEG_INF;
        for (int d = D - 1; d >= 0; --d) {
            const double lp = duration_probs[j * D + d];
            const double m  = std::max(cum, lp);
            cum = (m == NEG_INF) ? lp : m + std::log(std::exp(cum - m) + std::exp(lp - m));
            survival_probs[d * N + j] = cum;
        }
    }

    std::vector<double> EMISSION_PROBS(D * N, 0.0);
    std::vector<double> EMISSION_CACHE(D * N, 0.0);
    std::vector<double> RESULT_B      (D * N * N);

    //* ── PHASE 1 — Initialization 0 <= t < D ──────────────────────────────── *//
    // The d loop is sequential (each row reads from d-1).
    // The j loop inside is fully independent — vectorize and parallelize it.
    for (int d = 0; d < D; ++d) {
        const double* em_d = &emission_probs[obs_seq[d] * N];
        #pragma omp parallel for simd schedule(static)
        for (int j = 0; j < N; ++j) {
            const double prev = (d > 0) ? EMISSION_PROBS[(d - 1) * N + j] : 0.0;
            EMISSION_PROBS[d * N + j] = prev + em_d[j];
        }
    }
    // d and j are fully independent here — collapse both loops.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int d = 0; d < D; ++d)
        for (int j = 0; j < N; ++j) {
            delta_t[d * N + j] = duration_probs[j * D + d] + start_probs[j] + EMISSION_PROBS[d * N + j];
            psi_dur[j * T + d] = d + 1;
        }

    //* ── PHASE 2 — Induction t >= 1 ────────────────────────────────────────── *//
    // The t loop is sequential (each step reads delta_t written by t-1).
    // We use a single persistent omp parallel region to avoid the cost of
    // spawning/joining a thread team on every t step.
    // Sequential bookkeeping uses omp single (implicit barrier included).
    // Parallel inner loops use omp for.  The implicit barrier at the end of
    // each omp for / omp single enforces the data-flow ordering within each step.
    int    tau_sh  = 0;
    const double* em_t_sh = nullptr;

    #pragma omp parallel default(shared)
    {
        for (int t = 1; t < T; ++t) {

            // Sequential setup — one thread computes, barrier broadcasts.
            #pragma omp single
            {
                tau_sh  = std::min(t, D);
                em_t_sh = &emission_probs[obs_seq[t] * N];
            }
            // After omp single's implicit barrier all threads see tau_sh / em_t_sh.
            const int     tau  = tau_sh;
            const double* em_t = em_t_sh;

            // ── Emission probs ────────────────────────────────────────────── //
            if (t > D) {
                // Rolling update: EP[d,j] = cache[d,j] + em_t[j]
                // d and j are fully independent.
                #pragma omp for collapse(2) schedule(static)
                for (int d = 0; d < D; ++d)
                    for (int j = 0; j < N; ++j)
                        EMISSION_PROBS[d * N + j] = EMISSION_CACHE[d * N + j] + em_t[j];
                // Cache shift — sequential, runs after EP is fully written.
                #pragma omp single
                std::memcpy(EMISSION_CACHE.data() + N, EMISSION_PROBS.data(),
                            (D - 1) * N * sizeof(double));
            } else {
                // Scratch build (reversed cumsum). The d loop is sequential
                // (each row depends on d-1); the j loop inside is independent.
                // d = 0: assign
                #pragma omp for schedule(static)
                for (int j = 0; j < N; ++j)
                    EMISSION_PROBS[j] = em_t[j];
                // d >= 1: accumulate — implicit barrier after each omp for
                // ensures row d-1 is complete before row d reads it.
                for (int d = 1; d < tau; ++d) {
                    const double* em_back = &emission_probs[obs_seq[t - d] * N];
                    const double* prev_p  = &EMISSION_PROBS[(d - 1) * N];
                    double*       cur_p   = &EMISSION_PROBS[d * N];
                    #pragma omp for schedule(static)
                    for (int j = 0; j < N; ++j)
                        cur_p[j] = prev_p[j] + em_back[j];
                }
                // Initialise rolling cache once t == D (sequential, one thread).
                #pragma omp single
                {
                    if (t == D)
                        std::memcpy(EMISSION_CACHE.data() + N, EMISSION_PROBS.data(),
                                    (D - 1) * N * sizeof(double));
                }
            }

            // ── Brick Update: RB[d,j,i] = EP[d,j] + delta_t[t-1-d,i] + AP[d,j,i] ── //
            // d, j, i are fully independent — collapse all three loops.
            #pragma omp for collapse(3) schedule(static)
            for (int d = 0; d < tau; ++d)
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < N; ++i)
                        RESULT_B[d * N * N + j * N + i] =
                            EMISSION_PROBS[d * N + j]
                            + delta_t[(t - 1 - d) * N + i]
                            + AP[d * N * N + j * N + i];

            // ── Argmax over (d, i) for each j ────────────────────────────── //
            // Each j is independent — distribute j across threads.
            // The implicit barrier after this omp for guarantees delta_t[t*N+j]
            // is fully written before the next t iteration reads it.
            #pragma omp for schedule(static)
            for (int j = 0; j < N; ++j) {
                double best_val = NEG_INF;
                int    best_d   = 0;
                int    best_i   = 0;

                for (int d = 0; d < tau; ++d) {
                    const double* rb_dj = &RESULT_B[d * N * N + j * N];
                    for (int i = 0; i < N; ++i) {
                        if (rb_dj[i] > best_val) {
                            best_val = rb_dj[i];
                            best_d   = d;
                            best_i   = i;
                        }
                    }
                }

                const bool update = (t >= D) || (best_val > delta_t[t * N + j]);
                if (update) {
                    delta_t  [t * N + j] = best_val;
                    psi_state[j * T + t] = best_i;
                    psi_dur  [j * T + t] = best_d + 1;
                }
            }
        } // for t
    } // omp parallel

    //* ── TAIL ADJUSTMENT — t = T-1 ──────────────────────────────────────────── *//
    // j iterations are fully independent — parallelize directly.
    {
        const int t   = T - 1;
        const int tau = std::min(t, D);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; ++j) {
            double best_val = NEG_INF;
            int    best_d   = 0;
            int    best_i   = 0;

            for (int d = 0; d < tau; ++d) {
                const double ep = EMISSION_PROBS[d * N + j];
                for (int i = 0; i < N; ++i) {
                    const double val = ep
                        + delta_t[(t - 1 - d) * N + i]
                        + trans_mat[j * N + i]
                        + survival_probs[d * N + j];
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

    return backtracking_termination(delta, psi_state, psi_dur, N, T);
}


std::vector<int> decode_tensor_viterbi_cuda(
    const int                  n_states,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& duration_probs_linear,
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
    double* d_duration_probs_linear = nullptr;
    double* d_start_probs    = nullptr;
    double* d_duration_probs = nullptr;
    int*    d_obs_seq        = nullptr;
    hsmm_to_gpu(N, O, D, T,
                trans_mat, emission_probs, duration_probs_linear, start_probs, duration_probs, obs_seq,
                d_trans_mat, d_emission_probs, d_duration_probs_linear, d_start_probs, d_duration_probs, d_obs_seq);

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
    // TODO: Add survival probs computation
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
        d_emission_probs, d_duration_probs_linear, 
        d_obs_seq,
        d_em,
        d_best_state_ji, d_best_d_ji,
        d_psi_state, d_psi_dur,
        T, N, D);
    
    //* ── Tail Adjustment ─────────────────────────────────────────────────────── *//
    // TODO: Tail Adjustment kernel

    CUDA_CHECK(cudaMemcpy(delta.data(),     d_delta,     N * T * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(psi_state.data(), d_psi_state, N * T * sizeof(int),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(psi_dur.data(),   d_psi_dur,   N * T * sizeof(int),    cudaMemcpyDeviceToHost));

    hsmm_free_gpu(d_trans_mat, d_emission_probs, d_duration_probs_linear, d_start_probs, d_duration_probs, d_obs_seq);
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
    const double* d_emission_probs, const double* duration_probs_linear,
    const int* d_obs_seq,
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
