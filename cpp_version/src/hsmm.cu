#include "hsmm.hpp"
#include "kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

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



// ── cuda_launch_timed ─────────────────────────────────────────────────────── //
// Wrapper che lancia un kernel e restituisce il tempo di esecuzione in secondi.
// Uso:
//   float ms = cuda_launch_timed("nome", [&](){
//       my_kernel<<<grid, block>>>(args...);
//   });
template <typename KernelFunc>
float cuda_launch_timed(const std::string& label, KernelFunc&& launch)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    std::forward<KernelFunc>(launch)();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "[CUDA] " << label << ": "
              << std::fixed << std::setprecision(6)
              << ms / 1000.0f << " s\n";
    std::cout.flush();

    return ms;
}

HSMM::HSMM(const std::vector<std::string>&  states,
           const std::vector<std::string>&  emissions,
           const std::vector<double>&       trans_mat,
           const std::vector<double>&       emission_probs,
           const std::vector<double>&       start_probs,
           const std::vector<double>&       duration_probs)
{
    states_           = states;
    emissions_        = emissions;
    trans_mat_        = trans_mat;
    emission_probs_   = emission_probs;
    start_probs_      = start_probs;
    duration_probs_   = duration_probs;

    N_ = static_cast<int>(states_.size());
    O_ = static_cast<int>(emissions_.size());
    D_ = static_cast<int>(duration_probs.size() / N_);
}

HSMM::HSMM(const std::string& json_data_path)
{
    std::ifstream file(json_data_path);
    if (!file.is_open())
        throw std::runtime_error("load_sleep_model: cannot open \"" + json_data_path + "\"");

    nlohmann::json cfg;
    file >> cfg;

    const int O = cfg["n_bins"].get<int>();

    // -------- STATES --------
    std::vector<std::string> states;
    for (const auto& s : cfg["states"])
        states.push_back(s["name"].get<std::string>());

    const int N = static_cast<int>(states.size());

    // -------- EMISSIONS --------
    std::vector<std::string> emissions(O);
    for (int o = 0; o < O; ++o)
        emissions[o] = std::to_string(o);

    // -------- OBS SEQ --------
    const int T = cfg["obs_seq"].size();

    const auto& raw_obs = cfg["obs_seq"];
    std::vector<int> obs_seq(raw_obs.size());
    for (std::size_t t = 0; t < raw_obs.size(); ++t)
        obs_seq[t] = raw_obs[t].get<int>() - 1;

    // -------- TRANS --------
    std::vector<double> trans_mat(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            trans_mat[j*N + i] = cfg["trans_mat"][i][j].get<double>();

    // -------- EMISSION PROBS --------
    std::vector<double> emission_probs(O * N, 0.0);
    for (int s = 0; s < N; ++s) {
        const auto& ep = cfg["states"][s]["emission_probs"];
        for (int o = 0; o < O; ++o)
            emission_probs[o * N + s] = ep[o].get<double>();
    }

    // -------- START --------
    std::vector<double> start_probs = cfg["pi"].get<std::vector<double>>();

    // -------- DURATIONS --------
    const int D = cfg["states"][0]["duration_probs"].size();

    std::vector<double> duration_probs;
    duration_probs.reserve(N * D);

    for (int s = 0; s < N; ++s) {
        const auto& dp = cfg["states"][s]["duration_probs"];
        for (const auto& val : dp)
            duration_probs.push_back(val.get<double>());
    }

    states_           = states;
    emissions_        = emissions;
    trans_mat_        = trans_mat;
    emission_probs_   = emission_probs;
    start_probs_      = start_probs;
    duration_probs_   = duration_probs;

    N_ = N;
    O_ = O;
    D_ = D;
    T_ = T;

    obs_seq_ = obs_seq;
}

void HSMM::print() const {
    const int N = num_states();
    const int O = num_emissions();
    const int D = max_duration();
    const int T = obs_length();

    std::cout << "===== HSMM MODEL =====\n";

    // Dimensioni
    std::cout << "\nDimensions:\n";
    std::cout << "  N (states)    = " << N << "\n";
    std::cout << "  O (emissions) = " << O << "\n";
    std::cout << "  D (max dur)   = " << D << "\n";
    std::cout << "  T (obs len)   = " << T << "\n";

    // Stati
    std::cout << "\nStates (" << N << "):\n";
    for (int i = 0; i < N; ++i)
        std::cout << "  [" << i << "] " << states_[i] << "\n";

    // Emissioni
    std::cout << "\nEmissions (" << O << "):\n";
    for (int o = 0; o < O; ++o)
        std::cout << "  [" << o << "] " << emissions_[o] << "\n";

    // Start probabilities
    std::cout << "\nStart probabilities (pi):\n";
    for (int i = 0; i < N; ++i)
        std::cout << "  " << states_[i] << ": " << start_probs_[i] << "\n";

    // Transition matrix
    std::cout << "\nTransition matrix (N x N):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(8) << trans_mat_[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Emission probabilities
    std::cout << "\nEmission probabilities (O x N):\n";
    for (int o = 0; o < O; ++o) {
        std::cout << "Obs " << o << ": ";
        for (int s = 0; s < N; ++s) {
            std::cout << std::setw(8) << emission_probs_[o * N + s] << " ";
        }
        std::cout << "\n";
    }


    // Duration probabilities
    std::cout << "\nDuration probabilities:\n";
    for (int s = 0; s < N; ++s) {
        std::cout << "State " << states_[s] << ": [ ";

        for (int d = 0; d < D; ++d) {
            int idx = s * D + d;
            std::cout << duration_probs_[idx] << " ";
        }

        std::cout << "]\n";
    }

    std::cout << "\n======================\n";
}

void HSMM::to_log_space()
{
    for (double& v : trans_mat_)
        v = std::log(v + SMOOTHNESS);

    for (double& v : emission_probs_)
        v = std::log(v + SMOOTHNESS);

    for (double& v : start_probs_)
        v = std::log(v + SMOOTHNESS);

    for (double& v : duration_probs_)
        v = std::log(v + SMOOTHNESS);
}

//! ---------------------------------------------------------------------- 
//! Viterbi Algorithm
//! ----------------------------------------------------------------------

void HSMM::find_t_maxs(const std::vector<double>& Sjid,
                        std::vector<double>&        max_vals,
                        std::vector<int>&           max_states,
                        std::vector<int>&           max_durs) const
{
    // TODO: for each destination state j (0..N_-1):
}
 
std::vector<int> HSMM::backtracking_termination(const std::vector<double>& delta,
                                                 const std::vector<int>&    psi_state,
                                                 const std::vector<int>&    psi_dur,
                                                 int                        T) const
{
    std::vector<int> path(T, 0);

    // ── Termination — trova il miglior stato finale ───────────────────────── //
    int t = T - 1;
    int curr_state = 0;
    double best_val = delta[t * N_ + 0];
    for (int n = 1; n < N_; ++n) {
        if (delta[t * N_ + n] > best_val) {
            best_val   = delta[t * N_ + n];
            curr_state = n;
        }
    }

    // ── Backtracking ──────────────────────────────────────────────────────── //
    while (t > 0) {
        int d      = psi_dur  [t * N_ + curr_state];
        int prev_s = psi_state[t * N_ + curr_state];

        // Riempi il segmento [start_t, t]
        int start_t = t - d + 1;
        for (int k = start_t; k <= t; ++k)
            path[k] = curr_state;

        // Vai indietro
        t          = t - d;
        curr_state = prev_s;
    }

    return path;
}
 
std::vector<int> HSMM::decoding_vanilla_viterbi()
{
    // TODO: implement tensor Viterbi
    return {};
}

void HSMM::hsmm_to_gpu(
    double*& d_trans_mat,
    double*& d_emission_probs,
    double*& d_start_probs,
    double*& d_duration_probs,
    int*&    d_obs_seq)
{
    int T = static_cast<int>(obs_seq_.size());

    CUDA_CHECK(cudaMalloc(&d_trans_mat,      N_ * N_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_emission_probs, O_ * N_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_start_probs,    N_      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_duration_probs, N_ * D_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_obs_seq,        T_      * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_trans_mat,      trans_mat_.data(),      N_ * N_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emission_probs, emission_probs_.data(), O_ * N_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start_probs,    start_probs_.data(),    N_      * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_duration_probs, duration_probs_.data(), N_ * D_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_seq,        obs_seq_.data(),        T_      * sizeof(int),    cudaMemcpyHostToDevice));
}

void HSMM::hsmm_free_gpu(
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

std::vector<int> HSMM::decoding_tensor_viterbi()
{
    // CPU-side data structures
    const int T = T_;
    const int N = N_;
    const int D = D_;

    std::vector<double> delta(T * N, SMOOTHNESS);       // delta[t*N + n]
    std::vector<int>    delta_state(T * N, 0);
    std::vector<int>    delta_dur  (T * N, 0);

    // GPU memory allocation
    double* d_trans_mat      = nullptr;
    double* d_emission_probs = nullptr;
    double* d_start_probs    = nullptr;
    double* d_duration_probs = nullptr;
    int*    d_obs_seq        = nullptr;
    
    double* d_delta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_delta, N * T * sizeof(double)));
    
    // Load data to GPU
    hsmm_to_gpu(d_trans_mat, d_emission_probs, d_start_probs, d_duration_probs, d_obs_seq);


    // ── PHASE 1 — Initialization (0 <= t < D) ────────────────────────────── //
    // Python: PAST_DELTA[d, n] = duration_probs[d, n] + start_probs[n]
    // C++:    duration_probs_[n*D + d]
    std::vector<double> PAST_DELTA(D * N);
    for (int d = 0; d < D; ++d)
        for (int n = 0; n < N; ++n)
            PAST_DELTA[d*N + n] = duration_probs_[n*D + d] + start_probs_[n];

    // Python: cum_emission[t, n] = cumsum(emission_probs[obs_seq[:D], :], axis=0)
    // C++:    emission_probs_[o*N + n]
    std::vector<double> CUM_EMISSION(D * N, 0.0);
    for (int t = 0; t < D; ++t) {
        int obs = obs_seq_[t];
        for (int n = 0; n < N; ++n) {
            double prev = (t > 0) ? CUM_EMISSION[(t-1)*N + n] : 0.0;
            CUM_EMISSION[t*N + n] = prev + emission_probs_[obs*N + n];
        }
    }

    // Python: delta[0:D] = PAST_DELTA + EMISSION_PROBS
    for (int t = 0; t < D; ++t)
        for (int n = 0; n < N; ++n)
            delta[t*N + n] = PAST_DELTA[t*N + n] + CUM_EMISSION[t*N + n];

    // ── AP: (D x N x N) ──────────────────────────────────────────────────── //
    // Python: AP[d, i, j] = trans_mat[i, j] + duration_probs[d, i]

    double* d_AP = nullptr;
    CUDA_CHECK(cudaMalloc(&d_AP, D * N * N * sizeof(double)));

    // ── AP ───────────────────────────────────────────────────────────────────── //
    // limit: N*N <= 1024 (max threads per block) -> N <= 32
    dim3 block(N, N, 1);   // N×N thread per blocco
    dim3 grid(D, 1, 1);     // D blocchi
    
    cuda_launch_timed("kernel_compute_AP", [&](){
        kernel_compute_AP<<<grid, block>>>(d_trans_mat, d_duration_probs, d_AP, N, D);
    });

    std::vector<double> AP(D * N * N);
    CUDA_CHECK(cudaMemcpy(AP.data(), d_AP, D * N * N * sizeof(double), cudaMemcpyDeviceToHost));

    // ── Salva AP su file ─────────────────────────────────────────────────────── //
    std::ofstream ap_file("../data/ap_cuda.bin", std::ios::binary);
    ap_file.write(reinterpret_cast<const char*>(AP.data()), D * N * N * sizeof(double));
    ap_file.close();
    std::cout << "[DEBUG] AP salvato in ap_cuda.bin\n";

    // ! temporaneo: copia delta su GPU (per ora delta è solo CPU, ma in futuro sarà direttamente in global)
    CUDA_CHECK(cudaMemcpy(d_delta, delta.data(), N * T * sizeof(double), cudaMemcpyHostToDevice));

    // ── PHASE 2 — Induction (t >= 1) ─────────────────────────────────────────── //
    // AP[d*N*N + j*N + i] = trans_mat[j,i] + duration_probs[d,j]   (già calcolato)
    //
    // Per ogni stato corrente j, troviamo il miglior (d, i_prev) tale che:
    //   score(j, d, i) = EMISSION_PROBS[d,j] + PAST_DELTA[d,i] + AP[d,j,i]
    //                  = emis_cum(j, d+1 passi fino a t)
    //                  + delta[t-1-d, i]
    //                  + trans_mat[j,i] + dur_prob[d,j]

    std::vector<double> EMISSION_PROBS(D * N, 0.0);

    for (int t = 1; t < T; ++t) {

        const int past_len = std::min(t, D);   // d valido: 0 .. past_len-1

        // TODO: Calcolare Emission su GPU
        // ── 1. Emission Tensor  ("Produttoria") ─────────────────────────────── //
        // EMISSION_PROBS[d*N + n] = Σ_{k=0}^{d} emission_probs[obs_seq[t-k], n]
        // (cumsum sul segmento rovesciato che termina in t)
        for (int n = 0; n < N; ++n) {
            double cum = 0.0;
            for (int d = 0; d < past_len; ++d) {
                int obs = static_cast<int>(obs_seq_[t - d]);
                cum += emission_probs_[obs * N + n];
                EMISSION_PROBS[d * N + n] = cum;
            }
        }

        /* //TODO: PAST_DELTA * AP * EMISSIONS su GPU
         * 1. PAST_DELTA: //! delta direttamente in global, nessuna copia
         * 2. AP: //! in constant memory
         * 3. EMISSION_PROBS: //? per ora cpu
        */ 

        // ── 2. Past Delta Tensor ────────────────────────────────────────────── //
        // PAST_DELTA[d*N + n] = delta[(t-1-d)*N + n]   (finestra rovesciata)
        for (int d = 0; d < past_len; ++d)
            for (int n = 0; n < N; ++n)
                PAST_DELTA[d * N + n] = delta[(t - 1 - d) * N + n];



        // TODO: fuse argmax with induction kernel
        // ── 3. Argmax su (d, i_prev) per ogni stato corrente j ─────────────── //
        for (int j = 0; j < N; ++j) {
            double best_val = -std::numeric_limits<double>::infinity();
            int    best_d   = 0;
            int    best_i   = 0;

            for (int d = 0; d < past_len; ++d) {
                const double ep = EMISSION_PROBS[d * N + j];
                for (int i = 0; i < N; ++i) {
                    const double val = ep
                                    + PAST_DELTA[d * N + i]
                                    + AP[d * N * N + j * N + i];
                    if (val > best_val) {
                        best_val = val;
                        best_d   = d;
                        best_i   = i;
                    }
                }
            }

            // Se t < D, la PHASE 1 ha già scritto delta[t,j]:
            // sovrascriviamo solo se il valore induttivo è almeno altrettanto buono.
            // (Python: cond = best_vals < delta[t,:] → non aggiornare se cond è True)
            const bool update = (t >= D) || (best_val >= delta[t * N + j]);
            if (update) {
                delta      [t * N + j] = best_val;
                delta_state[t * N + j] = best_i;
                delta_dur  [t * N + j] = best_d + 1;
            }
        }
    }
    
    // Retrieve data from GPU


    // Backtracking
    std::vector<int> path = backtracking_termination(delta, delta_state, delta_dur, T);
    
    // Free GPU Memory
    hsmm_free_gpu(d_trans_mat, d_emission_probs, d_start_probs, d_duration_probs, d_obs_seq);
    CUDA_CHECK(cudaFree(d_AP));

    return path;
}

