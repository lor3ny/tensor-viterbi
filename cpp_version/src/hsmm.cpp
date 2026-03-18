#include "hsmm.hpp"

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
    const auto& raw_obs = cfg["obs_seq"];
    std::vector<int> obs_seq(raw_obs.size());
    for (std::size_t t = 0; t < raw_obs.size(); ++t)
        obs_seq[t] = raw_obs[t].get<int>() - 1;

    // -------- TRANS --------
    std::vector<double> trans_mat;
    trans_mat.reserve(N * N);
    for (const auto& row : cfg["trans_mat"])
        for (const auto& val : row)
            trans_mat.push_back(val.get<double>());

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

    obs_seq_ = obs_seq;
}

#include <iostream>
#include <iomanip>

void HSMM::print() const {
    const int N = num_states();
    const int O = num_emissions();
    const int D = 4;

    std::cout << "===== HSMM MODEL =====\n";

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
    // TODO: return T-element state path
    return {};
}
 
std::vector<int> HSMM::decoding_vanilla_viterbi()
{
    // TODO: implement tensor Viterbi
    return {};
}

std::vector<int> HSMM::decoding_tensor_viterbi()
{
    // TODO: implement tensor Viterbi
    //
    // Suggested local variables (mirrors Python implementation):
    //   int T = obs_length();
    //   std::vector<double> delta      (T * N_, 0.0);
    //   std::vector<int>    delta_state(T * N_, 0);
    //   std::vector<int>    delta_dur  (T * N_, 0);
    //
    //   std::vector<double> PAST_DELTA     (N_ * D_, 0.0);
    //   std::vector<double> EMISSION_PROBS (N_ * D_, 0.0);
    //   std::vector<double> DELTA_EMISSION (N_ * N_ * D_, 0.0);
    //   std::vector<double> AP             (N_ * N_ * D_, 0.0);
    //   std::vector<double> RESULT_B       (N_ * N_ * D_, 0.0);
    //
    // --- INITIALIZATION ---
    //   Build AP[:, :, :] = start_probs[j] * emission_probs[obs[0], j]
    //   Call find_t_maxs(AP, p_maxs, s_maxs, d_maxs)
    //   delta[0, :] = p_maxs
    //   delta_state[0, :] = -1   (no predecessor)
    //   delta_dur  [0, :] = 1
    //
    // --- INDUCTION (t = 1 .. T-1) ---
    //   Rebuild AP = trans_mat[i,j] * duration_probs[j,d]  (N×N×D)
    //   For each t:
    //     1. Fill EMISSION_PROBS[j, d] = prod of emission_probs[obs[t-d'..t-1], j]
    //        for d_val in 1..min(D_, t+1)
    //     2. Fill PAST_DELTA[:, :window] from delta[max(0,t-D_)..t-1, :] reversed
    //     3. DELTA_EMISSION[i,j,d] = PAST_DELTA[i,d] * AP[i,j,d]
    //     4. RESULT_B[i,j,d]       = EMISSION_PROBS[j,d] * DELTA_EMISSION[i,j,d]
    //     5. find_t_maxs(RESULT_B, p_maxs, s_maxs, d_maxs)
    //     6. delta[t,:] = p_maxs;  delta_state[t,:] = s_maxs;  delta_dur[t,:] = d_maxs+1
    //
    // --- TERMINATION ---
    //   return backtracking_termination(delta, delta_state, delta_dur, T)
    return {};
}