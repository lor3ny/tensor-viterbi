#include "hsmm.hpp"


HSMM::HSMM(const std::vector<std::string>&  states,
           const std::vector<std::string>&  emissions,
           const std::vector<double>&       trans_mat,
           const std::vector<double>&       emission_probs,
           const std::vector<double>&       start_probs,
           const std::vector<double>&       duration_probs)
{
    this->states_ = states;
    this->emissions_ = emissions;
    this->trans_mat_ = trans_mat;
    this->emission_probs_ = emission_probs;
    this->start_probs_ = start_probs;
    this->duration_probs_ = duration_probs;
}

HSMM::HSMM(const std::string& json_data_path){

    std::ifstream file(json_data_path);
    if (!file.is_open())
        throw std::runtime_error("load_sleep_model: cannot open \"" + json_data_path + "\"");
 
    nlohmann::json cfg;
    file >> cfg;

    const int n_bins       = cfg["n_bins"].get<int>();      // O — emission bins

    std::vector<std::string> sleep_states;
    for (const auto& s : cfg["states"])
        sleep_states.push_back(s["name"].get<std::string>());
 
    const int N = static_cast<int>(sleep_states.size());

    std::vector<std::string> sleep_emissions(n_bins);
    for (int o = 0; o < n_bins; ++o)
        sleep_emissions[o] = std::to_string(o);

    const auto& raw_obs = cfg["obs_seq"];
    std::vector<int> sleep_obs_seq(raw_obs.size());
    for (std::size_t t = 0; t < raw_obs.size(); ++t)
        sleep_obs_seq[t] = raw_obs[t].get<int>() - 1;   // 1-based → 0-based
 

    std::vector<double> sleep_trans_mat;
    sleep_trans_mat.reserve(N * N);
    for (const auto& row : cfg["trans_mat"])
        for (const auto& val : row)
            sleep_trans_mat.push_back(val.get<double>());
 

    std::vector<double> sleep_emission_probs(n_bins * N, 0.0);
    for (int s = 0; s < N; ++s) {
        const auto& ep = cfg["states"][s]["emission_probs"];
        for (int o = 0; o < n_bins; ++o)
            sleep_emission_probs[o * N + s] = ep[o].get<double>();
    }
 

    std::vector<double> sleep_start_probs = cfg["pi"].get<std::vector<double>>();
 
    std::vector<double> sleep_duration_probs;
    for (int s = 0; s < N; ++s) {
        const auto& dp = cfg["states"][s]["duration_probs"];
        for (const auto& val : dp)
            sleep_duration_probs.push_back(val.get<double>());
    }
 
    HSMM hsmm_sleep(sleep_states,
                    sleep_emissions,
                    sleep_trans_mat,
                    sleep_emission_probs,
                    sleep_start_probs,
                    sleep_duration_probs);
 
    hsmm_sleep.set_obs_seq(sleep_obs_seq);
    return;
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