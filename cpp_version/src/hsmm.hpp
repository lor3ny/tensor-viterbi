#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <fstream>

#include "json.hpp"

class HSMM {


private:

    std::vector<std::string> states_;           // size N
    std::vector<std::string> emissions_;        // size O
    std::vector<double>      trans_mat_;        // N×N  row-major
    std::vector<double>      emission_probs_;   // O×N  row-major
    std::vector<double>      start_probs_;      // N
    std::vector<double>      duration_probs_;   // N×D  row-major

    int N_; // number of states
    int O_; // number of distinct emissions
    int D_; // maximum duration

    std::vector<int> obs_seq_;  // integer-coded observation sequence (T)


public:

    /**
     * @param states         State labels (size N).
     * @param emissions      Emission labels (size O).
     * @param trans_mat      N×N transition matrix (row-major).
     * @param emission_probs O×N emission probability matrix (row-major).
     * @param start_probs    N-element initial state distribution.
     * @param duration_probs N×D duration probability matrix (row-major).
     */
    HSMM(const std::vector<std::string>&  states,
         const std::vector<std::string>&  emissions,
         const std::vector<double>&       trans_mat,
         const std::vector<double>&       emission_probs,
         const std::vector<double>&       start_probs,
         const std::vector<double>&       duration_probs);
    
    HSMM(const std::string& json_data_path);
    
    int num_states()    const { return static_cast<int>(states_.size()); }
    int num_emissions() const { return static_cast<int>(emissions_.size()); }
    int max_duration()  const { return D_; }
    int obs_length()    const { return static_cast<int>(obs_seq_.size()); }
    void set_obs_seq(const std::vector<int>& obs_seq){ this->obs_seq_ = obs_seq; }
    
    void print() const;

    // ------------------------------------------------------------------ //
    // Viterbi Algorithm
    // ------------------------------------------------------------------ //

    void find_t_maxs(const std::vector<double>& Sjid,
                     std::vector<double>&        max_vals,
                     std::vector<int>&           max_states,
                     std::vector<int>&           max_durs) const;

    std::vector<int> backtracking_termination(const std::vector<double>& delta,
                                              const std::vector<int>&    psi_state,
                                              const std::vector<int>&    psi_dur,
                                              int                        T) const;

    std::vector<int> decoding_tensor_viterbi();

    std::vector<int> decoding_vanilla_viterbi();

};