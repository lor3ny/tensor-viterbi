#pragma once

#include <vector>

namespace hsmm {

/**
 * Decode using CPU tensor Viterbi.
 * All matrix parameters must be in log space.
 *
 * @param n_states       Number of states N.
 * @param trans_mat      N×N transition matrix, row-major.
 * @param emission_probs O×N emission probability matrix, row-major.
 * @param start_probs    N-element initial state distribution.
 * @param duration_probs N×D duration probability matrix, row-major.
 * @param obs_seq        T-element observation sequence, 0-indexed.
 */
std::vector<int> decode_tensor_viterbi(
    int                        n_states,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& emission_probs_linear,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq
);

/**
 * Decode using GPU (CUDA) tensor Viterbi.
 * All matrix parameters must be in log space.
 */
std::vector<int> decode_tensor_viterbi_cuda(
    int                        n_states,
    const std::vector<double>& trans_mat,
    const std::vector<double>& emission_probs,
    const std::vector<double>& emission_probs_linear,
    const std::vector<double>& start_probs,
    const std::vector<double>& duration_probs,
    const std::vector<int>&    obs_seq
);

} // namespace hsmm
