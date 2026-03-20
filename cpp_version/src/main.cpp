#include "hsmm.hpp"

#include <iostream>
#include <chrono>

int main() {
  
    // Load model from JSON
    HSMM model = HSMM("../data/sleep_data_10states_20_5.json");
    model.to_log_space();
    //model.print();

    // Run Tensor Viterbi
    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<int> result = model.decoding_tensor_viterbi();
    auto t_end = std::chrono::high_resolution_clock::now();
    

    // ── Print execution time───────────────────────────────────────────────────── //
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Execution time GPU Viterbi: " << std::fixed << std::setprecision(4) << elapsed << " seconds\n";

    // ── Print result ──────────────────────────────────────────────────────────── //
    std::cout << "[";
    for (int t = 0; t < static_cast<int>(result.size()); ++t)
        std::cout << result[t] << " ";
    std::cout << "]";

    return 0;
}
