#include "hsmm.hpp"

#include <iostream>
#include <chrono>

void save_path(const int* result, int T, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (f == nullptr) {
        fprintf(stderr, "[ERROR] save_path: impossibile aprire '%s': %s\n",
                filename, strerror(errno));
        return;
    }

    for (int t = 0; t < T; t++) {
        fprintf(f, "%d ", result[t]);
    }

    fclose(f);
}


int main() {
  
    // Load model from JSON
    HSMM model = HSMM("../data/sleep_data_10states_10000_100.json");
    model.to_log_space();
    //model.print();

    double kernel_ms; 
    // Run Tensor Viterbi
    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<int> result = model.decoding_tensor_viterbi(&kernel_ms);
    auto t_end = std::chrono::high_resolution_clock::now();
    

    // ── Print execution time───────────────────────────────────────────────────── //
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Execution time GPU Viterbi: " << std::fixed << std::setprecision(4) << elapsed << " seconds\n";
    std::cout << "Execution time w/o malloc/memcpy: " << std::fixed << std::setprecision(4) << kernel_ms << " seconds\n";


    // ── Print result ──────────────────────────────────────────────────────────── //
    std::cout << "[";
    for (int t = 0; t < static_cast<int>(result.size()); ++t)
        std::cout << result[t] << " ";
    std::cout << "]";

    save_path(result.data(), result.size(), "../data/cuda_result.txt");

    return 0;
}
