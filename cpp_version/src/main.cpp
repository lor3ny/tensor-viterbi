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
    HSMM model = HSMM("../data/20states_100000steps_200dur.json");
    model.to_log_space();
    model.print();

    // [VANILLA]
    auto start_v = std::chrono::high_resolution_clock::now();
    std::vector<int> result_cpp = model.decoding_vanilla_viterbi();
    auto end_v = std::chrono::high_resolution_clock::now();
    
    // ── Print execution time──//
    double elapsed_v = std::chrono::duration<double>(end_v - start_v).count();
    std::cout << "Execution time C++ Viterbi: " << std::fixed << std::setprecision(4) << elapsed_v << " seconds\n";


    // [TENSOR]
    double kernel_ms; 
    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::vector<int> result_gpu = model.decoding_tensor_viterbi(&kernel_ms);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    

    // ── Print execution time──//
    double elapsed_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
    std::cout << "Execution time GPU Viterbi: " << std::fixed << std::setprecision(4) << elapsed_gpu << " seconds\n";
    std::cout << "Execution time w/o malloc/memcpy: " << std::fixed << std::setprecision(4) << kernel_ms << " seconds\n";


    // ── Print result ──────────────────────────────────────────────────────────── //
    // std::cout << "[";
    // for (int t = 0; t < static_cast<int>(result.size()); ++t)
    //     std::cout << result[t] << " ";
    // std::cout << "]";

    save_path(result_gpu.data(), result_gpu.size(), "../data/cuda_result.txt");
    save_path(result_cpp.data(), result_cpp.size(), "../data/cpp_result.txt");

    return 0;
}
