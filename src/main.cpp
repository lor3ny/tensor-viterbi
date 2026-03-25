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

void print_result(const int* result, int T) {

    for (int t = 0; t < T; t++) {
        printf("%d ", result[t]);
    }
    printf("\n");
}

int main() {
    
    // Load model from JSON
    HSMM model = HSMM("../data/3states_20steps_4dur.json");
    // HSMM model = HSMM("../data/20states_100000steps_200dur.json");
    // HSMM model = HSMM("../data/32states_100000steps_1024dur.json");
    // HSMM model = HSMM("../data/64states_100000steps_1024dur.json");
    // HSMM model = HSMM("../data/sleep_data_10states_100000_200.json");
    model.to_log_space();
    //model.print();

    // [Tensor C++]
    auto start_v = std::chrono::high_resolution_clock::now();
    std::vector<int> result_cpp = model.decode_tensor_viterbi();
    auto end_v = std::chrono::high_resolution_clock::now();
    
    // ── Print execution time──//
    double elapsed_v = std::chrono::duration<double>(end_v - start_v).count();
    std::cout << "Execution time C++ Viterbi: " << std::fixed << std::setprecision(6) << elapsed_v << " seconds\n";

    // [TENSOR CUDA]
    // auto start_gpu = std::chrono::high_resolution_clock::now();
    // std::vector<int> result_cuda = model.decode_tensor_viterbi_cuda();
    // auto end_gpu = std::chrono::high_resolution_clock::now();
    

    // ── Print execution time──//
    // double elapsed_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
    // std::cout << "Execution time GPU Viterbi: " << std::fixed << std::setprecision(6) << elapsed_gpu << " seconds\n";


    // // Save results to files
    // save_path(result_cpp.data(), result_cpp.size(), "../data/cpp_result.txt");
    // save_path(result_cuda.data(), result_cuda.size(), "../data/cuda_result.txt");

    return 0;
}
