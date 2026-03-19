#include "hsmm.hpp"

#include <iostream>

int main() {
  

    HSMM model = HSMM("sleep_data.json");
    model.to_log_space();
    model.print();

    model.decoding_tensor_viterbi();

    return 0;
}
