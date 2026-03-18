#include "hsmm.hpp"

#include <iostream>

int main() {
  

    HSMM model = HSMM("sleep_data.json");

    model.print();
    
    return 0;
}
