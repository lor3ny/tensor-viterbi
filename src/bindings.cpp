#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hsmm.hpp"

namespace py = pybind11;


static py::array_t<int> _run(const std::string& json_path, bool cuda) {
    HSMM model(json_path);

    std::vector<int> result = cuda ? model.decode_tensor_viterbi_cuda()
                                   : model.decode_tensor_viterbi();

    auto out = py::array_t<int>(result.size());
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}


PYBIND11_MODULE(_native, m) {
    m.doc() = "Native C++/CUDA HSMM Viterbi decoders";

    m.def("decode_tensor_viterbi_cpp",
          [](const std::string& json_path) { return _run(json_path, false); },
          py::arg("json_path"),
          "Run tensor Viterbi on CPU (C++).");

    m.def("decode_tensor_viterbi_cuda",
          [](const std::string& json_path) { return _run(json_path, true); },
          py::arg("json_path"),
          "Run tensor Viterbi on GPU (CUDA).");
}
