#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hsmm.hpp"

namespace py = pybind11;

// ------------------------------------------------------------------ //
// Helper: run C++ or CUDA Viterbi from numpy arrays
// ------------------------------------------------------------------ //
// Expected shapes (log-space, same convention as Python HSMM):
//   trans_mat      : (N, N)  row-major
//   emission_probs : (O, N)  row-major
//   start_probs    : (N,)
//   duration_probs : (D, N)  row-major
//   obs_seq        : (T,)    integer-coded
// ------------------------------------------------------------------ //

static py::array_t<int> _run(
    py::array_t<double, py::array::c_style | py::array::forcecast> trans_mat,
    py::array_t<double, py::array::c_style | py::array::forcecast> emission_probs,
    py::array_t<double, py::array::c_style | py::array::forcecast> start_probs,
    py::array_t<double, py::array::c_style | py::array::forcecast> duration_probs,
    py::array_t<int,    py::array::c_style | py::array::forcecast> obs_seq,
    bool cuda
) {
    auto r_tm = trans_mat.unchecked<2>();
    auto r_ep = emission_probs.unchecked<2>();
    auto r_sp = start_probs.unchecked<1>();
    auto r_dp = duration_probs.unchecked<2>();
    auto r_os = obs_seq.unchecked<1>();

    int N = static_cast<int>(r_tm.shape(0));
    int O = static_cast<int>(r_ep.shape(0));
    int D = static_cast<int>(r_dp.shape(0));
    int T = static_cast<int>(r_os.shape(0));

    std::vector<std::string> states(N), emissions(O);
    for (int i = 0; i < N; ++i) states[i]    = std::to_string(i);
    for (int i = 0; i < O; ++i) emissions[i] = std::to_string(i);

    std::vector<double> tm(r_tm.data(0, 0), r_tm.data(0, 0) + N * N);
    std::vector<double> ep(r_ep.data(0, 0), r_ep.data(0, 0) + O * N);
    std::vector<double> sp(r_sp.data(0),    r_sp.data(0)    + N);
    std::vector<double> dp(r_dp.data(0, 0), r_dp.data(0, 0) + D * N);
    std::vector<int>    os(r_os.data(0),    r_os.data(0)    + T);

    HSMM model(states, emissions, tm, ep, sp, dp);
    model.set_obs_seq(os);

    std::vector<int> result = cuda ? model.decode_tensor_viterbi_cuda()
                                   : model.decode_tensor_viterbi();

    auto out = py::array_t<int>(T);
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}


py::array_t<int> decode_tensor_viterbi_cpp(
    py::array_t<double> trans_mat,
    py::array_t<double> emission_probs,
    py::array_t<double> start_probs,
    py::array_t<double> duration_probs,
    py::array_t<int>    obs_seq
) {
    return _run(trans_mat, emission_probs, start_probs, duration_probs, obs_seq, false);
}

py::array_t<int> decode_tensor_viterbi_cuda(
    py::array_t<double> trans_mat,
    py::array_t<double> emission_probs,
    py::array_t<double> start_probs,
    py::array_t<double> duration_probs,
    py::array_t<int>    obs_seq
) {
    return _run(trans_mat, emission_probs, start_probs, duration_probs, obs_seq, true);
}


PYBIND11_MODULE(_native, m) {
    m.doc() = "Native C++/CUDA HSMM Viterbi decoders";

    m.def("decode_tensor_viterbi_cpp", &decode_tensor_viterbi_cpp,
          py::arg("trans_mat"), py::arg("emission_probs"),
          py::arg("start_probs"), py::arg("duration_probs"), py::arg("obs_seq"),
          "Run tensor Viterbi on CPU (C++).");

    m.def("decode_tensor_viterbi_cuda", &decode_tensor_viterbi_cuda,
          py::arg("trans_mat"), py::arg("emission_probs"),
          py::arg("start_probs"), py::arg("duration_probs"), py::arg("obs_seq"),
          "Run tensor Viterbi on GPU (CUDA).");
}
