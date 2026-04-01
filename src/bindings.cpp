#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hsmm.hpp"

namespace py = pybind11;

using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<int,    py::array::c_style | py::array::forcecast>;


#ifndef NO_GPU
enum class Backend { CPP, OMP, CUDA };
#else
enum class Backend { CPP, OMP };
#endif

static py::array_t<int> _run(
    const int n_states,
    arr_d trans_mat,                // (N, N) — log space, Python row-major layout
    arr_d emission_probs,           // (O, N) — log space
    arr_d duration_probs_linear,    // (D, N) — linear space
    arr_d start_probs,              // (N,)   — log space
    arr_d duration_probs,           // (D, N) — log space, Python layout D×N
    arr_i obs_seq,                  // (T,)   — 0-indexed int
    Backend backend)
{
    const int N = n_states;
    const int O = static_cast<int>(emission_probs.shape(0));
    const int D = static_cast<int>(duration_probs.shape(0));
    const int T = static_cast<int>(obs_seq.size());

    // trans_mat (N,N): Python row-major flat[j*N+i] = Python[j,i] — matches C++ layout
    std::vector<double> tm(trans_mat.data(), trans_mat.data() + N * N);

    // emission_probs (O,N): Python row-major flat[o*N+s] — matches C++ layout
    std::vector<double> ep(emission_probs.data(), emission_probs.data() + O * N);

    // start_probs (N,)
    std::vector<double> sp(start_probs.data(), start_probs.data() + N);

    // duration_probs : Python (D,N) → C++ needs (N,D) row-major: dp[s*D+d]
    auto dp_buf = duration_probs.unchecked<2>();
    std::vector<double> dp(N * D);
    for (int s = 0; s < N; ++s)
        for (int d = 0; d < D; ++d)
            dp[s * D + d] = dp_buf(d, s);
    
    // duration_probs_linear : Python (D,N) → C++ needs (N,D) row-major: dpl[s*D+d]
    auto dpl_buf = duration_probs_linear.unchecked<2>();
    std::vector<double> dpl(N * D);
    for (int s = 0; s < N; ++s)
        for (int d = 0; d < D; ++d)
            dpl[s * D + d] = dpl_buf(d, s);
    
    // obs_seq (T,)
    std::vector<int> obs(obs_seq.data(), obs_seq.data() + T);

    std::vector<int> result;
    switch (backend) {
#ifndef NO_GPU
        case Backend::CUDA: result = hsmm::decode_tensor_viterbi_cuda(N, tm, ep, dpl, sp, dp, obs); break;
#endif
        case Backend::OMP:  result = hsmm::decode_tensor_viterbi_omp (N, tm, ep, dpl, sp, dp, obs); break;
        default:            result = hsmm::decode_tensor_viterbi     (N, tm, ep, dpl, sp, dp, obs); break;
    }

    auto out = py::array_t<int>(result.size());
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}


PYBIND11_MODULE(_native, m) {
    m.doc() = "Native C++/CUDA HSMM Viterbi decoders";

    m.def("decode_tensor_viterbi_cpp",
          [](int n_states,
             arr_d trans_mat, arr_d emission_probs, arr_d duration_probs_linear,
             arr_d start_probs, arr_d duration_probs,
             arr_i obs_seq) {
              return _run(n_states, trans_mat, emission_probs, duration_probs_linear,
                          start_probs, duration_probs, obs_seq, Backend::CPP);
          },
          py::arg("n_states"), py::arg("trans_mat"), py::arg("emission_probs"), py::arg("duration_probs_linear"),
          py::arg("start_probs"), py::arg("duration_probs"), py::arg("obs_seq"),
          "Run tensor Viterbi on CPU (C++). Data must already be in log space.");

    m.def("decode_tensor_viterbi_omp",
          [](int n_states,
             arr_d trans_mat, arr_d emission_probs, arr_d duration_probs_linear,
             arr_d start_probs, arr_d duration_probs,
             arr_i obs_seq) {
              return _run(n_states, trans_mat, emission_probs, duration_probs_linear,
                          start_probs, duration_probs, obs_seq, Backend::OMP);
          },
          py::arg("n_states"), py::arg("trans_mat"), py::arg("emission_probs"),
          py::arg("duration_probs_linear"), py::arg("start_probs"),
          py::arg("duration_probs"), py::arg("obs_seq"),
          "Run tensor Viterbi on CPU with OpenMP parallelism. Data must already be in log space.");

#ifndef NO_GPU
    m.def("decode_tensor_viterbi_cuda",
          [](int n_states,
             arr_d trans_mat, arr_d emission_probs, arr_d duration_probs_linear,
             arr_d start_probs, arr_d duration_probs,
             arr_i obs_seq) {
              return _run(n_states, trans_mat, emission_probs, duration_probs_linear,
                          start_probs, duration_probs, obs_seq, Backend::CUDA);
          },
          py::arg("n_states"), py::arg("trans_mat"), py::arg("emission_probs"), py::arg("duration_probs_linear"),
          py::arg("start_probs"), py::arg("duration_probs"), py::arg("obs_seq"),
          "Run tensor Viterbi on GPU (CUDA). Data must already be in log space.");
#endif // NO_GPU
}
