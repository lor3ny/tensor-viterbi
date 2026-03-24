import time
import numpy as np

from py_src.tensor_viterbi import run_log_tensor_viterbi_cached
from py_src.tensor_viterbi import run_log_tensor_viterbi_no_cache
from py_src.tensor_viterbi import run_tensor_viterbi

from validation.hsmmlearn_viterbi import validate
from py_src.hsmm import HSMM


if __name__ == "__main__":


    data_path = "data/20states_1000steps_20dur.json"
    #data_path = "data/sleep_data_10states_100_10.json"
    # hsmm_sleep.print_model()

    # [Vanilla Viterbi]
    # hsmm_sleep = load_sleep_model(data_path)
    # start_time = time.time()
    # v_predicted_states = hsmm_sleep.run_vanilla_viterbi()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Execution time of Vanilla Viterbi: {execution_time:.4f} seconds")

    # # [Tensor Viterbi]
    # hsmm_sleep = load_sleep_model(data_path)
    # start_time = time.time()
    # t_predicted_states = hsmm_sleep.run_log_tensor_viterbi()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Execution time of Log Tensor Viterbi (NO CACHE): {execution_time:.4f} seconds")

    # [Tensor Viterbi Cached]
    hsmm_sleep = HSMM.load_model(data_path)
    start_time = time.perf_counter()
    tc_predicted_states = run_log_tensor_viterbi_cached(hsmm_sleep)
    execution_time = time.perf_counter() - start_time
    print(f"Execution time of Log Tensor Viterbi: {execution_time:.4f} seconds")

    # Validation
    # validate("Vanilla vs Baseline", v_predicted_states, data_path)
    # print(v_predicted_states)
    # validate("Tensor vs Baseline", t_predicted_states, data_path)
    # print(t_predicted_states)
    validate("Tensor (Cached) vs Baseline", tc_predicted_states, data_path)

    np.savetxt("data/python_result.txt", tc_predicted_states.reshape(1, -1), fmt='%d', delimiter=' ') # used for gpu validation
