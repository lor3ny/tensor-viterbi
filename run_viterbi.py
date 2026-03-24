import time
import numpy as np

from py_src.tensor_viterbi import decode_log_tensor_viterbi_cached
from py_src.tensor_viterbi import decode_log_tensor_viterbi_no_cache
from py_src.tensor_viterbi import decode_tensor_viterbi
from py_src.tensor_viterbi import decode_vanilla_viterbi

from validation.hsmmlearn_viterbi import validate
from py_src.hsmm import HSMM


def TIME_MEASURE(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"Execution time of {func.__name__}: {elapsed:.4f} seconds")
    return result



if __name__ == "__main__":


    data_path = "data/3states_20steps_4dur.json"
    hsmm_sleep = HSMM.load_model(data_path)
    #data_path = "data/sleep_data_10states_100_10.json"
    # hsmm_sleep.print_model()

    v_predicted_states = TIME_MEASURE(decode_vanilla_viterbi, hsmm_sleep)

    tc_predicted_states = TIME_MEASURE(decode_log_tensor_viterbi_cached, hsmm_sleep)

    print(v_predicted_states)
    print(tc_predicted_states)


    validate("Vanilla vs Baseline", v_predicted_states, data_path)
    validate("Tensor (Cached) vs Baseline", tc_predicted_states, data_path)

    #np.savetxt("data/python_result.txt", tc_predicted_states.reshape(1, -1), fmt='%d', delimiter=' ') # used for gpu validation
