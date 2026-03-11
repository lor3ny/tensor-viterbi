import numpy as np

def compute_ap(A, P):
    """
    Compute AP = OuterProduct(A[i,j], P[i,d])
    A: NxN matrix
    P: NxD matrix
    AP: NxNxD tensor
    """
    # AP[i,j,d] = A[i,j] * P[i,d]
    return A[:, :, np.newaxis] * P[:, np.newaxis, :]  # (N,N,1) * (N,1,D) -> NxNxD


def method_a(AP, PAST_DELTA, EMISSION_PROBABILITY):
    """
    Method A:
    DELTA_EMISSION = OuterProduct(PAST_DELTA, EMISSION_PROBABILITY)  -> NxD outer -> NxNxD? 
    RESULT = ElementWiseProduct(AP, DELTA_EMISSION)
    
    PAST_DELTA:           NxD  (rows = source states i, cols = delay offsets d)
    EMISSION_PROBABILITY: NxD  (rows = target states j, cols = delay offsets d)
    DELTA_EMISSION[i,j,d] = PAST_DELTA[i,d] * EMISSION_PROBABILITY[j,d]
    RESULT[i,j,d]         = AP[i,j,d] * DELTA_EMISSION[i,j,d]
    """
    # PAST_DELTA[i,d]           -> (N,1,D)
    # EMISSION_PROBABILITY[j,d] -> (1,N,D)
    DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBABILITY[np.newaxis, :, :]  # NxNxD
    RESULT = AP * DELTA_EMISSION  # NxNxD element-wise
    return RESULT


def method_b(AP, PAST_DELTA, EMISSION_PROBABILITY):
    """
    Method B:
    Step 1 - Y_BroadcastProduct: broadcast PAST_DELTA[i,d] across AP's j-axis
        DELTA_EMISSION[i,j,d] = PAST_DELTA[i,d] * AP[i,j,d]

    Step 2 - X_BroadcastProduct: broadcast EMISSION_PROBABILITY[j,d] across result's i-axis
        RESULT[i,j,d] = EMISSION_PROBABILITY[j,d] * DELTA_EMISSION[i,j,d]
    """
    # Step 1: Y_BroadcastProduct — PAST_DELTA (N,D) broadcast over j-axis of AP (N,N,D)
    DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * AP  # (N,1,D) * (N,N,D) -> NxNxD

    # Step 2: X_BroadcastProduct — EMISSION_PROBABILITY (N,D) broadcast over i-axis
    RESULT = EMISSION_PROBABILITY[np.newaxis, :, :] * DELTA_EMISSION  # (1,N,D) * (N,N,D) -> NxNxD
    return RESULT


def find_t_maxs(RESULT):
    """
    t_MAXs: foreach S[j]-plane (i.e., for each j), find MAX over (i, d)
    Output shape: (N,) — one max value per j
    """
    return np.max(RESULT, axis=(0, 2))  # max over i-axis(0) and d-axis(2) -> shape (N,)


# --- Main loop ---
def run(A, P, DELTAS, times, Dmin, Dmax):
    N, D = P.shape

    results_a = []
    results_b = []
    t_maxs_list = []

    AP = compute_ap(A, P)  # NxNxD — precomputed outside the loop
    for t in times:
        # Slice DELTAS window: shape (N, D) assuming DELTAS is shape (T, N, D)
        PAST_DELTA = DELTAS[t - Dmax : t - Dmin]          # slice along time axis
        PAST_DELTA = np.prod(PAST_DELTA, axis=0)           # collapse to NxD if needed

        # Placeholder: replace with real emission computation
        EMISSION_PROBABILITY = np.ones((N, D))             # NxD

        # Method A
        RESULT_A = method_a(AP, PAST_DELTA, EMISSION_PROBABILITY)
        results_a.append(RESULT_A)

        # Method B
        # RESULT_B = method_b(AP, PAST_DELTA, EMISSION_PROBABILITY)
        # results_b.append(RESULT_B)

        # Verify methods are equivalent
        # assert np.allclose(RESULT_A, RESULT_B), "Method A and B diverged!"

        # t_MAXs: for each j-plane, find max over (i, d)
        t_maxs = find_t_maxs(RESULT_A)                     # shape (N,)
        t_maxs_list.append(t_maxs)

    return results_a, results_b, t_maxs_list