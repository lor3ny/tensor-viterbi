from curses import window
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import time
import json
import sys


from validation.hsmmlearn_viterbi import validate


# ----- DEBUG VOXEL


def debug_visualize(input):
    # Map values to colors using a colormap
    norm = mcolors.Normalize(vmin=input.min(), vmax=input.max())
    colormap = cm.viridis

    # voxels() needs a (X, Y, Z, 4) RGBA inputay for facecolors
    colors = colormap(norm(input))  # shape: (5, 5, 5, 4)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(input, facecolors=colors, edgecolor='k', linewidth=0.3)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, ax=ax, shrink=0.5, label='Value')

    plt.show()


# --- 3. HELPER FUNCTIONS ---

def get_log_emission_probs(obs, means, stds):
    """Calculate log P(observation | state) for all t, all states."""
    T = len(obs)
    n = len(means)
    log_probs = np.zeros((T, n))
    for s in range(n):
        # Log PDF of Gaussian
        var = stds[s]**2
        denom = np.sqrt(2 * np.pi * var)
        exponent = -0.5 * ((obs - means[s])**2) / var
        log_probs[:, s] = exponent - np.log(denom)
    return log_probs

# --- 4. HSMM VITERBI ALGORITHM ---


# obs: observations sequence of length T
# trans_mat: N x N transition matrix (0 on diagonal for HSMM)
# duration_probs: N x D matrix of duration probabilities
# means, stds: N-length arrays for emission distributions

class HSMM:
    def __init__(self, states, emissions, trans_mat, emission_prob, start_probs, duration_probs):
            self.states = states
            self.emissions = emissions
            self.trans_mat = trans_mat
            self.emission_probs = emission_prob
            self.start_probs = start_probs
            self.duration_probs = duration_probs

    def set_obs_sequence(self, obs_seq):
        self.obs_seq = obs_seq


    # This function is helpful if you don't have the emission probs, but you have only a mean ad std value.
    def gen_log_emission_probs(self, means, stds):
        """Calculate log P(observation | state) for all t, all states."""
        T = len(self.obs_seq)
        n = len(self.states)
        log_probs = np.zeros((T, n))
        for s in range(n):
            # Log PDF of Gaussian
            var = stds[s]**2
            denom = np.sqrt(2 * np.pi * var)
            exponent = -0.5 * ((self.obs_seq - means[s])**2) / var
            log_probs[:, s] = exponent - np.log(denom)
        return log_probs


    def find_t_maxs(self, Sjid, delta=None, d=None):
        """
        t_MAXs: foreach S[j]-plane (i.e., for each j), find MAX over (i, d)
        RESULT shape: (N, N, D)
        
        Returns:
            max_vals:   (N,) — max probability per j-plane
            max_states: (N,) — i coordinate (state) of the max per j-plane
            max_durs:   (N,) — d coordinate (duration) of the max per j-plane
        """
        N = Sjid.shape[1]
        
        # Reshape each j-plane (N, D) into a flat array, find argmax, then unravel
        max_vals   = np.zeros(N)
        max_states = np.zeros(N, dtype=int)
        max_durs   = np.zeros(N, dtype=int)

        for j in range(N):
            plane = Sjid[:, j, :]          # shape (N, D) — the j-th plane
            
            flat_idx = np.argmax(plane)      # argmax over flattened (N*D)
            i, d = np.unravel_index(flat_idx, plane.shape)  # recover (i, d) coords

            if d != None:
                plane[i, d]<delta[j, d]


            max_vals[j]> delta[j, d]
            max_states[j] = i               # x coordinate = source state
            max_durs[j]   = d               # y coordinate = duration

        return max_vals, max_states, max_durs


    def backtracking_termination(self, delta, psi_state, psi_dur, T):
        #! THIS SECTION CAN BE PORTED ON CPU
        #* TERMINATION
        path = np.zeros(T, dtype=int)

        # Find best ending state at T-1
        t = T - 1
        best_last_state = np.argmax(delta[t])
        curr_state = best_last_state
        
        while t > 0:

            d = psi_dur[t, curr_state]
            prev_s = psi_state[t, curr_state]
            
            # Fill the segment
            start_t = t - d + 1
            path[start_t : t+1] = curr_state
            
            # Move back
            t = t - d
            curr_state = prev_s
  
        return path
    

    # In a 3D NumPy array with shape (a, b, c):

    # axis 0 → depth (z) — the first index, selects a 2D "slice"
    # axis 1 → rows — the second index, selects a row within a slice
    # axis 2 → columns — the third index, selects a column within a row

    def run_tensor_viterbi(self):

        T = len(self.obs_seq)  # time steps
        N = len(self.states) # states count
        D = self.duration_probs.shape[1]
        
        delta = np.full((T, N), 0.0)        
        delta_state = np.zeros((T, N), dtype=int)
        delta_dur = np.zeros((T, N), dtype=int)
        max_vals   = np.zeros(N)
        max_states = np.zeros(N, dtype=int)
        max_durs   = np.zeros(N, dtype=int)

        #PAST_DELTA = np.ones((D, N))

        #self.emission_probs = self.emission_probs.T
        self.duration_probs = self.duration_probs.T
        self.trans_mat = self.trans_mat.T
        
        EMISSION_PROBS = np.ones((D,N))
        DELTA_EMISSION = np.ones((N, N, D))
        AP = np.ones((N, N, D)) # Precompute AP outside the loop
        

        #* INITIALIZATION
        #! PHASE 1 - INITIALIZATION 0<=t<D

        PAST_DELTA = self.duration_probs * self.start_probs[np.newaxis,:]
  
        #* METHOD 1
        # for d in range(0,D):
        #     obs = int(self.obs_seq[d])
        #     self.emission_probs[obs, :]
        #     EMISSION_PROBS[d:,:] *= self.emission_probs[obs, :][:,np.newaxis].T

        #* METHOD 2
        obs_indices = self.obs_seq[:D].astype(int)  # shape: (D,)
        emission_rows = self.emission_probs[obs_indices, :]
        cum_product = np.cumprod(emission_rows, axis=0)  # shape: (D, num_states)
        EMISSION_PROBS *= cum_product  # shape: (num_states, D)

        #? POTREMMO RIMUOVERE LA TRASPOSIZIONE
        delta[0:D] = (PAST_DELTA * EMISSION_PROBS)


        #! PHASE 2 - INDUCTION  t>0   

        execution_time_max = 0.0
        execution_time_tens = 0.0

        AP = self.trans_mat[np.newaxis, :, :] * self.duration_probs[ :, :, np.newaxis]  # (N,N,1) * (N,1,D) -> NxNxD  
        for t in range(1, T):

            #! ----- TOO SLOW -----
            start_time = time.time()
            # Slice DELTAS window: shape (N, D) assuming DELTAS is shape (T, N, D)
            EMISSION_PROBS = np.ones((D, N))

            # for d_val in range(0,  min(D, t)):
            #     segment_indices = np.array(self.obs_seq[t - d_val : t+1], dtype=int)
            #     relevant_probs = self.emission_probs[segment_indices, :]   # DxN
            #     EMISSION_PROBS[d_val, :] = np.prod(relevant_probs, axis=0)
            
            segment_indices = self.obs_seq[max(0, t - D):t].astype(int)  # shape: (D,)
            relevant_probs = self.emission_probs[segment_indices, :]
            cum_product = np.cumprod(np.flip(relevant_probs, axis=0), axis=0)  # shape: (D, num_states)
            EMISSION_PROBS *= cum_product  # shape: (num_states, D)
            
            end_time = time.time()
            execution_time_tens += end_time - start_time
            #! ----- TOO SLOW -----

            window = delta[max(0, t-D) : t, :]
            PAST_DELTA[:window.shape[0], :] = window[::-1]

            #* Method A
            # DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBS[np.newaxis, :, :]  # NxNxD
            # RESULT_A = AP #* DELTA_EMISSION  # NxNxD element-wise

            #* Method B
            DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * AP  # (D,N) * (D,N,N) -> DxNxN
            RESULT_B = EMISSION_PROBS[ :, :, np.newaxis] * DELTA_EMISSION  # (N,D) * (D,N,N) -> DxNxN
        
            #! ----- TOO SLOW -----
            start_time = time.time()
            for j in range(N):
                plane = RESULT_B[:, j, :]          # shape (N, D) — the j-th plane
                
                flat_idx = np.argmax(plane[0:min(t,D)])      # argmax over flattened (N*D)
                d, i = np.unravel_index(flat_idx, plane.shape)  # recover (d, i) coords

                if t < D and plane[d, i] < delta[t, j]:
                    max_vals[j] = delta[t, j]
                    max_states[j] = delta_state[t, j]
                    max_durs[j] = delta_dur[t, j]
                else:
                    max_vals[j] = plane[d, i]
                    max_states[j] = i               
                    max_durs[j]   = d              

            delta[t, :] = max_vals 
            delta_state[t, :] = max_states
            delta_dur[t, :] = max_durs+1
            #! ----- TOO SLOW -----

            end_time = time.time()
            execution_time_max += end_time - start_time

        print(f"MAX Section of Tensor Viterbi: {execution_time_max:.4f} seconds")
        print(f"EMISSION Section of Tensor Viterbi: {execution_time_tens:.4f} seconds")

        path = self.backtracking_termination(delta, delta_state, delta_dur, T)
        
        return path, delta


    # We use np.log() + smoothing to transform multiplications in additions
    def run_viterbi(self):

        T = len(self.obs_seq)  
        N = len(self.states) 
        D = self.duration_probs.shape[1]  
        
        delta = np.zeros((T, N))
        psi_state = np.zeros((T, N), dtype=int)
        psi_dur = np.zeros((T, N), dtype=int)

        #! PHASE 1 - INITIALIZATION 0<=t<D
        #* delta(0,sj) = pi(sj) * P(d|sj) * |-|{k = t-d}(b(sj, seq_obs(k))
        for state in range(N):
            for d in range(0, D):
                obs_score = 1.0    
                for tau in range(0,d+1):
                    obs_index = int(self.obs_seq[tau])
                    obs_score *= self.emission_probs[obs_index, state]

                dur_score = self.duration_probs[state, d]
                start_prob = self.start_probs[state]
                score = start_prob *  dur_score * obs_score
                if score > delta[d, state]:
                    delta[d, state] = score
                    psi_dur[d, state] = d+1
                    psi_state[d, state] = state

        #! PHASE 2 - INDUCTION  t>0
        #* delta(t, sj) = max{d} ( max{si} ( delta(t-d,si) * a(si,sj) ) * P(d|sj) * |-|{k = t-d}(b(sj, seq_obs(k)))  
        for t in range(1, T):
            for sj in range(N):
                for d in range(1, D+1):
                    if t - d < 0: 
                        continue # Cannot look back past 0 here
                    
                    # |-|{k = t-d}(b(sj, seq_obs(k)
                    obs_score = 1.0
                    for tau in range(0,d):
                        obs_index = int(self.obs_seq[t-tau])
                        obs_score *= self.emission_probs[obs_index, sj]
                    
                    # P(d|Sj)
                    dur_score = self.duration_probs[sj, d-1]
                    
                    best_prev_score = -np.inf
                    best_prev_state = -1
                    for si in range(N):
                        # HSMMs handle self-loops via duration, Skip impossibile transitions. 
                        #! But with product is necessary, maybe can be inserted but doesn't change much in terms of performance
                        # if si == sj or self.trans_mat[si, sj] == 0: 
                        #     continue

                        # Score = delta(t-d,si) + a(si,sj)-Transition + Duration + Emissions
                        total_score = self.trans_mat[si, sj] * dur_score * delta[t - d, si] * obs_score
   
                        if total_score > best_prev_score:
                            best_prev_score = total_score
                            best_prev_state = si
                    
                    # Update Delta if this duration d is better than others for ending at t
                    if best_prev_score > delta[t, sj]:
                        delta[t, sj] = best_prev_score
                        psi_state[t, sj] = best_prev_state
                        psi_dur[t, sj] = d

        path = self.backtracking_termination(delta, psi_state, psi_dur, T)
            
        return path, delta
    


def load_sleep_model(json_path: str = "hsmm_config.json") -> HSMM:

    with open(json_path, "r") as f:
        cfg = json.load(f)
 
    # ── scalars ──────────────────────────────────────────────────────────────
    time_steps   = int(cfg["n_steps"])
    max_duration = int(cfg["M"])
    n_bins       = int(cfg["n_bins"])
    seed         = int(cfg["seed"])
 
    np.random.seed(seed)
    random.seed(seed)
 
    sleep_states = [s["name"] for s in cfg["states"]]   # ["Awake", ...]
    J = len(sleep_states)
 
    sleep_emissions = np.arange(n_bins)                  # shape (13,)
 
    sleep_obs_seq = np.array(cfg["obs_seq"], dtype=float) - 1   # shape (100,)

    sleep_trans_mat = np.array(cfg["trans_mat"], dtype=float)         # shape (4, 4)
 
    emission_by_state = np.array(
        [s["emission_probs"] for s in cfg["states"]], dtype=float
    )                                                            # shape (4, 13)
    sleep_emission_probs = emission_by_state.T                   # shape (13, 4)
 
    sleep_start_probs = np.array(cfg["pi"], dtype=float)        # shape (4,4)
 
    sleep_duration_probs = np.array(
        [s["duration_probs"] for s in cfg["states"]], dtype=float
    )                                                            # shape (4, M)
 
    hsmm_sleep = HSMM(
        sleep_states, 
        sleep_emissions, 
        sleep_trans_mat, 
        sleep_emission_probs, 
        sleep_start_probs, 
        sleep_duration_probs
    )
    hsmm_sleep.set_obs_sequence(sleep_obs_seq)

    return hsmm_sleep


if __name__ == "__main__":

    hsmm_sleep = load_sleep_model("sleep_data.json")

    start_time = time.time()
    v_predicted_states, delta_v = hsmm_sleep.run_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time
    # print("Predicted States:")
    # print(v_predicted_states)
    print(f"Execution time of Vanilla Viterbi: {execution_time:.4f} seconds")

    start_time = time.time()
    t_predicted_states, delta_t = hsmm_sleep.run_tensor_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time

    # print("Predicted States:")
    # print(t_predicted_states)
    print(f"Execution time of Tensor Viterbi: {execution_time:.4f} seconds")

    np.testing.assert_array_equal(
        t_predicted_states,
        v_predicted_states,
        err_msg="Predicted state sequences are different"
    )
    
    np.testing.assert_allclose(
        delta_v,
        delta_t,
        rtol=1e-10,
        atol=1e-15,
        err_msg="Results are different"
    )

    # print(delta_t)
    # print(delta_v)
    validate(t_predicted_states)
