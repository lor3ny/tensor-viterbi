from curses import window
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import time
import json
from deprecated import deprecated


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

    def print_model(self):
        N = len(self.states)
        O = len(self.emissions)
        D = self.duration_probs.shape[0]
        T = len(self.obs_seq)

        print("===== HSMM MODEL =====")

        # Dimensioni
        print("\nDimensions:")
        print(f"  N (states)    = {N}")
        print(f"  O (emissions) = {O}")
        print(f"  D (max dur)   = {D}")
        print(f"  T (obs len)   = {T}")

        # Stati
        print(f"\nStates ({N}):")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s}")

        # Emissioni
        print(f"\nEmissions ({O}):")
        for o, e in enumerate(self.emissions):
            print(f"  [{o}] {e}")

        # Start probabilities
        print("\nStart probabilities (pi):")
        for i, s in enumerate(self.states):
            print(f"  {s}: {self.start_probs[i]:.6f}")

        # Transition matrix
        print("\nTransition matrix (N x N):")
        for i in range(N):
            row = "  ".join(f"{self.trans_mat[i, j]:8.6f}" for j in range(N))
            print(f"  {row}")

        # Emission probabilities
        print("\nEmission probabilities (O x N):")
        for o in range(O):
            row = "  ".join(f"{self.emission_probs[o, s]:8.6f}" for s in range(N))
            print(f"  Obs {o}: {row}")

        # Duration probabilities
        print("\nDuration probabilities:")
        for s in range(N):
            row = "  ".join(f"{self.duration_probs[d, s]:.6f}" for d in range(D))
            print(f"  State {self.states[s]}: [ {row} ]")

        print("\n======================")

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
    

    @deprecated(reason="It doesn't work after 370 timesteps because it goes on underflow, use the log-space function.")
    def run_tensor_viterbi(self):

        self.duration_probs = self.duration_probs.T
        self.trans_mat = self.trans_mat.T

        T = len(self.obs_seq)  # time steps
        N = len(self.states) # states count
        D = self.duration_probs.shape[0]
        
        delta = np.full((T, N), 0.0)        
        delta_state = np.zeros((T, N), dtype=int)
        delta_dur = np.zeros((T, N), dtype=int)
        
        #PAST_DELTA = np.ones((D, N))
        EMISSION_PROBS = np.ones((D,N))
        DELTA_EMISSION = np.ones((N, N, D))
        AP = np.ones((N, N, D)) # Precompute AP outside the loop
        

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
        cum_emission = np.cumprod(emission_rows, axis=0)  # shape: (D, num_states)
        EMISSION_PROBS = cum_emission  # shape: (num_states, D)

        delta[0:D] = (PAST_DELTA * EMISSION_PROBS)


        #! PHASE 2 - INDUCTION  t>0   
        AP = self.trans_mat[np.newaxis, :, :] * self.duration_probs[ :, :, np.newaxis]  # (N,N,1) * (N,1,D) -> NxNxD  
        for t in range(1, T):

            #! ----- TOO SLOW -----
            # Slice DELTAS window: shape (N, D) assuming DELTAS is shape (T, N, D)
            #EMISSION_PROBS = np.ones((D, N))
            # for d_val in range(0,  min(D, t)):
            #     segment_indices = np.array(self.obs_seq[t - d_val : t+1], dtype=int)
            #     relevant_probs = self.emission_probs[segment_indices, :]   # DxN
            #     EMISSION_PROBS[d_val, :] = np.prod(relevant_probs, axis=0)
            #! ----- TOO SLOW -----


            segment_indices = self.obs_seq[max(0, t - D + 1):t+1].astype(int)  # shape: (D,)
            relevant_probs = self.emission_probs[segment_indices, :]
            cum_emission = np.cumprod(np.flip(relevant_probs, axis=0), axis=0)  # shape: (D, num_states)
            EMISSION_PROBS[:cum_emission.shape[0],:] = cum_emission  # shape: (num_states, D)

            window = delta[max(0, t-D) : t, :]
            PAST_DELTA[:window.shape[0], :] = window[::-1]

            #* Method A
            # DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBS[np.newaxis, :, :]  # NxNxD
            # RESULT_A = AP #* DELTA_EMISSION  # NxNxD element-wise

            #* Method B
            DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * AP  # (D,N) * (D,N,N) -> DxNxN
            RESULT_B = EMISSION_PROBS[ :, :, np.newaxis] * DELTA_EMISSION  # (N,D) * (D,N,N) -> DxNxN
        
            #! ----- TOO SLOW -----
            # for j in range(N):
            #     plane = RESULT_B[:, j, :]          # shape (N, D) — the j-th plane
                
            #     flat_idx = np.argmax(plane[0:min(t,D),:])      # argmax over flattened (N*D)
            #     d, i = np.unravel_index(flat_idx, plane.shape)  # recover (d, i) coords

            #     if t < D and plane[d, i] < delta[t, j]:
            #         max_vals[j] = delta[t, j]
            #         max_states[j] = delta_state[t, j]
            #         max_durs[j] = delta_dur[t, j]
            #     else:
            #         max_vals[j] = plane[d, i]
            #         max_states[j] = i               
            #         max_durs[j]   = d        

            # delta[t, :] = max_vals 
            # delta_state[t, :] = max_states
            # delta_dur[t, :] = max_durs+1
             #! ----- TOO SLOW -----


            planes = RESULT_B.transpose(1, 0, 2)          # (N, N, D): planes[j] = RESULT_B[:, j, :]

            # Slice rows up to min(t, D) for all j simultaneously
            sliced = planes[:, :min(t, D), :]             # (N, min(t,D), D)

            # Argmax over the flattened (min(t,D), D) sub-matrix for each j
            flat_idx = np.argmax(sliced.reshape(N, -1), axis=1)   # (N,)

            # Unravel flat indices into (d, i) — relative to sliced shape, NOT plane.shape
            slice_shape = sliced.shape[1:]                         # (min(t,D), D)
            d_arr, i_arr = np.unravel_index(flat_idx, slice_shape) # each (N,)

            # Gather the actual max values from the FULL plane (as original code does)
            best_vals = planes[np.arange(N), d_arr, i_arr]        # (N,)

            # Condition: t < D AND best value < delta[t, j]
            if t < D:
                cond = best_vals < delta[t, :]                     # (N,) boolean mask
            else:
                cond = np.zeros(N, dtype=bool)                     # all False → always use else

            # Build outputs with np.where, write back
            delta[t, :]       = np.where(cond, delta[t, :],       best_vals)
            delta_state[t, :] = np.where(cond, delta_state[t, :], i_arr)
            delta_dur[t, :]   = np.where(cond, delta_dur[t, :],   d_arr) + 1

        path = self.backtracking_termination(delta, delta_state, delta_dur, T)
        
        return path
    

    #? We use the official signature of Numpy for 3D tensors (d,y,x)
    #? axis 0 → depth (z) — the first index, selects a 2D "slice"
    #? axis 1 → rows — the second index, selects a row within a slice
    #? axis 2 → columns — the third index, selects a column within a row
    def run_log_tensor_viterbi_cached(self):

        T = len(self.obs_seq)
        N = len(self.states) 
        D = self.duration_probs.shape[0]
        
        delta = np.full((T, N), -np.inf)        
        delta_state = np.zeros((T, N), dtype=int)
        delta_dur = np.zeros((T, N), dtype=int)

        #! PHASE 1 - INITIALIZATION 0<=t<D
        PAST_DELTA = self.duration_probs + self.start_probs[np.newaxis,:]
  
        #* Method A
        # for d in range(0,D):
        #     obs = int(self.obs_seq[d])
        #     self.emission_probs[obs, :]
        #     EMISSION_PROBS[d:,:] *= self.emission_probs[obs, :][:,np.newaxis].T

        #* Method B
        obs_indices = self.obs_seq[:D].astype(int)  # shape: (D,)
        emission_rows = self.emission_probs[obs_indices, :]
        cum_emission = np.cumsum(emission_rows, axis=0)  # shape: (D, num_states)
        EMISSION_PROBS = cum_emission  # shape: (num_states, D)

        delta[0:D] = (PAST_DELTA + EMISSION_PROBS)

        #! PHASE 2 - INDUCTION  t>0   
        AP = self.trans_mat[np.newaxis, :, :] + self.duration_probs[ :, :, np.newaxis]  # (N,N,1) + (N,1,D) = NxNxD  
        EMISSION_CACHE = np.zeros((D,N), dtype=float)
        for t in range(1, T):


            if t>D: 
                _index_t = self.obs_seq[t].astype(int)
                _probs_t = self.emission_probs[_index_t, :] 
                EMISSION_CACHE += _probs_t

                EMISSION_PROBS[:D,:] = EMISSION_CACHE            # shape: (D,N)

                EMISSION_CACHE = np.zeros((D,N), dtype=float)
                EMISSION_CACHE[1:,:] = EMISSION_PROBS[:D-1,:]
            else:
                #* Emission Tensor
                segment_indices = self.obs_seq[max(0, t - D+1):t+1].astype(int)    # shape: (D,)
                relevant_probs = self.emission_probs[segment_indices, :]           # shape: (D,N)
                cum_emission = np.cumsum(np.flip(relevant_probs, axis=0), axis=0)  # shape: (D,N)

                EMISSION_PROBS[:cum_emission.shape[0],:] = cum_emission            # shape: (D,N)

                if t==D:
                    EMISSION_CACHE[1:,:] = cum_emission[:D-1,:]

            
            #* Past Delta Tensor
            window = delta[max(0, t-D) : t, :]
            PAST_DELTA[:window.shape[0], :] = window[::-1]

            #* New Delta(t)
            #* - Method A
            # DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBS[np.newaxis, :, :] 
            # RESULT_A = AP #* DELTA_EMISSION  # NxNxD element-wise

            #* - Method B
            DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] + AP              # (D,N) + (D,N,N) = DxNxN
            RESULT_B = EMISSION_PROBS[ :, :, np.newaxis] + DELTA_EMISSION   # (N,D) + (D,N,N) = DxNxN
        
            planes = RESULT_B.transpose(1, 0, 2)          # (N, N, D): planes[j] = RESULT_B[:, j, :]
            sliced = planes[:, :min(t, D), :]           
            flat_idx = np.argmax(sliced.reshape(N, -1), axis=1)   # (N,)

            slice_shape = sliced.shape[1:]                         # (min(t,D), D)
            d_arr, i_arr = np.unravel_index(flat_idx, slice_shape) # each (N,)

            best_vals = planes[np.arange(N), d_arr, i_arr]        # (N,)

            if t < D:
                cond = best_vals < delta[t, :]                     # (N,) boolean mask
            else:
                cond = np.zeros(N, dtype=bool)                     # all False → always use else

            delta[t, :]       = np.where(cond, delta[t, :],       best_vals)
            delta_state[t, :] = np.where(cond, delta_state[t, :], i_arr)
            delta_dur[t, :]   = np.where(cond, delta_dur[t, :],   d_arr) + 1

        path = self.backtracking_termination(delta, delta_state, delta_dur, T)
        
        return path




    #? We use the official signature of Numpy for 3D tensors (d,y,x)
    #? axis 0 → depth (z) — the first index, selects a 2D "slice"
    #? axis 1 → rows — the second index, selects a row within a slice
    #? axis 2 → columns — the third index, selects a column within a row
    def run_log_tensor_viterbi(self):

        T = len(self.obs_seq)
        N = len(self.states) 
        D = self.duration_probs.shape[0]
        
        delta = np.full((T, N), -np.inf)        
        delta_state = np.zeros((T, N), dtype=int)
        delta_dur = np.zeros((T, N), dtype=int)

        #! PHASE 1 - INITIALIZATION 0<=t<D
        PAST_DELTA = self.duration_probs + self.start_probs[np.newaxis,:]
  
        #* Method A
        # for d in range(0,D):
        #     obs = int(self.obs_seq[d])
        #     self.emission_probs[obs, :]
        #     EMISSION_PROBS[d:,:] *= self.emission_probs[obs, :][:,np.newaxis].T

        #* Method B
        obs_indices = self.obs_seq[:D].astype(int)  # shape: (D,)
        emission_rows = self.emission_probs[obs_indices, :]
        cum_emission = np.cumsum(emission_rows, axis=0)  # shape: (D, num_states)
        EMISSION_PROBS = cum_emission  # shape: (num_states, D)

        delta[0:D] = (PAST_DELTA + EMISSION_PROBS)

        #! PHASE 2 - INDUCTION  t>0   
        AP = self.trans_mat[np.newaxis, :, :] + self.duration_probs[ :, :, np.newaxis]  # (N,N,1) + (N,1,D) = NxNxD  
        for t in range(1, T):

            if(t % 100000 == 0):
                print(t)

            #! ----- TOO SLOW -----
            # EMISSION_PROBS = np.ones((D, N))
            # for d_val in range(0,  min(D, t)):
            #     segment_indices = np.array(self.obs_seq[t - d_val : t+1], dtype=int)
            #     relevant_probs = self.emission_probs[segment_indices, :]   
            #     EMISSION_PROBS[d_val, :] = np.prod(relevant_probs, axis=0)
            #! ----- TOO SLOW -----

            #* Emission Tensor
            segment_indices = self.obs_seq[max(0, t - D+1):t+1].astype(int)  # shape: (D,)
            relevant_probs = self.emission_probs[segment_indices, :]
            cum_emission = np.cumsum(np.flip(relevant_probs, axis=0), axis=0)  # shape: (D, num_states)
            EMISSION_PROBS[:cum_emission.shape[0],:] = cum_emission            # shape: (num_states, D)
            
            #* Past Delta Tensor
            window = delta[max(0, t-D) : t, :]
            PAST_DELTA[:window.shape[0], :] = window[::-1]

            #* New Delta(t)
            #* - Method A
            # DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBS[np.newaxis, :, :] 
            # RESULT_A = AP #* DELTA_EMISSION  # NxNxD element-wise

            #* - Method B
            DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] + AP              # (D,N) + (D,N,N) = DxNxN
            RESULT_B = EMISSION_PROBS[ :, :, np.newaxis] + DELTA_EMISSION   # (N,D) + (D,N,N) = DxNxN
        
            #! ----- TOO SLOW -----
            # for j in range(N):
            #     plane = RESULT_B[:, j, :]          # shape (N, D) — the j-th plane
                
            #     flat_idx = np.argmax(plane[0:min(t,D),:])      # argmax over flattened (N*D)
            #     d, i = np.unravel_index(flat_idx, plane.shape)  # recover (d, i) coords

            #     if t < D and plane[d, i] < delta[t, j]:
            #         max_vals[j] = delta[t, j]
            #         max_states[j] = delta_state[t, j]
            #         max_durs[j] = delta_dur[t, j]
            #     else:
            #         max_vals[j] = plane[d, i]
            #         max_states[j] = i               
            #         max_durs[j]   = d        

            # delta[t, :] = max_vals 
            # delta_state[t, :] = max_states
            # delta_dur[t, :] = max_durs+1
            #! ----- TOO SLOW -----


            planes = RESULT_B.transpose(1, 0, 2)          # (N, N, D): planes[j] = RESULT_B[:, j, :]
            sliced = planes[:, :min(t, D), :]           
            flat_idx = np.argmax(sliced.reshape(N, -1), axis=1)   # (N,)

            slice_shape = sliced.shape[1:]                         # (min(t,D), D)
            d_arr, i_arr = np.unravel_index(flat_idx, slice_shape) # each (N,)

            best_vals = planes[np.arange(N), d_arr, i_arr]        # (N,)

            if t < D:
                cond = best_vals < delta[t, :]                     # (N,) boolean mask
            else:
                cond = np.zeros(N, dtype=bool)                     # all False → always use else

            delta[t, :]       = np.where(cond, delta[t, :],       best_vals)
            delta_state[t, :] = np.where(cond, delta_state[t, :], i_arr)
            delta_dur[t, :]   = np.where(cond, delta_dur[t, :],   d_arr) + 1

        path = self.backtracking_termination(delta, delta_state, delta_dur, T)
        
        return path



    def run_vanilla_viterbi(self):

        T = len(self.obs_seq)  
        N = len(self.states) 
        D = self.duration_probs.shape[0]  
        
        delta = np.full((T, N), -np.inf)
        psi_state = np.zeros((T, N), dtype=int)
        psi_dur = np.zeros((T, N), dtype=int)

        #! PHASE 1 - INITIALIZATION 0<=t<D
        #* delta(0,sj) = pi(sj) * P(d|sj) * |-|{k = t-d}(b(sj, seq_obs(k))
        for state in range(N):
            for d in range(1, D+1):
                obs_score = 0.0
                for tau in range(0,d):
                    obs_index = int(self.obs_seq[tau])
                    obs_score += self.emission_probs[obs_index, state]

                dur_score = self.duration_probs[d-1, state]
                start_prob = self.start_probs[state]
                score = start_prob + dur_score + obs_score
                if score > delta[d-1, state]:
                    delta[d-1, state] = score
                    psi_dur[d-1, state] = d
                    psi_state[d-1, state] = state

        #! PHASE 2 - INDUCTION  t>0
        #* delta(t, sj) = max{d} ( max{si} ( delta(t-d,si) * a(si,sj) ) * P(d|sj) * |-|{k = t-d}(b(sj, seq_obs(k)))  
        for t in range(1, T):
            #print(t)
            for sj in range(N):
                for d in range(1, D+1):
                    if t - d < 0: 
                        continue # Cannot look back past 0 here
                    
                    # |-|{k = t-d}(b(sj, seq_obs(k)
                    obs_score = 0.0
                    for tau in range(t - d, t+1):
                        obs_index = int(self.obs_seq[tau])
                        obs_score += self.emission_probs[obs_index, sj]
                    
                    # P(d|Sj)
                    dur_score = self.duration_probs[d-1, sj]
                    
                    best_prev_score = -np.inf
                    best_prev_state = -1
                    for si in range(N):
                        # HSMMs handle self-loops via duration, Skip impossibile transitions. 
                        #! But with product is necessary, maybe can be inserted but doesn't change much in terms of performance
                        # if si == sj: 
                        #     continue

                        total_score = self.trans_mat[si, sj] + dur_score + delta[t - d, si] + obs_score
   
                        if total_score > best_prev_score:
                            best_prev_score = total_score
                            best_prev_state = si
                    
                    if best_prev_score > delta[t, sj]:
                        delta[t, sj] = best_prev_score
                        psi_state[t, sj] = best_prev_state
                        psi_dur[t, sj] = d

        path = self.backtracking_termination(delta, psi_state, psi_dur, T)
            
        print(delta)

        return path
    


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
    )      
    

    smoothness = 1e-30
    hsmm_sleep = HSMM(
        sleep_states, 
        sleep_emissions, 
        np.log(sleep_trans_mat.T + smoothness), 
        np.log(sleep_emission_probs + smoothness), 
        np.log(sleep_start_probs + smoothness), 
        np.log(sleep_duration_probs.T + smoothness)
    )
    hsmm_sleep.set_obs_sequence(sleep_obs_seq)

    return hsmm_sleep


if __name__ == "__main__":

    data_path = "data/sleep_data_10states_1000000_1000.json"

    hsmm_sleep = load_sleep_model(data_path)
    # hsmm_sleep.print_model()

    # start_time = time.time()
    # v_predicted_states = hsmm_sleep.run_vanilla_viterbi()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Execution time of Vanilla Viterbi: {execution_time:.4f} seconds")

    # validate("Vanilla vs Baseline", v_predicted_states, data_path)

    # np.testing.assert_array_equal(
    #     t_predicted_states,
    #     v_predicted_states,
    #     err_msg="Vanilla is different from Tensor based."
    # )

    hsmm_sleep = load_sleep_model(data_path)

    start_time = time.time()
    t_predicted_states = hsmm_sleep.run_log_tensor_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of Log Tensor Viterbi (NO CACHE): {execution_time:.4f} seconds")

    hsmm_sleep = load_sleep_model(data_path)

    start_time = time.time()
    t_predicted_states = hsmm_sleep.run_log_tensor_viterbi_cached()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of Log Tensor Viterbi: {execution_time:.4f} seconds")
    validate("Tensor vs Baseline", t_predicted_states, data_path)
