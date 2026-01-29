


def init():
  # code here

def inductive(states, input_obs, A, B, t_0):
  # code here

  t_vals(len(states), len(input_obs))
  t_vals[0] = t_0

  for t in range(1, len(input_obs)):

    ob = input_obs(pos)
    max_vt = 0


    # questa si può scrivere come grossa computazione matriciale
    for sj in states:
      for si in states:
        for d in D:
          max_vtij = t_vals(si, t-d) * A(i,j) * P(i,j,d)

          if max_vtij > max_vt:
            max_vt = max_vtij
    
    max_vt = max_vt * B(j, ob)

    t_val[t] = max_vt




  for s in states

def backtracking_termination():
  # code here

def main():
  states = ['H','S','M']
  obs = ['R','G', 'B','V']
  input_obs ['R', 'R', 'G', 'B', 'R', 'V']

  pi = [1,1,1]
  
  A = [0.1,0.2,0.7,
       0.4,0.5,0.1,
       0.2,0.2,0.6]

  B = [0.1,0.1,0.3,0.5,
       0.25,0.25,0.25,0.25,
       0.3,0.1,0.3,0.3]

if __name__ == "__main__":
  main()
