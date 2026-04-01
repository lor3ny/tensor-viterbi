#!/bin/bash

# Define parameter arrays
states=(10 15 25 50 75)
durations=(100 250 500 1000)
timesteps=(1000 10000 100000)


# Loop through all combinations
for s in "${states[@]}"; do
    for d in "${durations[@]}"; do
        for t in "${timesteps[@]}"; do
            
            # Define a unique filename for this combination
            config_file="data/${s}states_${t}steps_${d}dur.json"
            
            # Create the JSON file
            cat <<EOF > "$config_file"
{
  "states": $s,
  "duration": $d,
  "timesteps": $t
}
EOF

            # Submit the job to SLURM, passing the path as an argument
            echo "Submitting job for: State=$s, Duration=$d, Timesteps=$t"
            sbatch run.slrm "$config_file"
            
        done
    done
done