#!/usr/bin/env bash
# save this as run_all.sh and make it executable (chmod +x run_all.sh)

# define your hyper‑params
algos=(DQN PPO)
timesteps=(1000000 2000000)
seeds=(0 42 123)

for algo in "${algos[@]}"; do
  for ts in "${timesteps[@]}"; do
    for sd in "${seeds[@]}"; do
      echo "[START] $algo ts=$ts seed=$sd"
      # each call runs in the background
      python assignment.py --algo "$algo" --timesteps "$ts" --seed "$sd" &
    done
  done
done

# wait for all background jobs to finish
wait
echo "[DONE] all trainings complete"
