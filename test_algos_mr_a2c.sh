#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

algos=(
  mr-a2c
)

envs=(
  POMDP-heavenhell_2-episodic-v0
  # POMDP-shopping_5-episodic-v1
  # PO-pos-CartPole-v1
  # gv_yaml/gv_four_rooms.7x7.yaml
  # gv_yaml/gv_memory.5x5.yaml
  # gv_yaml/gv_memory_four_rooms.7x7.yaml
  # "gv_yaml/gv_memory.5x5.yaml --latent-type GV-MEMORY"
  # "gv_yaml/gv_memory_four_rooms.7x7.yaml --latent-type GV-MEMORY"
  # extra-dectiger-v0
  # extra-cleaner-v0
  # extra-car-flag-v0
)

envs=(
  POMDP-heavenhell_2-episodic-v0
  PO-pos-CartPole-v1
  extra-dectiger-v0
  extra-cleaner-v0
  extra-car-flag-v0
  gv_yaml/gv_four_rooms.7x7.yaml
)

args=(
  --max-simulation-timesteps 500
  # --max-simulation-timesteps 5_000
  # --max-simulation-timesteps 5_000_000
  --max-episode-timesteps 100
  --simulation-num-episodes 2
  --history-model rnn
  # --history-model attention
  # --attention-num-heads 1
  # --truncated-histories-n 10
  # --normalize-hs-features
  # --hs-features-dim 64
  # --gv-state-model-type cnn
  --gv-state-grid-model-type fc
  --gv-state-representation-layers 2
  --gv-observation-grid-model-type fc
  --gv-observation-representation-layers 2
  --negentropy-value-from 0.1
  --negentropy-value-to 0.01
  --negentropy-nsteps 1_000_000
)

warnings="-W ignore"
# warnings=""

if [[ "$1" == "-v" ]]; then
  shift
  echo "running with standard output"
  echo
  silent=0
else
  echo "running without standard output"
  echo
  silent=1
fi

debug=""

if [[ "$1" == "--debug" ]]; then
  shift
  echo "running with debugging"
  echo
  debug="-m ipdb -c continue"
  silent=0
fi

for env in "${envs[@]}"; do
  for algo in "${algos[@]}"; do
    cmd="python $warnings $debug ./main_mr_a2c.py $env $algo ${args[@]} $@"
    echo $cmd
    [[ "$silent" -eq 1 ]] && $cmd > /dev/null || $cmd
    [[ "$?" -eq 0 ]] && echo "SUCCESS" || echo "FAIL"
  done
done

exit 0
