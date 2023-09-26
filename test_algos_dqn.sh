#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

algos=(
  # dqn
  adqn
  adqn-vr
  adqn-state
  adqn-state-vr
  adqn-short
  adqn-short-vr
)

envs=(
  # POMDP-heavenhell-episodic-v0
  # POMDP-shopping_5-episodic-v1
  # PO-pos-CartPole-v1
  # "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml"
  # "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml --gv-ignore-state-channel"
  # "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml --gv-ignore-color-channel"
  # "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml --gv-ignore-state-channel --gv-ignore-color-channel"
  "gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml --latent-type beacon-color"
  # gv-yaml/asym-rlpo/gv_memory.5x5.yaml
  # gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml
  # "gv-yaml/asym-rlpo/gv_memory.5x5.yaml --latent-type beacon-color"
  # "gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml --latent-type beacon-color"
  # extra-dectiger-v0
  # extra-cleaner-v0
  # extra-car-flag-v0
)

args=(
  --episode-buffer-prepopulate-timesteps 100
  --max-simulation-timesteps 500
  # --max-simulation-timesteps 5_000
  # --max-simulation-timesteps 5_000_000
  --max-episode-timesteps 100
  --history-model rnn
  # --history-model attention
  # --attention-num-heads 1
  # --truncated-histories-n 10
  # --history-model-memory-size 10
  # --normalize-hs-features
  # --hs-features-dim 64
  # --gv-state-model-type cnn

  --gv-cnn "$PWD/hpsearch/mr-a2c/gv-cnn.v1.yaml"
  --gv-state-submodels agent-grid-cnn
  --gv-state-representation-layers 0
  --gv-observation-submodels grid-cnn
  --gv-observation-representation-layers 0
)

silent=true

if [[ "$1" == "-v" ]]; then
  shift
  echo "running with standard output"
  echo
  silent=false
else
  echo "running without standard output"
  echo
  silent=true
fi

if [[ "$1" == "--debug" ]]; then
  shift
  echo "running with debugging"
  echo
  debug="-m ipdb -c continue"
  silent=false
fi

if [ "$silent" = true ]; then
  outstream="/dev/null";
else
  outstream="/dev/stdout";
fi

warnings="-W ignore"

for env in "${envs[@]}"; do
  for algo in "${algos[@]}"; do
    command="python $warnings $debug ./main_dqn.py $env $algo ${args[*]} $*"
    echo "$command"

    if $command >$outstream; then
      echo "SUCCESS"
    else
      echo "FAIL"
    fi

  done
done

exit 0
