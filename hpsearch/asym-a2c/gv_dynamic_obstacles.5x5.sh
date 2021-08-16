#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_dynamic_obstacles.5x5.yaml
./hpsearch.sh $env --wandb-tag gv --max-simulation-timesteps 1_000_000

# --negentropy-value-from 1.0 (better than 0.1 and 10.0)
# --optim-lr-actor 0.0001 (better than 0.0003 and 0.001)

# all methods equivalent

exit 0
