#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_dynamic_obstacles.9x9.yaml
./hpsearch.sh $env --max-simulation-timesteps 500_000

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.0001 (better than 0.0003 and 0.001)

# all methods equivalent

exit 0
