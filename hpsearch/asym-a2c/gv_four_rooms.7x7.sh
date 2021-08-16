#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_four_rooms.7x7.yaml
./hpsearch.sh $env --wandb-tag gv --max-simulation-timesteps 1_000_000

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)

# all methods equivalent

exit 0
