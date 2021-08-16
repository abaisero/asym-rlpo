#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_nine_rooms.16x16.yaml
./hpsearch.sh $env --wandb-tag gv

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.0003 (better than 0.0001 and 0.001)

exit 0
