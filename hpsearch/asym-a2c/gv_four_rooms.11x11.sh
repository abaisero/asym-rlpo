#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_four_rooms.11x11.yaml
./hpsearch.sh $env

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.003 (better than 0.0001 and 0.001)

# all methods equivalent

exit 0
