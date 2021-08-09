#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_nine_rooms.10x10.yaml
./hpsearch.sh $env

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.0003 (better than 0.0001 and 0.001)

exit 0
