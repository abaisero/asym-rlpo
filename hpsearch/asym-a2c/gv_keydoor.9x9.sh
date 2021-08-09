#!/bin/bash

env=$HOME/scratch/asym-rlpo-gv-yaml/gv_keydoor.9x9.yaml
./hpsearch.sh $env

# MILDLY better with the following
# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.001 (better than 0.0001 and 0.0003)
# --optim-lr-critic 0.001 (better than 0.0001 and 0.0003)

# all methods equivalent

exit 0
