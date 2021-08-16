#!/bin/bash

env=PO-pos-CartPole-v1
./hpsearch.sh $env --wandb-tag openai

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.0003 (better than 0.0001 and 0.001)
# --optim-lr-critic 0.0001 (better than 0.0003 and 0.001)

# here, reactive policies work poorly, but asymmetric ones work well.  this is
# the only case which contradicts my hypothesis of (reactive=good <=>
# state-only=good).

exit 0
