#!/bin/bash

# ============== #
# CONFIGURE THIS #
# ============== #

SEED=1
# one of {"C3", "bayeslr","twotower", "swlinucb", "dlinucb", "gpforget", "onlinenucb", "onlinents", "crb"}
ALGORITHM=C3

# ================================================== #

EXP_DIR="results/mind/$ALGORITHM"

python3 scripts/eval_mind.py $EXP_DIR $ALGORITHM "configs/mind.json" "configs/mindargs/$ALGORITHM.json" \
    "data/MINDlarge_train" "data/MINDlarge_dev" "vectors" --seed $SEED
