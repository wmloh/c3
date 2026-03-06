#!/bin/bash

# ============== #
# CONFIGURE THIS #
# ============== #

SEED=0
DATASET=shuttle           # one of {"cover", "magic", "mnist", "shuttle"}
ALGORITHM=C3  # one of {"C3", "linucb", "neuralucb", "lts", "neuralts", "squarecb", "neurallinear"}

# ================================================== #

CONFIG_PATH="configs/$DATASET.json"
EXP_DIR="results/$DATASET/$ALGORITHM"

python3 scripts/eval_openml.py $EXP_DIR $ALGORITHM $CONFIG_PATH --seed $SEED
