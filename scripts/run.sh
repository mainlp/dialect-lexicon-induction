#!/bin/bash

# source: https://stackoverflow.com/a/246128 (CC-BY-SA 4.0)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR=$SCRIPT_DIR/../src

DATASET=$1
EXPERIMENT=$2
SEEDS=(1 2 3)

if [[ "$DATASET" == "dialemma" && "$EXPERIMENT" == "ablation" ]]; then
  SEEDS=( {1..40} )
fi

echo "Running ${#SEEDS[@]} random seeds"

# compute results for individual random seeds
for SEED in "${SEEDS[@]}"; do
    echo "Processing random_seed=$SEED ($EXPERIMENT)"
    python $SRC_DIR/run_${DATASET}_bli.py --random_seed $SEED --experiment $EXPERIMENT
done

# aggregate results accross random seeds
python $SRC_DIR/run_${DATASET}_bli.py --experiment $EXPERIMENT\_summarize
