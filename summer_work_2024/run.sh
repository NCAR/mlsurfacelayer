#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <arg>"
  exit 1
fi

# Get the argument
ARG=$1

# Name of your conda environment
CONDA_ENV="RAL"

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Loop to call the Python scripts with the argument
# for i in {0..10}; do
for i in {0..9}; do
    python createKfoldYaml.py ../config/${ARG}.yml
    python ../scripts/train_offshore_models_mvco.py ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml 
    python o_mvco_model_analysis.py ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml 
done

# Deactivate the conda environment
conda deactivate