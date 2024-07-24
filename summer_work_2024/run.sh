#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <arg>"
  exit 1
fi

# VARS
ARG=$1                                                  # Get the argument
LOGFILE="../../data/model_output/${ARG}/timing_log.txt" # Name the time file for recording
LOGFILE2="../../data/model_output/${ARG}/error_log.txt" # Name the time file for recording
# CONDA_ENV="RAL"                                         # Name of your conda environment
# CONDA_ENV="../../data/conda-envs/ral"                   # Name of your conda environment on casper
CREATEKFOLD_SCRIPT="createKfoldYaml.py"
TRAIN_MODEL_SCRIPT="../scripts/train_offshore_models_mvco.py"
DRAW_GRAPHS_SCRIPT="o_mvco_model_analysis.py"
CALC_METRIC_SCRIPT="average_metrics.py"                 # calc average of all Kfold runs

mkdir -p "../../data/model_output/${ARG}"               # Create the parent directories if they do not exist
> $LOGFILE                                              # Clear the log file if it exists
# > $LOGFILE2                                             # Clear the log file if it exists


# module load conda                                     # for casper
# cd mlsurfacelayer                                     # for casper
cd ..                                                   # for local machine
pip install . > /dev/null 2>&1 &
cd summer_work_2024

# Activate the conda environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate $CONDA_ENV

# Loop to call the Python scripts with the argument
echo "creating sub YAML's" | tee -a $LOGFILE && python $CREATEKFOLD_SCRIPT ../config/${ARG}.yml

for i in {0..9}; do
    MODEL_QC_PATH="../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml"
    if [ ! -e "${MODEL_QC_PATH}" ]; then
        echo "Skipping kfold ${i} as ${MODEL_QC_PATH} does not exist"
        continue
    fi

    LOGFILE2="../../data/model_output/${ARG}/error_log_${i}.txt" # Name the time file for recording
    > $LOGFILE2                                             # Clear the log file if it exists
    echo "Timing train_offshore_models_mvco.py for kfold ${i}" | tee -a $LOGFILE && { time (python $TRAIN_MODEL_SCRIPT ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml > $LOGFILE2 2>&1); } >> $LOGFILE 2>&1

    LOGFILE3="../../data/model_output/${ARG}/error_analysis_log_${i}.txt" # Name the time file for recording
    > $LOGFILE3                                             # Clear the log file if it exists
    echo "Timing o_mvco_model_analysis.py for kfold ${i}" | tee -a $LOGFILE && { time (python $DRAW_GRAPHS_SCRIPT ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml > $LOGFILE3 2>&1); } >> $LOGFILE 2>&1
done

echo 'calculating average metrics' | tee -a $LOGFILE && python $CALC_METRIC_SCRIPT ../config/${ARG}.yml

echo 'Timing o_mvco_model_analysis.py for average' | tee -a $LOGFILE && { time (python $DRAW_GRAPHS_SCRIPT ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml --average > $LOGFILE3 2>&1); } >> $LOGFILE 2>&1


# conda deactivate # Deactivate the conda environment