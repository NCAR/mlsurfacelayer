# kfold cross validation script

# Check if an argument is provided
# The arg is the experiment name
if [ -z "$1" ]; then
  echo "Usage: $0 <arg>"
  exit 1
fi

# Get the experiement name 
ARG=$1                     

# make a directory for the experiement output
mkdir -p "../../data/model_output/${ARG}"      

# Define log files for the experment
LOGFILE="../../data/model_output/${ARG}/timing_log.txt" # Log file 

# Define the conda env
CONDA_ENV="/glade/work/dettling/conda-envs/mlsl_env"  

# Script to create an indepenedent yaml file for every fold
CREATEKFOLD_SCRIPT="createKfoldYaml.py"

# Model trainer
TRAIN_MODEL_SCRIPT="../scripts/train_offshore_models_mvco.py"

# Automated image generator of results: scatter plots, predictor importances
DRAW_GRAPHS_SCRIPT="o_mvco_model_analysis.py"

# Average stats generator over all folds
CALC_METRIC_SCRIPT="average_metrics.py"                

# Clear the logfile
> $LOGFILE                                              

# Load conda env
module load conda
conda activate $CONDA_ENV

# Create yaml config files for each kfold 
# This should create the model output directories for each fold
echo "creating kfold YAML's" | tee -a $LOGFILE && python $CREATEKFOLD_SCRIPT ../config/${ARG}.yml

# Execute the model trainer for each fold
for i in {0..4}; do
    # This allows the assumed folds here to be >= the folds in the config *.yml but they shoud be the same
    MODEL_QC_PATH="../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml"
    if [ ! -e "${MODEL_QC_PATH}" ]; then
        echo "Skipping kfold ${i} as ${MODEL_QC_PATH} does not exist"
        continue
    fi
	
    # If for some reason k-fold model making and testing fails, this allows to restart and not regenerate folds that were complete
    FILE="../../data/model_output/${ARG}/model_QC_--kfold-${i}/surface_layer_model_predictions.csv"
    if [ -f "$FILE" ]; then
	echo "skipping this fold bc it already exists"
	continue
    fi

    # Train models: Define log, clear it, train the model
    LOGFILE2="../../data/model_output/${ARG}/error_log_${i}.txt" 
    > $LOGFILE2                                             
    echo "Timing train_offshore_models_mvco.py for kfold ${i}" | tee -a $LOGFILE && { time (python $TRAIN_MODEL_SCRIPT ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml > $LOGFILE2 2>&1); } >> $LOGFILE 2>&1

    # Images: Create logfile, clear it, run the image gen
    LOGFILE3="../../data/model_output/${ARG}/error_analysis_log_${i}.txt" 
    > $LOGFILE3                                             # Clear the log file if it exists
    echo "Timing o_mvco_model_analysis.py for kfold ${i}" | tee -a $LOGFILE && { time (python $DRAW_GRAPHS_SCRIPT ../../data/model_output/${ARG}/${ARG}--kfold-${i}.yml > $LOGFILE3 2>&1); } >> $LOGFILE 2>&1
done

# Average metrics: compute metrics across all folds
echo 'calculating average metrics' | tee -a $LOGFILE && python $CALC_METRIC_SCRIPT ../config/${ARG}.yml
