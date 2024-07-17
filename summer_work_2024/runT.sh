



#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <arg>"
  exit 1
fi

# VARS
ARG=$1    

cd ..                                                   # for local machine
pip install . && cd summer_work_2024


# Start time tracking
start_time=$(date +%s)

# Command 1
python testtuner_obj.py "../config/${ARG}.yml" "neural_network" "momentum_flux" -v 2 --project_name tuner_gridkdjfksdj_${PBS_ARRAY_INDEX}_mf --search_type grid #>& $TMPDIR/tuner_grid_out_${PBS_ARRAY_INDEX}_mf_output.txt

# Command 2
# python testtuner_obj.py "../config/${ARG}.yml" "neural_network" "heat_flux" -v 1 --project_name tuner_grid_${PBS_ARRAY_INDEX}_hf --search_type grid #>& $TMPDIR/tuner_grid_out_${PBS_ARRAY_INDEX}_hf_output.txt

# End time tracking
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script execution took $duration seconds." > execution_duration_L2.txt
