#!/bin/bash -l
#SBATCH --job-name=sfc
#SBATCH --account=NAML0001
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --output=sfc_layer_train.out.%j
export PATH="$HOME/miniconda3/envs/sfc/bin:$PATH"
cd ~/mlsurfacelayer/scripts/
python -u train_surface_models.py ../config/surface_layer_training_fasteddy_cabauw_20201203.yml 
python -u train_surface_models.py ../config/surface_layer_training_fasteddy_idaho_20201203.yml
