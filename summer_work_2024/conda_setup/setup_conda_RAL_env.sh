#!/bin/sh
module load conda

conda_loc='../../../data/conda-envs/ral'
mkdir -p $conda_loc


CONDA_OVERRIDE_CUDA="11.8" mamba env create --prefix $conda_loc -f RAL_env.yml

conda activate ral

#mkdir -p $CONDA_PREFIX/etc/conda/activate.d
#mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
mkdir -p $conda_loc/etc/conda/activate.d
mkdir -p $conda_loc/etc/conda/deactivate.d

#cp activate_env_vars.sh $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#cp deactivate_env_vars.sh $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
cp activate_env_vars.sh $conda_loc/etc/conda/activate.d/env_vars.sh
cp deactivate_env_vars.sh $conda_loc/etc/conda/deactivate.d/env_vars.sh

conda deactivate
