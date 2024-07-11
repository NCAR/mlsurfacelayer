export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

export CUDNN_PATH=${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn
export TENSORRT_PATH=${CONDA_PREFIX}/lib/python3.10/site-packages/tensorrt_libs
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:${TENSORRT_PATH}:${CONDA_PREFIX}/lib/:${LD_LIBRARY_PATH}

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CONDA_PREFIX}

# For CUDA Aware MPI support
export OMPI_MCA_opal_cuda_support=true

# For TF optimizations, see https://github.com/NVIDIA/DeepLearningExamples/issues/57
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2 # 2 (default) or sometimes 4 seems a little better
