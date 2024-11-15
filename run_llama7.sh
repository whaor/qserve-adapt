#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[4,8]
##SBATCH --cpus-per-task=8 --gpus=1 --mem=16G
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
module purge
module load anaconda/2021.11 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/12.2.0
source activate QServe
export PYTHONUNBUFFERED=1
export MODEL_PATH=/home/bingxing2/home/scx7kxn/models/llama2-7b

export NUM_GPU_PAGE_BLOCKS=1600
export GLOBAL_BATCH_SIZE=12
export PROMPT_LEN=128
export GENERATION_LEN=3968
python qserve_benchmark.py \
  --model $MODEL_PATH \
  --benchmarking \
  --precision w16a16kv4 \
  --group-size -1


# python qserve_benchmark.py \
#   --model $MODEL_PATH \
#   --benchmarking \
#   --precision w16a16kv8 \
#   --group-size -1
