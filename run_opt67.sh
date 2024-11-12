#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[4,8]
#SBATCH --gpus=1

module purge
module load anaconda/2021.11 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/12.2.0
source activate QServe
export PYTHONUNBUFFERED=1
export MODEL_PATH=/home/bingxing2/home/scx7kxn/models/opt-6.7b


export GLOBAL_BATCH_SIZE=20
export PROMPT_LEN=128
export GENERATION_LEN=1920
python qserve_benchmark.py \
  --model $MODEL_PATH \
  --benchmarking \
  --precision w16a16kv4 \
  --group-size -1