#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs-full/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load python/3.10/3.10.10
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

export HF_HOME=/bb/llm/gaf51275/jalm/taishi/.cache
export HF_DATASETS_CACHE=/bb/llm/gaf51275/jalm/taishi/.cache
export TRANSFORMERS_CACHE=/bb/llm/gaf51275/jalm/taishi/.cache

source venv/bin/activate

lang=$1
model_path=$2
tasks=arc_${lang},hellaswag_${lang},mmlu_${lang}
device=cuda

python main.py \
    --tasks=${tasks} \
    --batch_size 1 \
    --limit 20 \
    --model_args pretrained=$model_path,trust_remote_code=True,use_accelerate=True,dtype="bfloat16" \
    --device=${device}