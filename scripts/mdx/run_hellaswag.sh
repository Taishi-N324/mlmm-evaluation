#!/bin/bash

cd /model/taishi/mlmm-evaluation/
source venv/bin/activate

export HF_HOME=/model/taishi/.cache
export HF_DATASETS_CACHE=/model/taishi/.cache
export TRANSFORMERS_CACHE=/model/taishi/.cache

lang=$1
model_path=$2
model_id=$(basename $model_path) 
tasks=hellaswag_${lang}
device=cuda
output_dir="results/${model_id}_model_id_${lang}_lang_${tasks}_tasks" 

python main.py \
    --tasks=${tasks} \
    --batch_size 1 \
    --model_args pretrained=$model_path,trust_remote_code=True,use_accelerate=True,dtype="bfloat16" \
    --device=${device} \
    --output_path=${output_dir}  
