#!/bin/bash
# 激活虚拟环境, 请根据实际路径修改
source /home/ubuntu/miniconda3/bin/activate transkv

lm_eval \
    --model hf\
    --model_args pretrained=meta-llama/Meta-Llama-3-8B,trust_remote_code=true\
    --tasks arc_challenge,openbookqa,arc_easy,winogrande,hellaswag,piqa,boolq\
    --batch_size auto:1\
    --output_path /data/TransKV/results/