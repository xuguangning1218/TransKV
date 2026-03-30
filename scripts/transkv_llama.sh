#!/bin/bash

# ================= 设置执行参数 =================
DEVICE="0"
ROOT_DIR="/data/TransKV/" # 请根据实际路径修改
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
SAVE_DIR="${ROOT_DIR}/pca_matrices/LLama3/"
EVAL_OUTPUT_DIR="${ROOT_DIR}/results/transkv_LLama3/"
PRUNED_OUTPUT_DIR="${ROOT_DIR}/pruned_transkv_models/LLama3/"
BATCH_SIZE=64
NUM_WORKERS=4
SEED=42

# 激活虚拟环境, 请根据实际路径修改
source /home/xuguangning/miniconda3/bin/activate transkv

export PYTHONPATH="${ROOT_DIR}/model:${PYTHONPATH}"

DATASETS=("arc_challenge" "arc_easy" "boolq" "hellaswag" "openbookqa" "piqa" "winogrande")

# 定义每个数据集对应的分块大小，32表示每32层计算相应层数的Key和Value的PCA矩阵
declare -A CHUNK_SIZES
CHUNK_SIZES=(
    ["arc_challenge"]=32
    ["arc_easy"]=32
    ["boolq"]=16
    ["hellaswag"]=16
    ["openbookqa"]=16
    ["piqa"]=16
    ["winogrande"]=16
)

# 构建PCA矩阵
for DATASET in "${DATASETS[@]}"
do
    echo "--------------------------------------------------"
    
    # 获取当前数据集对应的分块大小，如果未定义则默认使用 8
    LAYERS_PER_CHUNK=${CHUNK_SIZES[$DATASET]:-8}
    
    echo "Starting PCA extraction process on ${DATASET} with layers_per_chunk=${LAYERS_PER_CHUNK}..."
    
    CUDA_VISIBLE_DEVICES=$DEVICE python ${ROOT_DIR}/model/LLama3/extract_pca.py \
        --dataset ${DATASET} \
        --model_name ${MODEL_NAME} \
        --save_dir ${SAVE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --layers_per_chunk ${LAYERS_PER_CHUNK}
done

echo "Process PCA extraction finished."

# 验证PCA矩阵
for DATASET in "${DATASETS[@]}"
do
    echo "Validating PCA matrix on ${DATASET}..."
    CUDA_VISIBLE_DEVICES=$DEVICE python ${ROOT_DIR}/model/LLama3/test_pca.py \
        --model_name ${MODEL_NAME} \
        --dataset ${DATASET} \
        --pca_path ${SAVE_DIR}/${DATASET}.pt \
        --layer_id 0 \
        --target_type "value" \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --save_plot_path "${ROOT_DIR}/pca_matrices/LLama3/validation_plot_${DATASET}.png"
done

echo "Process Validating PCA matrix finished."

# 使用TransKV剪枝并评估模型
for DATASET in "${DATASETS[@]}"
do
    echo "Pruning LLama3 with TransKV on ${DATASET}..."
    CUDA_VISIBLE_DEVICES=$DEVICE python ${ROOT_DIR}/model/LLama3/test_LLama3.py \
        --task ${DATASET} \
        --model_name ${MODEL_NAME} \
        --pca_path ${SAVE_DIR}/${DATASET}.pt \
        --pruned_output_dir ${PRUNED_OUTPUT_DIR} \
        --eval_output_dir ${EVAL_OUTPUT_DIR} \
        --pruned_ratio 0.5
    
    rm -rf ${PRUNED_OUTPUT_DIR}/
done

echo "Process Pruning LLama3 with TransKV finished."