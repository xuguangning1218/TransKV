#!/bin/bash

# ================= 设置执行参数 =================
DEVICE="cuda:0"
ROOT_DIR="/data/TransKV/" # 请根据实际路径修改
MODEL_NAME="google/vit-base-patch16-224"
DATA_PATH="/data/CLOVER-main/data/ImageNet/" # 请根据实际路径修改
SAVE_DIR="${ROOT_DIR}/pca_matrices/ViT/"
SAMPLE_RATIO=0.05
BATCH_SIZE=512
NUM_WORKERS=4
SEED=42

# 激活虚拟环境, 请根据实际路径修改
source /home/ubuntu/miniconda3/bin/activate transkv

export PYTHONPATH="${ROOT_DIR}/model:${PYTHONPATH}"

# 构建PCA矩阵
echo "Starting PCA extraction process on ImageNet..."
python ${ROOT_DIR}/model/ViT/extract_pca.py \
    --device ${DEVICE} \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --save_dir ${SAVE_DIR} \
    --sample_ratio ${SAMPLE_RATIO} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo "Process PCA extraction finished."

# 验证PCA矩阵
echo "Validating PCA matrix on ImageNet..."
python ${ROOT_DIR}/model/ViT/test_pca.py \
    --device ${DEVICE} \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --pca_path ${SAVE_DIR}/ImageNet_5.pt \
    --layer_id 0 \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --save_plot_path "${ROOT_DIR}/pca_matrices/ViT/validation_plot.png"

echo "Process Validating PCA matrix finished."

# 使用TransKV剪枝并评估模型
echo "Pruning ViT with TransKV on ImageNet..."
python ${ROOT_DIR}/model/ViT/test_ViT.py \
    --device ${DEVICE} \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --pca_path ${SAVE_DIR}/ImageNet_5.pt \
    --batch_size ${BATCH_SIZE} \
    --pruned_ratio 0.5

echo "Process Pruning ViT with TransKV finished."