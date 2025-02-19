#!/bin/bash

EPOCH_TIME=150
CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
DATA_DIR='../data/bcss/L0_1024_s512'
LEARNING_RATE=0.05
IMG_SIZE=224
BATCH_SIZE=24

# OUT_DIR='./logs/temp'
# python train.py --dataset bcss --cfg $CFG \
#     --root_path $DATA_DIR --max_epochs $EPOCH_TIME \
#     --output_dir $OUT_DIR --img_size $IMG_SIZE \
#     --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE

# OUT_DIR='./logs/frac'
# python train.py --dataset bcss --cfg $CFG \
#     --root_path $DATA_DIR --max_epochs $EPOCH_TIME \
#     --output_dir $OUT_DIR --img_size $IMG_SIZE \
#     --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --frac 0.5

# OUT_DIR='./logs/frac_1'
# python train.py --dataset bcss --cfg $CFG \
#     --root_path $DATA_DIR --max_epochs $EPOCH_TIME \
#     --output_dir $OUT_DIR --img_size $IMG_SIZE \
#     --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --frac 0.1

# OUT_DIR='./logs/frac_2'
# python train.py --dataset bcss --cfg $CFG \
#     --root_path $DATA_DIR --max_epochs $EPOCH_TIME \
#     --output_dir $OUT_DIR --img_size $IMG_SIZE \
#     --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --frac 0.01

FOLD=(0 1 2 3 4)
FRAC=(1.0 0.5 0.1 0.01)

###### Swin-UNet
for frac in ${FRAC[@]}
do
    for fold in ${FOLD[@]}
    do
        OUT_DIR="./logs/frac_${frac}/fold_${fold}"

        CUDA_VISIBLE_DEVICES=0 python train.py --dataset bcss --cfg $CFG \
            --root_path $DATA_DIR --max_epochs $EPOCH_TIME \
            --output_dir $OUT_DIR --img_size $IMG_SIZE \
            --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE \
            --frac ${frac} --fold ${fold}
    done
done
