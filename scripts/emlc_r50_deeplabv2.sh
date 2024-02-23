#/bin/bash

# baseline no mask
CUDA_VISIBLE_DEVICES=2 python ./evaluate.py \
--layers 50 \
--dataset_path dataset/ScribbleSup/VOC2012 \
--dataset VOC2012 \
--numclasses 21 \
--workers 2 \
--model_type res50_CAMATT \
--shrink_factor 1 \
--checkpoint_path log/train_deeplabv2_r50/last_checkpoint.pth \
--save_path emlc_r50_deeplabv2