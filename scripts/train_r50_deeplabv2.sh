#/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./train.py \
--layers 50 \
--model_type res50_cam \
--dataset_path dataset/SBD_leige \
--dataset SBD_pesudo_distancemap \
--train_path pascal_2012_scribble \
--distance_map_Scrib distance_map \
--distance_map_pseudo distance_pseudo \
--l_seg 1 \
--l_pesudo 1 \
--l_CAMATT 1 \
--label_smooth 0.1 \
--distanceScrib_entropy 1 \
--distanceCAM_entropy 1 \
--labelFusion_entropy 0 \
--numclasses 21 \
--workers 6 \
--batchsize 16 \
--lr 1e-3 \
--warmup 5 \
--wdecay 5e-4 \
--momentum 0.9 \
--epochs 50 \
--logdir ./log/train_r50_deeplabv2