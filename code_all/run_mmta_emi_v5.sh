#!/usr/bin/env bash
set -e

python emi_mmta_train_v5.py \
  --train_split train_split.csv \
  --valid_split valid_split.csv \
  --epoch 40 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --audio_dir audio_feats_wavlm_large \
  --face_image_dir vit \
  --wavlm_dir text_feats_ChatGLM3_whisper_large \
  --output_dir outputs_emi_mmta_v5 \
  --device cuda \
  --batch_size 96 \
  --eval 1 \
  --optimizer adamw \
  --seed 42 \
  --patience 8 \
  --dropout 0.2 \
  --num_workers 12 \
  --seq_len 128 \
  --hidden 256 \
  --lam_corr 1 \
  --corr_warmup_epochs 5 \
  --topk_checkpoints 5 \
  --use_amp 1 \
  --fusion_mode sample
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_mmta_emi_v5.sh


# 下面是0.46的参数

# python emi_mmta_train_v5.py \
#   --train_split train_split.csv \
#   --valid_split valid_split.csv \
#   --epoch 40 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --audio_dir audio_feats_wavlm_large \
#   --face_image_dir vit \
#   --wavlm_dir text_feats_ChatGLM3_whisper_large \
#   --output_dir outputs_emi_mmta_v5 \
#   --device cuda \
#   --batch_size 96 \
#   --eval 1 \
#   --optimizer adamw \
#   --seed 42 \
#   --patience 8 \
#   --dropout 0.2 \
#   --num_workers 12 \
#   --seq_len 128 \
#   --hidden 256 \
#   --lam_corr 1 \
#   --corr_warmup_epochs 5 \
#   --topk_checkpoints 5 \
#   --use_amp 1 \
#   --fusion_mode sample