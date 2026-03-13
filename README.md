# 1. 提取音频转文本特征 (Whisper Large)
bash run_audio2text_whisper_large_8gpu_shards.sh

# 2. 提取文本特征 (ChatGLM3)
bash run_extract_chatglm3_feats.sh

# 3. 提取声学特征 (WavLM Large)
bash run_extract_wavlm_large_8gpu_shards.sh

# 4. 提取视觉特征 (ViT)
bash run_vit_train_and_extract.sh

# 使用 8 块 GPU 进行验证测试
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash v15_ablation.sh
