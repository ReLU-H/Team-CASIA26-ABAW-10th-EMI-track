# Extract audio-to-text features using Whisper Large model
bash run_audio2text_whisper_large_8gpu_shards.sh

# Extract text embeddings using ChatGLM3
bash run_extract_chatglm3_feats.sh

# Extract acoustic features using WavLM Large model
bash run_extract_wavlm_large_8gpu_shards.sh

# Extract visual features using ViT (Vision Transformer)
bash run_vit_train_and_extract.sh

# Example or ablation study with 8 GPUs
bash v15_ablation.sh
