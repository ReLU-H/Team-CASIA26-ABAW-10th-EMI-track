# 1. extract different modality features
bash run_audio2text_whisper_large_8gpu_shards.sh
bash run_extract_chatglm3_feats.sh
bash run_extract_wavlm_large_8gpu_shards.sh
bash run_vit_train_and_extract.sh

# 2.get average pearson scores on validation set
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash v15_ablation.sh
