we used the below bash to extract the features.
bash run_audio2text_whisper_large_8gpu_shards.sh
bash run_extract_chatglm3_feats.sh
bash run_extract_wavlm_large_8gpu_shards
bash run_vit_train_and_extract.sh
then we use this to test the validation,and count the average pearson.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash v15_ablation.sh
