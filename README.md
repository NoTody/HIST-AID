# Temporal-MIMIC
This repo contains code for generating time-series radiology images and reports data by linking MIMIC-CXR and 
MIMIC-IV with train/evaluation code for aggregating and fusing representations from both modalities.

## Generation Dataset
Following README in [Generation Dataset](./data_generation) to generate dataset.

## To Run
To run code on one gpu for one machine
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12316 train.py --model_name "vitb16" --batch_size 8 \
    --max_epoch 15 --save_suffix "VIT_MM_1hr_imglen1_textlen50_decoder_ep15_s1000" --seed 1000 \
    --method "decoder" --fusion_method "Block" --num_workers 8 --mode "mm" \
    --train_path "./dataset/train_impressions_1hr_clear.csv" \
    --val_path "./dataset/val_impressions_1hr_clear.csv" \
    --test_path "./dataset/test_impressions_1hr_clear.csv" \
    --section "impression" --local_rank 0 --pos_encoding "learnable" --use_time \
    --img_lr 1e-5 --unpre_lr 1e-4 --text_lr 1e-5 --decoder_layers 3 --patient 5 \
    --run_name "VIT_MM_1hr_imglen1_textlen50_decoder_ep15_s1000" --project "Temporal_MIMIC" --text_len 200 \
    --text_time_series --img_max_len 1 --text_max_len 50 --grad_clip 3.0
```
where fusion method is Block (--fusion_method "Block"), multi-modal fusion (--mode "mm") is used, only 
impression section is used for text (--section "impression"), only text time series is used (--text_time_series)
with text maximum length to be 50 (--text_max_len 50) and image maximum length to be 1 (--img_max_len 1). 
See arguments in train.py for all argument options.

## Reference
Haoxu Huang, Cem M. Deniz, Kyunghyun Cho, Sumit Chopra, Divyam Madaan. "Temporal Fine-tuning of Medical Vision-Language Representations". In: NeurIPS Workshop on Medical Imaging Meets NeurIPS, New Orleans, LA, USA, 2023