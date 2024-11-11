# Temporal-MIMIC
This repository contains code for generating time-series radiology images and reports data by linking MIMIC-CXR and 
MIMIC-IV with train/evaluation code for aggregating and fusing representations from both modalities.

## Dependencies
Check requirements.txt for dependencies of this repository or install them by running
```
pip install -r ./requirements.txt
```

## Generation Dataset
Following README in [Generation Dataset](./data_generation) to generate dataset. The pre-processed datasets we used can be found at https://drive.google.com/drive/folders/15R5lcOg-mKjR2mBQAZ0oLEZCeJ94ZA8P?usp=sharing

## To Run
To run code on one gpu for one machine
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12323 train.py --model_name "vitb16" --batch_size 8 \
    --max_epoch 15 --save_suffix "VIT_early_width768_1hr_imglen1_textlen50_decoder_rope_ep15_s42" --seed 42 \
    --method "decoder" --num_workers 8 --mode "mm_early" \
    --train_path "./dataset/train.csv" \
    --val_path "./dataset/val.csv" \
    --test_path "./dataset/test.csv" \
    --section "impression" --local_rank 0 --pos_encoding "rope" --use_time \
    --img_lr 1e-5 --unpre_lr 1e-4 --text_lr 1e-5 --decoder_layers 3 --patient 15 \
    --run_name "VIT_early_width768_1hr_imglen1_textlen50_decoder_rope_ep15_s42" --project "HAIM" --text_len 200 \
    --text_time_series --img_max_len 1 --text_max_len 50 --grad_clip 3.0 --d_model 768 
```
where fusion method is Block (--fusion_method "Block"), early multi-modal fusion (--mode "mm_early") is used,  
impression section is used for text (--section "impression"), text time series is used (--text_time_series)
with text maximum length to be 50 (--text_max_len 50) and image maximum length to be 1 (--img_max_len 1). 
See arguments in train.py for all argument options.

## Reference
Haoxu Huang, Cem M. Deniz, Kyunghyun Cho, Sumit Chopra, Divyam Madaan. "HIST-AID: Leveraging Historical Patient Reports for Enhanced Multi-Modal Automated Diagnosis". In Proceedings of Machine Learning for Health, Vancouver, BC, Canada, 2024