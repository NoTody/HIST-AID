import argparse
import random
import csv
import os
import wandb
import numpy as np
from functools import partial

from dataset import MIMICCXRDataSet
from train_engine import MIMICCXRTrainer
from misc_utils import init_distributed_mode
from models import vit_model, bert_model, mm_model, mm_model_early, mm_model_intermediate

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transformers import AutoTokenizer

from logger import create_logger

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_SILENT"] = "true"

import sys

import urllib3

# increase field limit
csv.field_size_limit(sys.maxsize)

use_gpu = torch.cuda.is_available()
print(use_gpu)

# dimension of the output
nnClassCount = 13

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture']

def get_args():
    parser = argparse.ArgumentParser()
    
    # model parameters
    parser.add_argument("--model_name", type=str, choices=["vitb16"] , default="vitb16")
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--method", type=str , choices=["average", "decoder"], default="decoder")
    parser.add_argument("--fusion_method", type=str , choices=["Block", "ConcatMLP"] ,default="Block")
    parser.add_argument("--pretrained", type=str , choices=["hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"],
                        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--exclude_label", default=False, action='store_true')
    parser.add_argument("--section", type=str , choices=["all", "impression", "finding", "indication"], default="impression")
    parser.add_argument("--mode", type=str , choices=["mm", "img", "text", "mm_early", "mm_late", "mm_intermediate"], default="mm")
    parser.add_argument("--img_time_series", default=False, action='store_true')
    parser.add_argument("--text_time_series", default=False, action='store_true')
    parser.add_argument("--img_max_len", default=5, type=int)
    parser.add_argument("--text_max_len", default=50, type=int)
    parser.add_argument("--lock", default=False, action='store_true')
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_time", default=False, action='store_true')
    parser.add_argument("--gating", default=False, action='store_true')
    parser.add_argument("--use_amp", default=False, action='store_true')
    parser.add_argument("--contrastive", default=False, action='store_true')
    parser.add_argument("--pos_encoding", type=str , choices=["learnable", "fixed", "mixed"], default="learnable")
    parser.add_argument("--img_lr", type=float, default=1e-4)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--unpre_lr", type=float, default=1e-4)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--text_len", type=int, default=200)
    parser.add_argument("--patient", type=int, default=5)
    parser.add_argument("--peft", default=False, action='store_true')
    
    # dataset parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=10)
    
    # dataset path
    parser.add_argument("--train_path", type=str, default='./train.csv')
    parser.add_argument("--val_path", type=str, default='./val.csv')
    parser.add_argument("--test_path", type=str, default='./test.csv')
    
    # wandb
    parser.add_argument("--use_wandb", default=False, action='store_true')
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--project", type=str, default='')
    
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist-backend', default='nccl', help='used to set up distributed backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument("--log_dir", type=str, default="./logger_output")
    
    args = parser.parse_args()
    return args


def collate_fn_batch_encoding(batch, tokenizer, text_len, img_time_series=False):
    images, texts, labels, image_time, text_time = zip(*batch)

    image_encodings = []
    text_encodings = []

    for text in texts:
        while None in text:
            # removing None from list using remove method
            text.remove(None)

        text_encoding = tokenizer(
                list(text),
                max_length=text_len,
                padding="max_length",
                truncation=True,
                return_special_tokens_mask=False,
                return_tensors="pt")
        text_encodings.append(text_encoding)
    
    if img_time_series:
        for idx in range(len(images)):
            image_encodings.append({'imgs': images[idx]})
    else:
        image_encodings = images
    
    return image_encodings, text_encodings, labels, image_time, text_time


def main(args, logger):
    #print(args)
    logger.info(f"Args: {args}")
    
    gpu = torch.device('cuda')
    
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]
    
    # Training settings: batch size, maximum number of epochs
    trBatchSize = args.batch_size
    trMaxEpoch = args.max_epoch
    
    # Tranform data
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(imgtransCrop, 
                                     scale=(0.6, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=(0.4, 0.6), contrast=(0.4, 0.6), saturation=0, hue=0,
                )
            ],
            p=0.5,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize])

    transforms_val = transforms.Compose([
        transforms.Resize((imgtransCrop, imgtransCrop)),
        transforms.ToTensor(),
        normalize])

    token_model = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    tokenizer = AutoTokenizer.from_pretrained(token_model, use_fast=True)

    collate_fn = collate_fn_batch_encoding

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Load dataset
    # Train
    datasetTrain = MIMICCXRDataSet(args, args.train_path, args.img_time_series, \
        transforms_train, policy = "ones")
    logger.info(f"Train data length: {len(datasetTrain)}")
    # Datasampler
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        datasetTrain,
        shuffle=True,
        num_replicas=num_tasks,
        rank=global_rank,
    )

    dataLoaderTrain = DataLoader(dataset=datasetTrain, sampler=sampler_train, batch_size=trBatchSize, 
                                 num_workers=args.num_workers, pin_memory=True,
                                 collate_fn=partial(collate_fn, tokenizer=tokenizer, text_len=args.text_len,
                                                    img_time_series=args.img_time_series))
    # Val
    datasetValid = MIMICCXRDataSet(args, args.val_path, args.img_time_series, \
        transforms_val, policy = "ones")
    logger.info(f"Valid data length: {len(datasetValid)}")
    # Datasampler
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        datasetValid,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank
    )
    dataLoaderVal = DataLoader(dataset=datasetValid, sampler=sampler_val, batch_size=trBatchSize,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=partial(collate_fn, tokenizer=tokenizer, text_len=args.text_len,
                                                  img_time_series=args.img_time_series))
    # Test
    datasetTest = MIMICCXRDataSet(args, args.test_path, args.img_time_series, \
        transforms_val, policy = "ones")
    logger.info(f"Test data length: {len(datasetTest)}")
    # Datasampler
    sampler_test = torch.utils.data.distributed.DistributedSampler(
        datasetTest,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank
    )
    dataLoaderTest = DataLoader(dataset=datasetTest, sampler=sampler_test, batch_size=trBatchSize, 
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=partial(collate_fn, tokenizer=tokenizer, text_len=args.text_len,
                                                   img_time_series=args.img_time_series))

    if args.model_name == "vitb16":
        features_dim = 768
        out_size = nnClassCount
        
        logger.info(f"Use Mode: {args.mode}")
        
        if args.mode == 'mm':
            model = mm_model(args, features_dim, out_size,  \
                fusion_method=args.fusion_method, method=args.method, pretrained=args.pretrained, \
                lock=args.lock, use_time=args.use_time, pos_encoding=args.pos_encoding, \
                img_max_len=args.img_max_len, text_max_len=args.text_max_len).cuda(gpu)
        elif args.mode == 'mm_early':
            model = mm_model_early(args, features_dim, out_size, pretrained=args.pretrained, \
                lock=args.lock, use_time=args.use_time, pos_encoding=args.pos_encoding, \
                img_max_len=args.img_max_len, text_max_len=args.text_max_len).cuda(gpu)
        elif args.mode == 'mm_intermediate':
            model = mm_model_intermediate(args, features_dim, out_size, pretrained=args.pretrained, \
                lock=args.lock, use_time=args.use_time, pos_encoding=args.pos_encoding, \
                img_max_len=args.img_max_len, text_max_len=args.text_max_len).cuda(gpu)
        elif args.mode == 'img':
            model = vit_model(args, features_dim, out_size, pretrained=args.pretrained, \
                method=args.method, lock=args.lock, use_time=args.use_time, \
                pos_encoding=args.pos_encoding).cuda(gpu)
        elif args.mode == 'text':
            model = bert_model(args, features_dim, out_size, pretrained=args.pretrained, method=args.method,
                lock=args.lock, use_time=args.use_time, pos_encoding=args.pos_encoding).cuda(gpu)
        else:
            raise NotImplementedError(f"Mode {args.mode} Not Implemented!")
    else:
        raise NotImplementedError("Model Not Implemented!")
    
    gpu = torch.device('cuda')
    # Convert all BatchNorm layers to SyncBatchNorm layers
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Use DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], \
        broadcast_buffers=False, find_unused_parameters=True)

    logger.info("---------------------Fine-Tuning---------------------")
    logger.info("Training  ...")
    
    # unfreeze for finetuning
    if args.mode == 'img':
        for param in model.module.img_backbone.parameters():
            param.requires_grad = True
    if args.mode == 'mm' or args.mode == 'text':
        for param in model.module.text_backbone.parameters():
            param.requires_grad = True

    best_model = MIMICCXRTrainer.train(args, logger, model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, \
        checkpoint = None, save_suffix = args.save_suffix)

    # switch to evaluate mode
    logger.info("Testing  ...")
    MIMICCXRTrainer.test(args, logger, best_model, dataLoaderTest, nnClassCount, None, class_names)


def set_seed(seed):
    random_seed = args.seed + dist.get_rank()
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    #os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count()))
    # supress warning
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    args = get_args()
    
    if args.use_wandb and args.local_rank == 0:
        wandb.init(
            name = args.run_name,
            project = args.project,
        )
    
    init_distributed_mode(args)

    if args.seed != None:
        set_seed(args.seed)
        
    # logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = create_logger(output_dir=args.log_dir, dist_rank=dist.get_rank(), name=f"{args.save_suffix}")

    main(args, logger)