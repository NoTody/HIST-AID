import os
import copy
import time
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from misc_utils import remove_nan_gradients, compute_AUCs
from lr_sched_utils import get_cosine_schedule_with_warmup


class MIMICCXRTrainer():

    def train(args, logger, model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, save_suffix):
        best_model = copy.deepcopy(model)
        
        # different learning rate for encoder and decoder (since decoder is not pretrained)
        world_size = int(os.environ['WORLD_SIZE'])
        effective_batch_size = args.batch_size * world_size
        lr_mult = effective_batch_size / 64
        
        logger.info(f"World Size: {world_size}, Effective Batch Size: {effective_batch_size}")
        
        # image parameters
        img_pretrained_params = [param for name, param in model.module.named_parameters() \
            if 'decoder' not in name and 'img_backbone' in name ]
        # text parameters
        text_pretrained_params = [param for name, param in model.module.named_parameters() \
            if 'decoder' not in name and 'text_backbone' in name]
        # not pretrained parameters
        unpretrained_params = [param for name, param in model.module.named_parameters() \
            if 'decoder' in name or ('img_backbone' not in name and 'text_backbone' not in name)]
        
        # ignore text optimizer to avoid no gradient error when using amp
        if args.img_time_series and (not args.text_time_series):
            text_pretrained_params = []

        # warmup and training steps
        num_training_steps = len(dataLoaderTrain) * trMaxEpoch
        num_warmup_steps = int(0.1 * num_training_steps)
        
        optimizers = []
        schedulers = []

        betas = (0.9, 0.999)
        weight_decay = 0.01
        eps = 1e-5

        if img_pretrained_params != [] and (not args.lock):
            # optimizer
            img_pretrained_optimizer = optim.AdamW(img_pretrained_params, lr = args.img_lr * lr_mult, \
                betas = betas, eps = eps, weight_decay = weight_decay)
            optimizers.append(img_pretrained_optimizer)
            # scheduler
            img_pretrained_scheduler = get_cosine_schedule_with_warmup(img_pretrained_optimizer, num_warmup_steps, \
                num_training_steps, lr_end = args.img_lr * 1e-3 * lr_mult)
            schedulers.append(img_pretrained_scheduler)
        
        if text_pretrained_params != [] and (not args.lock):
            # optimizer
            text_pretrained_optimizer = optim.AdamW(text_pretrained_params, lr = args.text_lr * lr_mult, \
                betas = betas, eps = eps, weight_decay = weight_decay)
            optimizers.append(text_pretrained_optimizer)
            # scheduler
            text_pretrained_scheduler = get_cosine_schedule_with_warmup(text_pretrained_optimizer, num_warmup_steps, \
                num_training_steps, lr_end = args.text_lr * 1e-3 * lr_mult)
            schedulers.append(text_pretrained_scheduler)
            
        if unpretrained_params != []:
            # optimizer
            unpretrained_optimizer = optim.AdamW(unpretrained_params, lr = args.unpre_lr * lr_mult, \
                betas = betas, eps = eps, weight_decay = weight_decay)
            optimizers.append(unpretrained_optimizer)
            # scheduler
            unpretrained_scheduler = get_cosine_schedule_with_warmup(unpretrained_optimizer, num_warmup_steps, \
                num_training_steps, lr_end = args.unpre_lr * 1e-3 * lr_mult)
            schedulers.append(unpretrained_scheduler)

        # loss
        criterion = torch.nn.BCEWithLogitsLoss()

        # Train the network
        save_path = './model_saved/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
                
        aurocMAX = -1
        patient_count = 0
        train_start = []
        train_end = []
        for epochID in range(0, trMaxEpoch):
            train_start.append(time.time()) # training starts
            losst = MIMICCXRTrainer.epochTrain(args, model, dataLoaderTrain, optimizers, schedulers, criterion)
            train_end.append(time.time()) # training ends
            
            logger.info(f"Training loss: {losst},")
            
            aurocMean, lossv = MIMICCXRTrainer.epochVal(args, model, dataLoaderVal, criterion, nnClassCount)
            
            if args.contrastive:
                save_str = 'save'
                best_model = copy.deepcopy(model)
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_auroc': aurocMAX}, 
                            save_path + 'm-epoch_FL' + save_suffix + '.pth.tar')
            else:
                save_str = '----'
                if aurocMean > aurocMAX:
                    aurocMAX = aurocMean
                    best_model = copy.deepcopy(model)
                    torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                                'best_auroc': aurocMAX}, 
                                save_path + 'm-epoch_FL' + save_suffix + '.pth.tar')
                    save_str = 'save'
                    patient_count = 0
                else:
                    patient_count += 1
                
            logger.info(f"Epoch: {str(epochID + 1)} [{save_str}] validation auroc = {str(aurocMean)}, \
                validation loss = {str(lossv)}")
            
            if args.use_wandb and args.local_rank == 0:
                wandb.log({"Training Loss": losst, "Validation Loss": lossv, \
                        "Validation AUROC": aurocMean})
            
            # Early stopping condition
            if patient_count >= args.patient:
                logger.info(f"Early stopping after {epochID + 1} epochs.")
                break
                
        train_time = np.array(train_end) - np.array(train_start)
        logger.info("Train and Validation time for each epoch: {} seconds".format(train_time.round(0)))
        
        return best_model


    def epochTrain(args, model, dataLoaderTrain, optimizers, schedulers, criterion):
        losstrain = 0
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        model.train()

        with tqdm(total=len(dataLoaderTrain), desc=f'Epoch', unit='batch') as pbar:
            for batchID, batch in enumerate(dataLoaderTrain):
                for idx, optimizer in enumerate(optimizers):
                    optimizer.zero_grad()
                    
                    if args.use_wandb and args.local_rank == 0:
                        wandb.log({f"optimizer ({idx}) lr": optimizer.param_groups[0]['lr']})
                        
                x_img, x_text, varTarget, image_time, text_time = batch
                
                if args.img_time_series:
                    for img in x_img:
                        for key in img.keys():
                            img[key] = img[key].cuda(non_blocking=True)
                else:
                    x_img = torch.stack(x_img).cuda(non_blocking=True)

                varTarget = torch.stack(varTarget).cuda(non_blocking=True)

                for text in x_text:
                    for key in text.keys():
                        text[key] = text[key].cuda(non_blocking=True)
                
                # mix-precision
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    if 'mm' in args.mode:
                        varOutput = model(x_img, x_text, image_time, text_time)
                    elif args.mode == 'img':
                        varOutput = model(x_img, image_time)
                    elif args.mode == 'text':
                        varOutput = model(x_text, text_time)
                    else:
                        raise NotImplementedError(f"Mode {args.mode} Not Implemented!")
                    
                # avoid diverge
                varOutput["out"] = varOutput["out"].float()
                lossvalue = criterion(varOutput["out"], varTarget)
                
                pbar.set_postfix(loss=f'{lossvalue.item():.4f}')
                pbar.update(1)
                
                # backward
                if not torch.isnan(lossvalue):
                    scaler.scale(lossvalue).backward()
                
                losstrain += lossvalue.item()
                
                # unscale
                for idx in range(len(optimizers)):
                    scaler.unscale_(optimizers[idx])
                    
                remove_nan_gradients(model)
                
                # gradient clipping
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # step
                for idx in range(len(optimizers)):
                    scaler.step(optimizers[idx])
                    schedulers[idx].step()
                
                scaler.update()
                torch.cuda.synchronize()
        
        return losstrain / len(dataLoaderTrain)
    
    
    def epochVal(args, model, dataLoaderVal, criterion, nnClassCount):
        lossval = 0
        model.eval()
        
        sigmoid = torch.nn.Sigmoid()
        
        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()
        
        with torch.no_grad():
            for batchID, batch in enumerate(tqdm(dataLoaderVal)):
                x_img, x_text, varTarget, image_time, text_time = batch
                
                if args.img_time_series:
                    for img in x_img:
                        for key in img.keys():
                            img[key] = img[key].cuda(non_blocking=True)
                else:
                    x_img = torch.stack(x_img).cuda(non_blocking=True)

                varTarget = torch.stack(varTarget).cuda(non_blocking=True)
                outGT = torch.cat((outGT, varTarget.cpu()), 0)

                for text in x_text:
                    for key in text.keys():
                        text[key] = text[key].cuda(non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    if 'mm' in args.mode:
                        varOutput = model(x_img, x_text, image_time, text_time)
                    elif args.mode == 'img':
                        varOutput = model(x_img, image_time)
                    elif args.mode == 'text':
                        varOutput = model(x_text, text_time)
                    else:
                        raise NotImplementedError(f"Mode {args.mode} Not Implemented!")
                
                if args.contrastive:
                    img_feat, text_feat = varOutput['img_feat'], varOutput['text_feat']
                    lossvalue = criterion(img_feat, text_feat)
                else:
                    # avoid diverge
                    varOutput["out"] = varOutput["out"].float()
                    lossvalue = criterion(varOutput["out"], varTarget)
                
                lossval += lossvalue.item()
                
        aurocIndividual = compute_AUCs(outGT, outPRED.detach(), nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        return aurocMean, lossval / len(dataLoaderVal)
    
    
    def test(args, logger, model, dataLoaderTest, nnClassCount, class_names):
        model.eval()
        
        losstest = 0
        sigmoid = torch.nn.Sigmoid()
        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()

        # loss
        criterion = torch.nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batchID, batch in enumerate(tqdm(dataLoaderTest)):
                x_img, x_text, varTarget, image_time, text_time = batch
                
                if args.img_time_series:
                    for img in x_img:
                        for key in img.keys():
                            img[key] = img[key].cuda(non_blocking=True)
                else:
                    x_img = torch.stack(x_img).cuda(non_blocking=True)

                varTarget = torch.stack(varTarget).cuda(non_blocking=True)
                outGT = torch.cat((outGT, varTarget.cpu()), 0)

                for text in x_text:
                    for key in text.keys():
                        text[key] = text[key].cuda(non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    if 'mm' in args.mode:
                        varOutput = model(x_img, x_text, image_time, text_time)
                    elif args.mode == 'img':
                        varOutput = model(x_img, image_time)
                    elif args.mode == 'text':
                        varOutput = model(x_text, text_time)
                    else:
                        raise NotImplementedError(f"Mode {args.mode} Not Implemented!")
                
                varOutput["out"] = varOutput["out"].float()
                lossvalue = criterion(varOutput["out"], varTarget)

                losstest += lossvalue.item()
        
        aurocIndividual = compute_AUCs(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        #print('AUROC mean ', aurocMean)
        logger.info(f"AUROC Mean: {aurocMean}")
        
        for i in range (0, len(aurocIndividual)):
            logger.info(f"{class_names[i]}: {aurocIndividual[i]}")
    
        if args.use_wandb and args.local_rank == 0:
            wandb.log({"Test AUROC": aurocMean})