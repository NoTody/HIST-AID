
import os
import datetime

import torch
import torch.nn.functional as F

from contextlib import contextmanager
from sklearn.metrics import roc_auc_score

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, args.dist_url, gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=18000))
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    
def remove_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad.data).any()
            if has_nan:
                eps = 1e-6
                # Use torch.where to replace NaNs with eps
                param.grad.data = torch.where(torch.isnan(param.grad.data), 
                                            torch.full_like(param.grad.data, eps),
                                            param.grad.data)
                
def compute_AUCs(gt, pred, N_CLASSES):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            pass
    return AUROCs


def padding(sequences, max_length, pad_embedding=False):
    # Truncate and pad the sequences
    padded_sequences = []
    for seq in sequences:
        seq_length = seq.shape[0]

        # Truncate the sequence if it's longer than max_length
        if seq_length > max_length:
            seq = seq[-max_length:]
        # Pad the sequence if it's shorter than max_length
        elif seq_length < max_length:                              
            eps = 0
            padding_size = max_length - seq_length
            # The (0, 0, 0, padding_size) means pad zero at the end (bottom) of the 2nd dimension
            if pad_embedding:
                seq = F.pad(seq, (0, 0, 0, padding_size), "constant", eps) 
            else:
                seq = F.pad(seq, (0, padding_size), "constant", eps)

        padded_sequences.append(seq)
        
    padded_sequences = torch.stack(padded_sequences)
    return padded_sequences

def get_embed(text_encodings, model, pool):
    outputs = model(text_encodings)
    text_hidden_states = outputs.last_hidden_state
    if pool == 'cls':
        text_embed = text_hidden_states[:, 0, :]
    elif pool == 'full':
        text_embed = text_hidden_states
    elif pool == 'global_pool':
        text_embed = (text_hidden_states * text_encodings['attention_mask'].unsqueeze(-1))[:, 1:, :]
        text_embed = torch.mean(text_embed, dim=1)
    else:
        raise NotImplementedError("Wrong pool input!")

    return text_embed

def set_requires_grad_false(*models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = False

@contextmanager
def dummy_context():
    yield