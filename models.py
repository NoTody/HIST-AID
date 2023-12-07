import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from block import fusions

from ts_transformer import TSTransformerEncoderClassiregressor, EarlyTSTransformerEncoderClassiregressor, \
    BottleneckTSTransformerEncoderClassiregressor

from misc_utils import padding, get_embed, dummy_context, set_requires_grad_false


class vit_model(nn.Module):
    def __init__(self, args, features_dim, out_size, method='decoder', 
                 pretrained='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                 lock=False, use_time=False, pos_encoding='learnable'):
        
        super(vit_model, self).__init__()
        
        self.method = method
        self.img_time_series = args.img_time_series
        self.use_time = use_time
        self.lock = lock
        self.feature_dim_in = features_dim
        clip_model, _, _ = open_clip.create_model_and_transforms(pretrained)
        self.img_backbone = clip_model.visual.trunk
        
        if lock:
            set_requires_grad_false(self.img_backbone)
        
        if self.img_time_series:
            if method == 'decoder':
                feat_dim = 768
                max_len = 6
                d_model = 768
                n_heads = 16
                num_layers = 3
                dim_feedforward = 3074
                num_classes = 13
                
                self.img_decoder = TSTransformerEncoderClassiregressor(feat_dim, max_len, d_model, n_heads, \
                            num_layers, dim_feedforward, num_classes, use_time=use_time, pos_encoding=pos_encoding)

        if (not self.img_time_series) or method == 'average':
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim_in, out_size),
            )
        
        
    def forward(self, x_img=None, img_time=None, ensemble=False):
        out = {}
        if self.img_time_series:
            # convert text_time to tensor
            img_time = [torch.tensor(time).cuda() for time in img_time]
            # accumulate text embedding and take mean
            max_length = 5
            # padding masks initialized with cls token mask
            padding_masks = torch.Tensor([])

            if self.method == 'average':
                lengths =  [d["imgs"].shape[0] for d in x_img]
                x_img_stack = torch.cat([d["imgs"] for d in x_img]).cuda()
                img_embeddings = self.img_backbone(x_img_stack)
                img_embeddings = torch.split(img_embeddings, lengths)
                img_feature = torch.cat([torch.mean(embed, dim=0).view(1,-1) for embed in img_embeddings]).cuda()
                logits = self.classifier(img_feature)
            else:
                lengths =  [d["imgs"].shape[0] for d in x_img]
                x_img = torch.cat([d["imgs"] for d in x_img]).cuda()
                
                with torch.no_grad() if self.lock else dummy_context():
                    img_embeddings = self.img_backbone(x_img)

                # separate embeddings by time-series length
                img_embeddings = torch.split(img_embeddings, lengths)
                img_embeddings = [img_embedding for img_embedding in img_embeddings]

                # generate padding masks
                for img_embedding in img_embeddings:
                    if img_embedding != None:
                        true_length = img_embedding.shape[0]
                        indices = torch.arange(max_length)
                        mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                        if len(padding_masks) == 0:
                            padding_masks = mask.unsqueeze(dim=0)
                        else:
                            padding_masks = torch.cat((padding_masks, mask.unsqueeze(dim=0)), dim=0)
                
                # padding
                padding_masks = padding_masks.cuda()
                accum_embeddings_padded = padding(img_embeddings, max_length, pad_embedding=True)
                accum_embeddings_padded = F.normalize(accum_embeddings_padded, p=2, dim=-1)
                img_time_padded = padding(img_time, max_length, pad_embedding=False)
                # time-series forward
                logits, img_feature = self.img_decoder(accum_embeddings_padded, padding_masks, img_time_padded, output_feature=True)
        else:
            with torch.no_grad() if self.lock else dummy_context():
                img_feature = self.img_backbone(x_img)
            logits = self.classifier(img_feature)
            
        out = {}
        out['out'] = logits
        out['feature'] = img_feature

        return out
    
    
# bert model with fusion
class bert_model(nn.Module):
    def __init__(self, args, features_dim, out_size, method='decoder', pool='cls', lock=False,
                 pretrained='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                 use_time=False, pos_encoding='learnable'):
        super(bert_model, self).__init__()
        
        self.args = args
        clip_model, _, _ = open_clip.create_model_and_transforms(pretrained)
        self.text_backbone = clip_model.text.transformer
        
        self.pool = pool
        self.method = method
        self.lock = lock
        self.feature_dim_in = features_dim
        
        if args.text_time_series and self.method == 'decoder':
            feat_dim = 768
            max_len = 51
            d_model = 768
            n_heads = 16
            num_layers = 3
            dim_feedforward = 3074
            num_classes = 13
            
            self.text_decoder = TSTransformerEncoderClassiregressor(feat_dim, max_len, d_model, n_heads, \
                num_layers, dim_feedforward, num_classes, use_time=use_time, pos_encoding=pos_encoding)
        
        if (not args.text_time_series) or self.method == 'average':
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim_in, out_size),
            )

        # freeze models
        if lock:
            set_requires_grad_false(self.text_backbone)
        
    def forward(self, x_text, text_time, ensemble=False):
        if self.args.text_time_series:
            # convert text_time to tensor
            text_time = [torch.tensor(time).cuda() for time in text_time]
            # accumulate text embedding and take mean
            max_length = 50
            text_feature = None
            # padding masks initialized with cls token mask
            padding_masks = torch.Tensor([])
            # get time-series length of texts for embedding separation
            lengths =  [d["input_ids"].shape[0] for d in x_text]
            text_inp = torch.cat([d["input_ids"] for d in x_text]).cuda()

            # get embeddings
            with torch.no_grad() if self.lock else dummy_context():
                text_embeddings = get_embed(text_inp, self.text_backbone, self.pool)

            # separate embeddings by time-series length
            text_embeddings = torch.split(text_embeddings, lengths)

            if self.method == 'average':
                text_embedding = torch.cat([torch.mean(embed, dim=0).view(1,-1) for embed in text_embeddings]).cuda()
                logits = self.classifier(text_embedding)
            else:
                # generate padding masks
                for text_embedding in text_embeddings:
                    if text_embedding != None:
                        true_length = text_embedding.shape[0]
                        indices = torch.arange(max_length)
                        mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                        if len(padding_masks) == 0:
                            padding_masks = mask.unsqueeze(dim=0)
                        else:
                            padding_masks = torch.cat((padding_masks, mask.unsqueeze(dim=0)), dim=0)

                padding_masks = padding_masks.cuda()
                accum_embeddings_padded = padding(text_embeddings, max_length, pad_embedding=True)
                accum_embeddings_padded = F.normalize(accum_embeddings_padded, p=2, dim=-1)
                text_time_padded = padding(text_time, max_length, pad_embedding=False)
                # time-series forward
                logits, text_feature = self.text_decoder(accum_embeddings_padded, padding_masks, text_time_padded, output_feature=True)
        else:
            text_inp = torch.cat([d["input_ids"][-1].unsqueeze(dim=0) for d in x_text]).cuda()
            with torch.no_grad() if self.lock else dummy_context():
                text_feature = get_embed(text_inp, self.text_backbone, self.pool)
            logits = self.classifier(text_feature)
                
        out = {}
        out['out'] = logits
        out['feature'] = text_feature
        
        return out
    
    
class mm_model(nn.Module):
    def __init__(self, args, features_dim, out_size, method='average', fusion_method='MCB',
                 pool='cls', lock=False, use_time=False, pos_encoding='learnable', img_max_len=5, 
                 text_max_len=50, pretrained='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                 ):
        super(mm_model, self).__init__()

        self.img_time_series = args.img_time_series
        self.text_time_series = args.text_time_series
        self.img_max_len = img_max_len
        self.text_max_len = text_max_len
        self.method = method
        self.fusion_method = fusion_method
        self.lock = lock

        clip_model, _, _ = open_clip.create_model_and_transforms(pretrained)
        self.img_backbone = clip_model.visual.trunk
        self.text_backbone = clip_model.text.transformer
        
        self.relu = nn.ReLU()
        self.pool = pool
        self.img_backbone.fc = nn.Identity()
        
        # text decoder
        if self.text_time_series and self.method == 'decoder':
            # text decoder
            feat_dim = 768
            max_len = text_max_len+1
            d_model = 768
            n_heads = 16
            num_layers = args.decoder_layers
            dim_feedforward = d_model * 4
            num_classes = 13
            self.text_decoder = TSTransformerEncoderClassiregressor(feat_dim, max_len, d_model, n_heads, \
                num_layers, dim_feedforward, num_classes, use_time=use_time, pos_encoding=pos_encoding)
            
        if self.img_time_series and self.method == 'decoder':
            # image decoder
            feat_dim = 768
            max_len = img_max_len+1
            d_model = 768
            n_heads = 16
            num_layers = 3
            dim_feedforward = d_model * 4
            num_classes = 13
            self.img_decoder = TSTransformerEncoderClassiregressor(feat_dim, max_len, d_model, n_heads, \
                num_layers, dim_feedforward, num_classes, use_time=use_time, pos_encoding=pos_encoding)
                
        if fusion_method == 'Block':
            mm_dim = 1600
            self.fusion = fusions.Block([768, 768], output_dim=mm_dim, mm_dim=mm_dim)
        elif fusion_method == 'ConcatMLP':
            mm_dim = 1200
            dimensions = [mm_dim, mm_dim]
            self.fusion = fusions.ConcatMLP([768, 768], output_dim=mm_dim, dimensions=dimensions)
        else:
            raise NotImplementedError(f"{fusion_method} not implemented")
        
        if lock:
            set_requires_grad_false(self.img_backbone, self.text_backbone)

        if self.method == 'average':
            d_model = 768
            self.classifier = nn.Sequential(
                nn.Linear(d_model, out_size),
            )
        elif self.img_time_series and self.text_time_series:
            self.classifier = nn.Sequential(
                nn.Linear(mm_dim, out_size)
            )


    def forward(self, x_img, x_text, img_time=None, text_time=None):
        if self.text_time_series:
            # convert text_time to tensor
            text_time = [torch.tensor(time).cuda() for time in text_time]
            # accumulate text embedding and take mean
            max_length = self.text_max_len
            accum_embeddings = []
            # padding masks initialized with cls token mask
            text_padding_masks = torch.Tensor([])
            # concatenate texts
            # get time-series length of texts for embedding separation
            lengths =  [d["input_ids"].shape[0] if d["input_ids"].shape[0] <= max_length \
                else max_length for d in x_text]
            text_inp = torch.cat([d["input_ids"] if d["input_ids"].shape[0] <= max_length \
                else d["input_ids"][-max_length:] for d in x_text]).cuda()
            
            # get embeddings
            with torch.no_grad() if self.lock else dummy_context():
                text_embeddings = get_embed(text_inp, self.text_backbone, self.pool)
                
            # separate embeddings by time-series length
            text_embeddings = torch.split(text_embeddings, lengths)
            
            if self.method == 'average':
                text_feature = torch.cat([torch.mean(embed, dim=0).view(1,-1) for embed in text_embeddings]).cuda()
            else:
                # calculate padding embeddings
                for text_embedding in text_embeddings:
                    if text_embedding != None:
                        true_length = min(text_embedding.shape[0], max_length)
                        indices = torch.arange(max_length)
                        mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                        if len(text_padding_masks) == 0:
                            text_padding_masks = mask.unsqueeze(dim=0)
                        else:
                            text_padding_masks = torch.cat((text_padding_masks, mask.unsqueeze(dim=0)), dim=0)
                        accum_embeddings.append(text_embedding)
                        
                text_padding_masks = text_padding_masks.cuda()
                accum_embeddings_padded = padding(accum_embeddings, max_length, pad_embedding=True)
                accum_embeddings_padded = F.normalize(accum_embeddings_padded, p=2, dim=-1)
                text_time_padded = padding(text_time, max_length, pad_embedding=False)

                # text embedding
                if self.method == 'decoder':
                    _, text_feature  = self.text_decoder(accum_embeddings_padded, text_padding_masks, \
                        text_time_padded, output_feature=True, cls_token=True)
        else:
            text_inp = torch.cat([d["input_ids"][-1].unsqueeze(dim=0) for d in x_text]).cuda()
            with torch.no_grad() if self.lock else dummy_context():
                text_feature = get_embed(text_inp, self.text_backbone, self.pool)
                    
        # image embedding
        if self.img_time_series:
            # convert text_time to tensor
            img_time = [torch.tensor(time).cuda() for time in img_time]
            # accumulate text embedding and take mean
            max_length = self.img_max_len
            accum_embeddings = []
            # padding masks initialized with cls token mask
            img_padding_masks = torch.Tensor([])
            # concatenate texts
            # get time-series length of texts for embedding separation
            lengths =  [d["imgs"].shape[0] if d["imgs"].shape[0] <= max_length \
                else max_length for d in x_img]
            
            x_img = torch.cat([d["imgs"] if d["imgs"].shape[0] <= max_length \
                else d["imgs"][-max_length:] for d in x_img]).cuda()
            
            with torch.no_grad() if self.lock else dummy_context():
                img_embeddings = self.img_backbone(x_img)
            
            # separate embeddings by time-series length
            img_embeddings = torch.split(img_embeddings, lengths)
            img_embeddings = [img_embedding for img_embedding in img_embeddings]
            
            if self.method == 'average':
                img_feature = torch.cat([torch.mean(embed, dim=0).view(1,-1) for embed in img_embeddings]).cuda()
            else:
                # generate padding masks
                for img_embedding in img_embeddings:
                    if img_embedding != None:
                        true_length = min(img_embedding.shape[0], max_length)
                        indices = torch.arange(max_length)
                        mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                        if len(img_padding_masks) == 0:
                            img_padding_masks = mask.unsqueeze(dim=0)
                        else:
                            img_padding_masks = torch.cat((img_padding_masks, mask.unsqueeze(dim=0)), dim=0)
                        accum_embeddings.append(img_embedding)
                # padding
                img_padding_masks = img_padding_masks.cuda()
                accum_embeddings_padded = padding(accum_embeddings, max_length, pad_embedding=True)
                accum_embeddings_padded = F.normalize(accum_embeddings_padded, p=2, dim=-1)
                img_time_padded = padding(img_time, max_length, pad_embedding=False)
                logits, img_feature = self.img_decoder(accum_embeddings_padded, img_padding_masks, \
                        img_time_padded, output_feature=True, cls_token=True)
        else:
            with torch.no_grad() if self.lock else dummy_context():
                img_feature = self.img_backbone(x_img)

        if self.text_time_series and (not self.img_time_series):
            x_img, x_text = torch.nn.functional.normalize(img_feature, p=2, dim=-1), \
                torch.nn.functional.normalize(text_feature, p=2, dim=-1)
            x_fuse = self.fusion([x_img, x_text])
            logits = self.classifier(x_fuse)
        elif self.img_time_series and self.text_time_series:
            x_img, x_text = torch.nn.functional.normalize(img_feature, p=2, dim=-1), \
                torch.nn.functional.normalize(text_feature, p=2, dim=-1)
            x_fuse = self.fusion([x_img, x_text])
            logits = self.classifier(x_fuse)
        elif not self.img_time_series or not self.text_time_series:
            x_img, x_text = torch.nn.functional.normalize(img_feature, p=2, dim=-1), \
                torch.nn.functional.normalize(text_feature, p=2, dim=-1)
            x_fuse = self.fusion([x_img, x_text])
            logits = self.classifier(x_fuse)
        else:
            raise NotImplementedError("Wrong img/text_time_series input!")

        out = {}
        out['out'] = logits

        return out
    
    
class mm_model_early(nn.Module):
    def __init__(self, args, features_dim, out_size, pool='cls', img_max_len=5, text_max_len=50,
                 lock=False, pretrained='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                 use_time=False, pos_encoding='learnable'):
        super(mm_model_early, self).__init__()
        
        self.img_max_len = img_max_len
        self.text_max_len = text_max_len
        
        self.img_time_series = args.img_time_series
        self.text_time_series = args.text_time_series
        self.lock = lock

        clip_model, _, _ = open_clip.create_model_and_transforms(pretrained)
        self.img_backbone = clip_model.visual.trunk
        self.text_backbone = clip_model.text.transformer

        self.relu = nn.ReLU()
        self.pool = pool
        self.img_backbone.fc = nn.Identity()
        
        # text decoder
        feat_dim = 768
        d_model = 768
        n_heads = 16
        dim_feedforward = d_model * 4
        max_len_img = args.img_max_len
        max_len_text = args.text_max_len
        num_layers = args.decoder_layers
        num_classes = out_size
        
        self.decoder = EarlyTSTransformerEncoderClassiregressor(feat_dim, max_len_img, max_len_text, d_model, n_heads, \
            num_layers, dim_feedforward, num_classes, use_time=use_time, pos_encoding=pos_encoding)
        
        hidden_dim = feat_dim
        
        if lock:
            set_requires_grad_false(self.img_backbone, self.text_backbone)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, out_size)
        )

    def forward(self, x_img, x_text, img_time=None, text_time=None):
        if self.text_time_series:
            # convert text_time to tensor
            text_time = [torch.tensor(time).cuda() for time in text_time]
            # accumulate text embedding and take mean
            max_length = self.text_max_len
            accum_embeddings = []
            # padding masks initialized with cls token mask
            padding_masks_text = torch.Tensor([])
            # concatenate texts
            # get time-series length of texts for embedding separation
            lengths =  [d["input_ids"].shape[0] if d["input_ids"].shape[0] <= max_length \
                else max_length for d in x_text]
            text_inp = torch.cat([d["input_ids"] if d["input_ids"].shape[0] <= max_length \
                else d["input_ids"][-max_length:] for d in x_text]).cuda()
            
            # get embeddings
            with torch.no_grad() if self.lock else dummy_context():
                text_embeddings = get_embed(text_inp, self.text_backbone, self.pool)
                
            # separate embeddings by time-series length
            text_embeddings = torch.split(text_embeddings, lengths)
            
            # calculate padding embeddings
            for text_embedding in text_embeddings:
                if text_embedding != None:
                    true_length = min(text_embedding.shape[0], max_length)                                
                    indices = torch.arange(max_length)
                    mask = (indices < true_length)
                    if len(padding_masks_text) == 0:
                        padding_masks_text = mask.unsqueeze(dim=0)
                    else:
                        padding_masks_text = torch.cat((padding_masks_text, mask.unsqueeze(dim=0)), dim=0)
                        
                    accum_embeddings.append(text_embedding)
                    
            padding_masks_text = padding_masks_text.cuda()
            accum_embeddings_padded_text = padding(accum_embeddings, max_length, pad_embedding=True)
            accum_embeddings_padded_text = F.normalize(accum_embeddings_padded_text, p=2, dim=-1)
            time_padded_text = padding(text_time, max_length, pad_embedding=False)
        else:
            text_inp = torch.cat([d["input_ids"][-1].unsqueeze(dim=0) for d in x_text]).cuda()
            
            with torch.no_grad() if self.lock else dummy_context():
                accum_embeddings_padded_text = get_embed(text_inp, self.text_backbone, self.pool).unsqueeze(dim=1)
                
            padding_masks_text = torch.tensor([True], dtype=torch.bool).unsqueeze(dim=0).repeat(bs, 1).cuda()
            time_padded_text = torch.tensor([1.0000], dtype=torch.float).unsqueeze(dim=0).repeat(bs, 1).cuda()

        # image embedding
        if self.img_time_series:
            # convert text_time to tensor
            img_time = [torch.tensor(time).cuda() for time in img_time]
            # accumulate text embedding and take mean
            max_length = self.img_max_len
            accum_embeddings = []
            # padding masks initialized with cls token mask
            padding_masks_img = torch.Tensor([])
            # concatenate texts
            # get time-series length of texts for embedding separation
            lengths =  [d["imgs"].shape[0] if d["imgs"].shape[0] <= max_length \
                else max_length for d in x_img]
            
            x_img = torch.cat([d["imgs"] if d["imgs"].shape[0] <= max_length \
                else d["imgs"][-max_length:] for d in x_img]).cuda()
            
            with torch.no_grad() if self.lock else dummy_context():
                img_embeddings = self.img_backbone(x_img)
            
            # separate embeddings by time-series length
            img_embeddings = torch.split(img_embeddings, lengths)
            
            # generate padding masks
            for img_embedding in img_embeddings:
                if img_embedding != None:
                    true_length = min(img_embedding.shape[0], max_length)
                    indices = torch.arange(max_length)
                    mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                    if len(padding_masks_img) == 0:
                        padding_masks_img = mask.unsqueeze(dim=0)
                    else:
                        padding_masks_img = torch.cat((padding_masks_img, mask.unsqueeze(dim=0)), dim=0)
                    accum_embeddings.append(img_embedding)
            # padding
            padding_masks_img = padding_masks_img.cuda()
            accum_embeddings_padded_img = padding(accum_embeddings, max_length, pad_embedding=True)
            accum_embeddings_padded_img = F.normalize(accum_embeddings_padded_img, p=2, dim=-1)
            time_padded_img = padding(img_time, max_length, pad_embedding=False)
        else:
            bs = x_img.shape[0]
            accum_embeddings_padded_img = self.img_backbone(x_img).unsqueeze(dim=1).cuda()
            padding_masks_img = torch.tensor([True, True], dtype=torch.bool).unsqueeze(dim=0).repeat(bs, 1).cuda()
            time_padded_img = torch.tensor([1.0000], dtype=torch.float).unsqueeze(dim=0).repeat(bs, 1).cuda()
        
        # time-series forward
        _, feature = self.decoder(accum_embeddings_padded_img, accum_embeddings_padded_text, padding_masks_img, \
            time_padded_img, padding_masks_text, time_padded_text, output_feature=True, cls_token=True)
        
        logits = self.classifier(feature)

        out = {}
        out['out'] = logits

        return out
    
    
class mm_model_intermediate(nn.Module):
    def __init__(self, args, features_dim, out_size, pool='cls', img_max_len=5, text_max_len=50,
                 lock=False, pretrained='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                 use_time=False, pos_encoding='learnable'):
        super(mm_model_intermediate, self).__init__()
        
        self.img_max_len = img_max_len
        self.text_max_len = text_max_len

        self.img_time_series = args.img_time_series
        self.text_time_series = args.text_time_series
        self.lock = lock

        clip_model, _, _ = open_clip.create_model_and_transforms(pretrained)
        self.img_backbone = clip_model.visual.trunk
        self.text_backbone = clip_model.text.transformer

        self.relu = nn.ReLU()
        self.pool = pool
        self.img_backbone.fc = nn.Identity()
        
        # decoder
        feat_dim = 768
        d_model = 768
        n_heads = 16
        bottleneck_len = 4
        dim_feedforward = d_model * 4
        num_layers = args.decoder_layers
        num_classes = out_size
        
        self.decoder = BottleneckTSTransformerEncoderClassiregressor(feat_dim, img_max_len, text_max_len, bottleneck_len, \
            d_model, n_heads, num_layers, dim_feedforward, num_classes, use_time=use_time, pos_encoding=pos_encoding)

        if lock:
            set_requires_grad_false(self.img_backbone, self.text_backbone)

        # self.classifier = nn.Sequential(
        #     nn.Linear(feat_dim, out_size)
        # )

    def forward(self, x_img, x_text, img_time=None, text_time=None):
        if self.text_time_series:
            # convert text_time to tensor
            text_time = [torch.tensor(time).cuda() for time in text_time]
            # accumulate text embedding and take mean
            max_length = self.text_max_len
            accum_embeddings = []
            # padding masks initialized with cls token mask
            padding_masks_text = torch.Tensor([])
            # concatenate texts
            # get time-series length of texts for embedding separation
            lengths =  [d["input_ids"].shape[0] for d in x_text]
            text_inp = torch.cat([d["input_ids"] for d in x_text]).cuda()
            
            # get embeddings
            with torch.no_grad() if self.lock else dummy_context():
                text_embeddings = get_embed(text_inp, self.text_backbone, self.pool)
                
            # separate embeddings by time-series length
            text_embeddings = torch.split(text_embeddings, lengths)
            
            # calculate padding embeddings
            for text_embedding in text_embeddings:
                if text_embedding != None:
                    true_length = text_embedding.shape[0]
                    indices = torch.arange(max_length)
                    mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                    
                    if len(padding_masks_text) == 0:
                        padding_masks_text = mask.unsqueeze(dim=0)
                    else:
                        padding_masks_text = torch.cat((padding_masks_text, mask.unsqueeze(dim=0)), dim=0)
                        
                    accum_embeddings.append(text_embedding)
                    
            padding_masks_text = padding_masks_text.cuda()
            accum_embeddings_padded_text = padding(accum_embeddings, max_length, pad_embedding=True)
            accum_embeddings_padded_text = F.normalize(accum_embeddings_padded_text, p=2, dim=-1)
            time_padded_text = padding(text_time, max_length, pad_embedding=False)

        # image embedding
        if self.img_time_series:
            # convert text_time to tensor
            img_time = [torch.tensor(time).cuda() for time in img_time]
            # accumulate text embedding and take mean
            max_length = self.img_max_len
            accum_embeddings = []
            # padding masks initialized with cls token mask
            padding_masks_img = torch.Tensor([])
            # concatenate texts
            # get time-series length of texts for embedding separation
            lengths =  [d["imgs"].shape[0] if d["imgs"].shape[0] <= max_length \
                else max_length for d in x_img]
            
            x_img = torch.cat([d["imgs"] if d["imgs"].shape[0] <= max_length \
                else d["imgs"][-max_length:] for d in x_img]).cuda()
            
            with torch.no_grad() if self.lock else dummy_context():
                img_embeddings = self.img_backbone(x_img)
            
            # separate embeddings by time-series length
            img_embeddings = torch.split(img_embeddings, lengths)
            
            # generate padding masks
            for img_embedding in img_embeddings:
                if img_embedding != None:
                    true_length = min(img_embedding.shape[0], max_length)
                    indices = torch.arange(max_length)
                    mask = torch.concat((torch.tensor([True], dtype=torch.bool), indices < true_length), dim=0)
                    if len(padding_masks_img) == 0:
                        padding_masks_img = mask.unsqueeze(dim=0)
                    else:
                        padding_masks_img = torch.cat((padding_masks_img, mask.unsqueeze(dim=0)), dim=0)
                    accum_embeddings.append(img_embedding)
            # padding
            padding_masks_img = padding_masks_img.cuda()
            accum_embeddings_padded_img = padding(accum_embeddings, max_length, pad_embedding=True)
            accum_embeddings_padded_img = F.normalize(accum_embeddings_padded_img, p=2, dim=-1)
            time_padded_img = padding(img_time, max_length, pad_embedding=False)
            # time-series forward
            logits, feature = self.decoder(accum_embeddings_padded_img, accum_embeddings_padded_text, padding_masks_img, \
            time_padded_img, padding_masks_text, time_padded_text, output_feature=False, cls_token=True)
        else:
            padding_masks_img = torch.Tensor([])
            bs = x_img.shape[0]
            accum_embeddings_padded_img = self.img_backbone(x_img).unsqueeze(dim=1).cuda()
            padding_masks_img = torch.tensor([True, True], dtype=torch.bool).unsqueeze(dim=0).repeat(bs, 1).cuda()
            time_padded_img = torch.tensor([1.0000], dtype=torch.float).unsqueeze(dim=0).repeat(bs, 1).cuda()
            
        print(f"accum_embeddings_padded_img: {accum_embeddings_padded_img.shape}, \
              accum_embeddings_padded_text: {accum_embeddings_padded_text.shape}, \
              padding_masks_img: {padding_masks_img.shape}, \
              time_padded_img: {time_padded_img.shape}, \
              padding_masks_text: {padding_masks_text.shape}, \
              time_padded_text: {time_padded_text.shape}")
        
        logits, feature = self.decoder(accum_embeddings_padded_img, accum_embeddings_padded_text, padding_masks_img, \
            time_padded_img, padding_masks_text, time_padded_text, output_feature=False, cls_token=True)

        out = {}
        out['out'] = logits

        return out