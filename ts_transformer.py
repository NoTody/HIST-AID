# Modified from https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py

from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from attention_module import SelfAttention, precompute_freqs_cis

from misc_utils import generate_attention_mask, normalize


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0, **kwargs):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, input_sequence, time_padded=None):
        """
        Apply positional encoding to the input sequence.

        Args:
            input_sequence (torch.Tensor): The input sequence tensor.
                Shape: [sequence length, batch size, embed dim]
            time_padded (torch.Tensor, optional): The time padding tensor.
                Shape: [sequence length, batch size, embed dim]

        Returns:
            torch.Tensor: The input sequence tensor with positional encoding applied.
                Shape: [sequence length, batch size, embed dim]
        """
        encoded_sequence = input_sequence + self.pe[:, :input_sequence.size(1), :]
        return self.dropout(encoded_sequence)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe_time = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.pe_time, std=.02)

    def forward(self, input_sequence, time_padded=None):
        """
        Apply positional encoding to the input sequence.

        Args:
            input_sequence (torch.Tensor): The input sequence tensor.
                Shape: [sequence length, batch size, embed dim]
            time_padded (torch.Tensor, optional): The time padding tensor.
                Shape: [sequence length, batch size, embed dim]

        Returns:
            torch.Tensor: The input sequence tensor with positional encoding applied.
                Shape: [sequence length, batch size, embed dim]
        """
        if time_padded is not None:
            positional_encoding = self.pe_time.repeat(1, input_sequence.size(1), 1)
            positional_encoding = F.gelu(time_padded * positional_encoding)
            input_sequence += positional_encoding
        
        return self.dropout(input_sequence)


class MixedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0, **kwargs):
        super(MixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        
        self.pe_time = nn.Parameter(torch.empty(1, 1, d_model))  # requires_grad automatically set to True
        nn.init.trunc_normal_(self.pe_time, std=.02)

    def forward(self, input_sequence, time_padding=None):
        """
        Apply positional encoding to the input sequence.

        Args:
            input_sequence (torch.Tensor): The input sequence tensor.
                Shape: [sequence length, batch size, embed dim]
            time_padding (torch.Tensor, optional): The time padding tensor.
                Shape: [sequence length, batch size, embed dim]

        Returns:
            torch.Tensor: The input sequence tensor with positional encoding applied.
                Shape: [sequence length, batch size, embed dim]
        """
        if time_padding is not None:
            encoded_time = self.pe_time.repeat(1, input_sequence.size(1), 1)
            encoded_time = F.gelu(time_padding * encoded_time)
            input_sequence += self.pe[:input_sequence.size(0), :] + encoded_time
        
        return self.dropout(input_sequence)


class NoPositionalEncoding(nn.Module):

    def __init__(self, **kwargs):
        super(NoPositionalEncoding, self).__init__()

    def forward(self, x, time_padded=None):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        return x
    

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    elif pos_encoding == "mixed":
        return MixedPositionalEncoding
    elif pos_encoding == 'none' or pos_encoding == 'rope':
        return NoPositionalEncoding
    else:
        raise NotImplementedError("pos_encoding should be 'learnable'/'fixed'/'mixed'/'rope'/'none', \
            not '{}'".format(pos_encoding))
        
        
class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, \
        dropout=0.1, activation="relu", fused_attn=True):
        
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout, fused_attn=fused_attn)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout = Dropout(dropout)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)
        
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, freqs_cis: Tensor = None, is_causal=False,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src, attn = self.self_attn(src, attn_mask=src_mask, freqs_cis=freqs_cis)
        src = src + self.dropout1(src)  # (seq_len, batch_size, d_model)
        src = normalize(src, self.norm1)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src)  # (seq_len, batch_size, d_model)
        src = normalize(src, self.norm2)
        return src, attn
    

class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Transformer encoder for classification and regression
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0., pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False,
                 use_time=False, fused_attn=True):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.num_layers = num_layers
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.pos_encoding = pos_encoding

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model=d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, dim_feedforward, \
                dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, \
                dropout*(1.0 - freeze), activation=activation, fused_attn=fused_attn)

        self.transformer_encoder = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_encoder.append(encoder_layer)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.use_time = use_time
        
        # cls token initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks, time_padded, output_feature=False, cls_token=True):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        device = X.device
        if self.use_time and self.pos_encoding != 'rope':
            time_padded_pos = torch.cat((torch.zeros(X.shape[0], 1).cuda(), time_padded), dim=1).unsqueeze(dim=2)
            time_padded_pos = time_padded_pos.permute(1, 0, 2) # [seq_length, batch_size, feat_dim]
        else:
            time_padded_pos = None
        # concat cls token to input
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), dim=1)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp, time_padded_pos)  # add positional encoding
        inp = inp.permute(1, 0, 2)

        if self.pos_encoding == 'rope':
            cls_padded = torch.ones(time_padded.shape[0], 1).to(device)
            time_padded = torch.cat((cls_padded, time_padded), dim=1)
            t = time_padded * self.max_len
            freqs_cis = precompute_freqs_cis(dim=self.d_model // self.n_heads, t=t, theta=10000.0)
        else:
            freqs_cis = None
             
        attention_mask = generate_attention_mask(padding_masks, padding_masks, self.n_heads, device)

        for layer in self.transformer_encoder:
            out = layer(inp, attention_mask, freqs_cis=freqs_cis)
            
        output = self.act(out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)

        # Output
        if cls_token:
            output = output[:, 0, :]  # only use the first output vector (CLS token)
        else:
            output = output[:, 1:, :]
        
        if not output_feature:
            return self.output_layer(output), None
        else:
            return self.output_layer(output), output


class EarlyTSTransformerEncoderClassiregressor(nn.Module):
    """
    Transformer encoder for classification and regression with early fusion
    """

    def __init__(self, feat_dim, max_len_img, max_len_text, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False,
                 use_time=False, fused_attn=False):
        super(EarlyTSTransformerEncoderClassiregressor, self).__init__()

        self.max_len_img = max_len_img
        self.max_len_text = max_len_text
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.pos_encoding = pos_encoding
        self.fused_attn = fused_attn

        # proj different modality to same space with same dimension
        self.project_inp_img = ProjectionHead(feat_dim, d_model, d_model, dropout=0.1)
        self.project_inp_text = ProjectionHead(feat_dim, d_model, d_model, dropout=0.1)
        
        self.pos_enc_img = get_pos_encoder(pos_encoding)(d_model=d_model, dropout=dropout*(1.0 - freeze), max_len=max_len_img)
        self.pos_enc_text = get_pos_encoder(pos_encoding)(d_model=d_model, dropout=dropout*(1.0 - freeze), max_len=max_len_text)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, dim_feedforward, \
                dropout*(1.0 - freeze), activation=activation, fused_attn=fused_attn)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, \
                dropout*(1.0 - freeze), activation=activation, fused_attn=fused_attn)

        self.transformer_encoder = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_encoder.append(encoder_layer)

        self.act = _get_activation_fn(activation)

        self.dropout = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.use_time = use_time
        
        # register token initialization
        self.register_token_len = 4
        
        # cls token initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.img_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.text_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_token = nn.Parameter(torch.zeros(self.register_token_len, 1, d_model))
        self.learnable_tokens = [self.cls_token, self.img_token, self.text_token, self.register_token]
        
        self.output_layer = self.build_output_module(d_model, 1 + max_len_img + max_len_text, num_classes)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialization
        # trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        for token in self.learnable_tokens:
            nn.init.trunc_normal_(token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X_img, X_text, padding_masks_img, time_padded_img, padding_masks_text, \
        time_padded_text, output_feature=False, cls_token=True):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        device = X_img.device
        if self.use_time and self.pos_encoding != 'rope':
            # image
            time_padded_img_pos = time_padded_img.unsqueeze(dim=2)
            time_padded_img_pos = time_padded_img_pos.permute(1, 0, 2) # [seq_length, batch_size, feat_dim]
            # text
            time_padded_text_pos = time_padded_text.unsqueeze(dim=2)
            time_padded_text_pos = time_padded_text_pos.permute(1, 0, 2) # [seq_length, batch_size, feat_dim]
        else:
            time_padded_img_pos = None
            time_padded_text_pos = None
        
        padding_masks = torch.cat((padding_masks_img, padding_masks_text), dim=1)
        
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp_img = X_img.permute(1, 0, 2)
        inp_img = self.project_inp_img(inp_img) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp_img = self.pos_enc_img(inp_img, time_padded_img_pos)  # add positional encoding
        
        inp_text = X_text.permute(1, 0, 2)
        inp_text = self.project_inp_text(inp_text) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp_text = self.pos_enc_text(inp_text, time_padded_text_pos)  # add positional encoding
        
        # add image and text tokens
        inp_img = inp_img + self.img_token.expand(inp_img.shape[0], inp_img.shape[1], -1)
        inp_text = inp_text + self.text_token.expand(inp_text.shape[0], inp_text.shape[1], -1)
        
        # concate image and text embeddings across sequence dimension
        inp = torch.cat((inp_img, inp_text), dim=0)
        
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # concat cls token to input
        inp = torch.cat((self.cls_token.expand(-1, inp.shape[1], -1), inp), dim=0)
        inp = inp.permute(1, 0, 2)
        
        if self.pos_encoding == 'rope':
            cls_padded = torch.ones(time_padded_img.shape[0], 1).to(device)
            time_padded_img = torch.cat((cls_padded, time_padded_img), dim=1)
            t = torch.cat((time_padded_img, time_padded_text), dim=1) * self.max_len_text
            freqs_cis = precompute_freqs_cis(dim=self.d_model // self.n_heads, t=t, theta=10000.0)
        else:
            freqs_cis = None
            
        attention_mask = generate_attention_mask(padding_masks, padding_masks, self.n_heads, device)
        
        for layer in self.transformer_encoder:
            out, attn = layer(inp, attention_mask, freqs_cis=freqs_cis)
            
        if self.fused_attn:
            cls_weight_scores = None
        else:
            cls_weight_scores = attn.mean(dim=1)[:, :, :]

        output = self.act(out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        
        # Output
        if cls_token:
            output = output[:, 0, :]  # only use the first output vector (CLS token)
        else:
            output = output[:, 1:, :] # remove CLS token
        
        if not output_feature:
            return self.output_layer(output), None, cls_weight_scores
        else:
            return self.output_layer(output), output, cls_weight_scores
        
        
class BottleneckTSTransformerEncoderClassiregressor(nn.Module):
    """
    Transformer encoder for classification and regression with bottleneck fusion
    """
    def __init__(self, feat_dim, max_len_img, max_len_text, bottleneck_len, d_model, \
            n_heads, num_layers, dim_feedforward, num_classes, dropout=0.1, pos_encoding='learnable', \
            activation='gelu', norm='BatchNorm', freeze=False, use_time=False):
        super(BottleneckTSTransformerEncoderClassiregressor, self).__init__()

        self.max_len_img = max_len_img
        self.max_len_text = max_len_text
        self.bottleneck_len = bottleneck_len

        self.num_layers = num_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.pos_encoding = pos_encoding

        # proj different modality to same space with same dimension
        self.project_inp_img = ProjectionHead(feat_dim, d_model, d_model, dropout=0.1)
        self.project_inp_text = ProjectionHead(feat_dim, d_model, d_model, dropout=0.1)
        
        self.pos_enc_img = get_pos_encoder(pos_encoding)(d_model=d_model, dropout=dropout*(1.0 - freeze), max_len=max_len_img)
        self.pos_enc_text = get_pos_encoder(pos_encoding)(d_model=d_model, dropout=dropout*(1.0 - freeze), max_len=max_len_text)
        
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, dim_feedforward, \
                dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, \
                dropout*(1.0 - freeze), activation=activation)

        self.img_blocks = torch.nn.ModuleList()
        self.text_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.img_blocks.append(encoder_layer)
            self.text_blocks.append(encoder_layer)
            
        self.act = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        self.use_time = use_time
        
        # cls token initialization
        self.cls_token_img = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, d_model))
        self.bottleneck_token = nn.Parameter(torch.zeros(1, bottleneck_len, d_model))
        
        self.learnable_tokens = [self.cls_token_img, self.cls_token_text, self.bottleneck_token]
        
        self.output_layer_img = self.build_output_module(d_model, num_classes)
        self.output_layer_text = self.build_output_module(d_model, num_classes)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialization
        # trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        for token in self.learnable_tokens:
            nn.init.trunc_normal_(token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def build_output_module(self, d_model, num_classes):
        output_layer = nn.Linear(d_model, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X_img, X_text, padding_masks_img, time_padded_img, padding_masks_text, \
        time_padded_text, output_feature=False, cls_token=True):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        device = X_img.device
        bottlenectk_mask = torch.ones((X_img.shape[0], self.bottleneck_len), dtype=torch.bool).cuda()
        if self.use_time and self.pos_encoding != 'rope':
            # image
            time_padded_img_pos = time_padded_img.unsqueeze(dim=2)
            time_padded_img_pos = time_padded_img_pos.permute(1, 0, 2) # [seq_length, batch_size, feat_dim]
            # text
            time_padded_text_pos = time_padded_text.unsqueeze(dim=2)
            time_padded_text_pos = time_padded_text_pos.permute(1, 0, 2) # [seq_length, batch_size, feat_dim]
        else:
            time_padded_img_pos = None
            time_padded_text_pos = None
        
        # add mask for bottleneck
        padding_masks_img_bn = torch.concat((padding_masks_img, bottlenectk_mask), dim=1)
        padding_masks_text_bn = torch.concat((padding_masks_text, bottlenectk_mask), dim=1)

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp_img = X_img.permute(1, 0, 2)
        inp_img = torch.cat((self.cls_token_img.expand(-1, inp_img.shape[1], -1), inp_img), dim=0)
        inp_img = self.project_inp_img(inp_img) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp_img = self.pos_enc_img(inp_img, time_padded_img_pos)  # add positional encoding
        inp_img = inp_img.permute(1, 0, 2)
        
        inp_text = X_text.permute(1, 0, 2)
        inp_text = torch.cat((self.cls_token_text.expand(-1, inp_text.shape[1], -1), inp_text), dim=0)
        inp_text = self.project_inp_text(inp_text) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp_text = self.pos_enc_text(inp_text, time_padded_text_pos)  # add positional encoding
        inp_text = inp_text.permute(1, 0, 2)
        
        if self.pos_encoding == 'rope':
            self.max_len_img += 1
            self.max_len_text += 1
            # image
            cls_padded = torch.ones(time_padded_img.shape[0], 1).to(device)
            time_padded_img = torch.cat((cls_padded, time_padded_img), dim=1)
            t = time_padded_img * self.max_len_img
            freqs_cis_img = precompute_freqs_cis(dim=self.d_model // self.n_heads, t=t, theta=10000.0)
            # text
            cls_padded = torch.ones(time_padded_text.shape[0], 1).to(device)
            time_padded_text = torch.cat((cls_padded, time_padded_text), dim=1)
            t = time_padded_text * self.max_len_text
            freqs_cis_text = precompute_freqs_cis(dim=self.d_model // self.n_heads, t=t, theta=10000.0)
            # bn image
            bn_padded = torch.zeros(time_padded_img.shape[0], self.bottleneck_len).to(device)
            time_padded_img_bn = torch.cat((time_padded_img, bn_padded), dim=1)
            t = time_padded_img_bn * (self.max_len_img + self.bottleneck_len)
            freqs_cis_img_bn = precompute_freqs_cis(dim=self.d_model // self.n_heads, t=t, theta=10000.0)
            # bn text
            bn_padded = torch.zeros(time_padded_img.shape[0], self.bottleneck_len).to(device)
            time_padded_text_bn = torch.cat((time_padded_text, bn_padded), dim=1)
            t = time_padded_text_bn * (self.max_len_text + self.bottleneck_len)
            freqs_cis_text_bn = precompute_freqs_cis(dim=self.d_model // self.n_heads, t=t, theta=10000.0)
        else:
            freqs_cis_img, freqs_cis_text, freqs_cis_img_bn, freqs_cis_text_bn = None, None, None, None
        
        attention_mask_img = generate_attention_mask(padding_masks_img, \
            padding_masks_img, self.n_heads, device)
        attention_mask_text = generate_attention_mask(padding_masks_text, \
            padding_masks_text, self.n_heads, device)
        attention_mask_img_bn = generate_attention_mask(padding_masks_img_bn, \
            padding_masks_img_bn, self.n_heads, device)
        attention_mask_text_bn = generate_attention_mask(padding_masks_text_bn, \
            padding_masks_text_bn, self.n_heads, device)
        
        # inference
        bottleneck_token = nn.Parameter(self.bottleneck_token.expand(inp_img.shape[0], \
            self.bottleneck_len, self.d_model))
        fuse_layers = [2, 3]
        for i in range(self.num_layers):
            if i in fuse_layers:
                inp_img_mod = torch.concat([inp_img, bottleneck_token], dim=1)
                inp_text_mod = torch.concat([inp_text, bottleneck_token], dim=1)
                
                inp_img_mod = self.img_blocks[i](inp_img_mod, attention_mask_img_bn, freqs_cis=freqs_cis_img_bn)
                inp_text_mod = self.text_blocks[i](inp_text_mod, attention_mask_text_bn, freqs_cis=freqs_cis_text_bn)
                
                inp_img = inp_img_mod[:, :self.max_len_img, :]
                inp_text = inp_text_mod[:, :self.max_len_text, :]
            else:
                inp_img = self.img_blocks[i](inp_img, attention_mask_img, freqs_cis=freqs_cis_img)
                inp_text = self.text_blocks[i](inp_text, attention_mask_text, freqs_cis=freqs_cis_text)
        
        # Output
        output_img = self.act(inp_img)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output_img = self.dropout(output_img)
        
        output_text = self.act(inp_text)
        output_text = self.dropout(output_text)
        
        if cls_token:
            output_img = output_img[:, 0, :]  # only use the first output vector (CLS token)
            output_text = output_text[:, 0, :]  # only use the first output vector (CLS token)
        else:
            output_img = output_img[:, 1:, :] # remove CLS token
            output_text = output_text[:, 1:, :] # remove CLS token
        
        if not output_feature:
            return (self.output_layer_img(output_img) + self.output_layer_text(output_text)) / 2, None
        else:
            return (self.output_layer_img(output_img) + self.output_layer_text(output_text)) / 2, [output_img, output_text]