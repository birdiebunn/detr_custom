from lib2to3.pgen2.token import OP
from optparse import Option
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import copy

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_head=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feed_forward=2048, dropout=0.1,
                activation='relu', normalize_before=False,
                return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, num_head, dim_feed_forward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, num_head, dim_feed_forward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self.reset_parameters()
        self.d_model = d_model
        self.num_head = num_head

    def reset_parameters(self):
        for p in self.patameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        target = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(target, memory, memory_key_padding_mask=mask,
                        pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1,2), memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask, pos)
        
        if self.norm is not None:
            output =  self.norm(output)
        
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, dim_feed_forward=2048, dropout=0.1,
                activation='relu', norm_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        # Initialization layers for the feed-forward model:
        self.lin1 = nn.Linear(d_model, dim_feed_forward)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_feed_forward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.norm_before = norm_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k,
                            value=src2,
                            attn_mask=src_mask,
                            src_key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        
        src2 = self.lin1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.lin2(src2)

        src = src + self.dropout2(src2)
        return src                    

    def forward_post(self, src, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor]  = None,
                    pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, 
                            value=src, 
                            attn_mask=src_mask,
                            src_key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.lin1(src)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.lin2(src2)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.norm_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = target
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, target_mask=target_mask,
                        memory_mask=memory_mask,
                        target_key_padding_mask=target_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_head, dim_feed_forward=2048, dropout=0.1,
                activation='relu', norm_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_head, dropout)

        # Initialization of the layers for feedforward model
        self.lin1 = nn.Linear(d_model, dim_feed_forward)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_feed_forward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.norm_before = norm_before
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_pre(self, target, memory,
                    target_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    target_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        target2 = self.norm1(target)
        q = k = self.with_pos_embed(target2, query_pos)
        target2 = self.self_attn(q,k, value=target2, attn_mask=target_mask,
                                key_padding_mask=target)
        target = target + self.dropout(target2)
        target2 = self.norm2(target)
        target2 = self.multihead_attn(
                        query=self.with_pos_embed(target2, query_pos),
                        key=self.with_pos_embed(memory, pos),
                        value=memory,
                        attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask
                    )[0]
        target = target + self.dropout2(target2)
        
        target2 = self.norm3(target)
        target2 = self.lin1(target2)
        target2 = self.activation(target2)
        target2 = self.dropout(target2)
        target2 = self.lin2(target2)
        target = target + self.dropout3(target2)

        return target

    def forward_post(self, target, memory,
                    target_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    target_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(target, query_pos)
        target2 = self.self_attn(q, k, 
                                value=target,
                                attn_mask=target_mask,
                                key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.self_attn(q, k,
                                value=target,
                                attn_mask=target_mask,
                                key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        target = target + self.dropout2(target2)
        target = self.norm2(target)
        target2 = self.lin1(target)
        target2 = self.activation(target2)
        target2 = self.dropout(target2)
        target2 = self.lin2(target2)
        target = target + self.dropout3(target2)
        target = self.norm3(target)
        return target

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.norm_before:
            return self.forward_pre(target, memory, target_mask, memory_mask,
                                    target_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(target, memory, target_mask, memory_mask,
                                target_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        num_head=args.num_head,
        dim_feed_forward=args.dim_feed_forward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def get_activation_fn(activation):
    """
        Return an acviation function given a string
    """
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    return RuntimeError(F'activation function should be relu/gelu/glu, NOT {activation}.')

def get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])
