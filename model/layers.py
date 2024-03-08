import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Optional
from torch import Tensor
import fvcore.nn.weight_init as weight_init
from model.utils import PositionEmbeddingSine1D
from model.utils import PositionEmbeddingSine2D
from model.transformer import Transformer

def get_norm(norm, out_channels): 
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class QueryDecoder(nn.Module):
    def __init__(self, d_model=512, num_enc=0, num_dec=6, in_visual_dim=1024,in_text_dim=1024, return_intermediate_dec=False, num_queries=1):
        super().__init__()
        self.query_transformer = Transformer(d_model=d_model,
                                            nhead=8,
                                            num_encoder_layers=num_enc,
                                            num_decoder_layers=num_dec,
                                            dim_feedforward=2048,
                                            dropout=0.1,
                                            activation="relu",
                                            normalize_before=False,
                                            return_intermediate_dec=return_intermediate_dec
                                            )
        self.visual_proj = conv_layer(in_dim=in_visual_dim, out_dim=d_model, kernel_size=3, padding=1)
        self.text_proj = linear_layer(in_dim=in_text_dim, out_dim=d_model)
        self.visual_pos = PositionEmbeddingSine2D(d_model//2, normalize=True)

    def forward(self, visual, text):
        visual = self.visual_proj(visual)  
        text = self.text_proj(text) 
        visual_pos = self.visual_pos(visual)
        text_query = text.unsqueeze(1)
        hs, _ = self.query_transformer(src=visual, mask=None, query_embed=text_query, pos_embed=visual_pos)
        return hs   



class CrossModalFPNDecoder(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int, dim_feedforward: int = 2048, norm=None):
        """
        Args:
            feature_channels: list of fpn feature channel numbers.
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            dim_feedforward: number of vision-language fusion module ffn channel numbers.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.feature_channels = feature_channels

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            # in_channels: 4x -> 32x
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = conv_layer(in_channels, conv_dim, kernel_size=1)
            output_conv = conv_layer(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            # weight_init.c2_xavier_fill(lateral_conv)
            # weight_init.c2_xavier_fill(output_conv)
            stage = idx+1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = conv_layer(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # weight_init.c2_xavier_fill(self.mask_features)

        # vision-language cross-modal fusion
        self.text_pos = PositionEmbeddingSine1D(conv_dim, normalize=True)
        self.visual_pos = PositionEmbeddingSine2D(conv_dim//2, normalize=True)
        sr_ratios = [8, 4, 2, 1]        
        cross_attns = []
        for idx in range(len(feature_channels)): # res2 -> res5
            cross_attn = VisionLanguageBlock(conv_dim, dim_feedforward=dim_feedforward,
                                             nhead=8, sr_ratio=sr_ratios[idx])
            for p in cross_attn.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            stage = int(idx + 1)
            self.add_module("cross_attn_{}".format(stage), cross_attn)
            cross_attns.append(cross_attn)
        # place cross-attn in top-down order (from low to high resolution)
        self.cross_attns = cross_attns[::-1]


    def forward_features(self, features, text_features, text_masks):
        text_pos = self.text_pos(text_features).permute(2, 0, 1)   # [L,B,C]    
        text_features = text_features.permute(1, 0, 2)   # L, B, C

        for idx, f in enumerate(features[::-1][:-1]): # 32x -> 8x
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cross_attn = self.cross_attns[idx]

            b, c, h, w = f.shape
            pos = self.visual_pos(f)

            # NOTE: here the (h, w) is the size for current fpn layer
            vision_features = lateral_conv(f)  # [b, c, h, w]
            vision_features = rearrange(vision_features, 'b c h w -> (h w) b c', b=b)
            vision_pos = rearrange(pos, 'b c h w -> (h w) b c', b=b)

            cur_fpn = cross_attn(tgt=vision_features,
                                 memory=text_features,
                                 h=h, w=w,
                                 memory_key_padding_mask=text_masks,  
                                 pos=text_pos,
                                 query_pos=vision_pos
            ) # [h*w, b, c]
            cur_fpn = rearrange(cur_fpn, '(h w) b c -> b c h w', h=h, w=w)

            # upsample
            if idx == 0: # top layer
                y = output_conv(cur_fpn)
            else:
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="bilinear")
                y = output_conv(y)
        
        # 4x level
        lateral_conv = self.lateral_convs[-1]
        output_conv = self.output_convs[-1]
        cross_attn = self.cross_attns[-1]
        
        x = features[0]
        pos = self.visual_pos(x)
        b, c, h, w = x.shape

        vision_features = lateral_conv(x)  # [b, c, h, w]
        vision_features = rearrange(vision_features, 'b c h w -> (h w) b c', b=b)
        vision_pos = rearrange(pos, 'b c h w -> (h w) b c', b=b)


        cur_fpn = cross_attn(tgt=vision_features,
                             memory=text_features,
                             h=h, w=w,
                            #  tgt_key_padding_mask=vision_masks,
                             memory_key_padding_mask=text_masks,
                             pos=text_pos,
                             query_pos=vision_pos
        ) # [h*w, b, c]
        cur_fpn = rearrange(cur_fpn, '(h w) b c -> b c h w', h=h, w=w)
        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="bilinear")
        y = output_conv(y)
        return y   # [b, c, h, w], the spatial stride is 4x

    def forward(self, features, text_features, text_masks):
        """The forward function receives the vision and language features, 
            and outputs the mask features with the spatial stride of 4x.

        Args:
            features (list[Tensor]): backbone features (vision), length is number of FPN layers
                tensors: [b*t, ci, hi, wi]
            text_features (Tensor): text features (language)
                tensors: [b, length, c]
            text_masks (Tensor): text masks (language)
                tensors: [b, length, c]
            NOTE: the layer orders of both features and pos are res2 -> res5

        Returns:
            mask_features (Tensor): [b*t, mask_dim, h, w], with the spatial stride of 4x.
        """
        y = self.forward_features(features, text_features, text_masks)
        return self.mask_features(y)


class VisionLanguageBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sr_ratio=1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for downsample
        self.sr_ratio = sr_ratio

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, h, w,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        b = tgt.size(1) # hw, b, c 
        visual_l, b, _ = tgt.shape
        text_l, _, _ = memory.shape
        # self attn
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.sr_ratio > 1: # downsample
            q = rearrange(q, '(h w) b c -> b c h w', h=h, w=w)
            k = rearrange(k, '(h w) b c -> b c h w', h=h, w=w)
            v = rearrange(tgt, '(h w) b c -> b c h w', h=h, w=w)
            # downsample
            new_h = int(h * 1./self.sr_ratio)
            new_w = int(w * 1./self.sr_ratio)
            size = (new_h, new_w)
            q = F.interpolate(q, size=size, mode='bilinear')
            k = F.interpolate(k, size=size, mode='bilinear')
            v = F.interpolate(v, size=size, mode='bilinear')
            # shape for transformer
            q = rearrange(q, 'b c h w -> (h w) b c')
            k = rearrange(k, 'b c h w -> (h w) b c')
            v = rearrange(v, 'b c h w -> (h w) b c')
        else:
            v = tgt

        visual_length = q.shape[0]

        q = torch.cat([q, memory], dim=0)   # hw+l, b, c
        k = torch.cat([k, memory], dim=0)   
        v = torch.cat([v, memory], dim=0)
        
        visual_padding_mask = torch.zeros((b, visual_length), device=memory_key_padding_mask.device).bool()
        text_padding_mask = memory_key_padding_mask 

        padding_mask = torch.cat([visual_padding_mask, text_padding_mask], dim=1)   # b, hw+l
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None, key_padding_mask=padding_mask)[0][:visual_length] # [H*W+L, B, C] -> [H*W, B, C]


        if self.sr_ratio > 1:
            tgt2 = rearrange(tgt2, '(h w) b c -> b c h w', h=new_h, w=new_w)
            size = (h, w)  # recover to origin size
            tgt2 = F.interpolate(tgt2, size=size, mode='bilinear', align_corners=False) # [B, C, H, W]
            tgt2 = rearrange(tgt2, 'b c h w -> (h w) b c')
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attn
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class FusionModule(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        return tgt
