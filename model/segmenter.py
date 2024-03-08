import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.swin import build_model_swin

from einops import rearrange, repeat
from .layers import CrossModalFPNDecoder, QueryDecoder, FusionModule
from model.utils import PositionEmbeddingSine2D



class UniLSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        if cfg.swin_pretrain:
            self.backbone_visual = build_model_swin(cfg.swin_pretrain)


        # Cross-modal Fusion
        self.fusion = FusionModule(d_model=cfg.fusion_dim)
        self.fusion_proj = nn.Linear(512, cfg.fusion_dim)
        self.visual_fusion_pos = PositionEmbeddingSine2D(cfg.fusion_dim//2, normalize=True)   # half channel of vision feature

        self.all_fusion_proj = nn.ModuleList()
        self.all_fusion_module = nn.ModuleList()
        self.all_fusion_pos = nn.ModuleList()
        if cfg.swin_pretrain:
            for i in range(3):
                self.all_fusion_proj.append(nn.Linear(512, cfg.pixel_decoder_in[0]*2**i))
                self.all_fusion_module.append(FusionModule(d_model=cfg.pixel_decoder_in[0]*2**i))
                self.all_fusion_pos.append(PositionEmbeddingSine2D(cfg.pixel_decoder_in[0]*2**i//2, normalize=True))

        # Pixel Decoder
        self.pixel_decoder = CrossModalFPNDecoder(cfg.pixel_decoder_in, cfg.pixel_decoder_conv, cfg.pixel_decoder_mask, norm=None)

        # Query Decoder
        self.query_decoder = QueryDecoder(d_model=cfg.d_model, num_enc=cfg.num_enc, num_dec=cfg.num_dec, in_visual_dim=cfg.visual_in, in_text_dim=cfg.text_in, 
                                        return_intermediate_dec=cfg.aux_loss)



    def forward(self, img, word, mask=None, epoch=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool() # B, L

        # vis: 4✕ / 8✕ / 16✕ / 32✕
        # word: b, length, 512
        # state: b, 1024
        if self.cfg.swin_pretrain:
            vis = self.backbone_visual(img)
        else:
            vis = self.backbone.encode_image(img)
            vis = list(vis)
        word, state = self.backbone.encode_text(word)
        
        # Cross-modal Fusion
        b, c, h, w = vis[-1].shape
        visual_fusion_pos = self.visual_fusion_pos(vis[-1])
        visual_fusion_pos = rearrange(visual_fusion_pos, 'b c h w -> (h w) b c', b=b)
        query_input = rearrange(vis[-1], 'b c h w -> (h w) b c', b=b)
        word_proj = self.fusion_proj(word) 
        word_proj = rearrange(word_proj, 'b l c -> l b c')
        query_input = self.fusion(tgt=query_input,
                                memory=word_proj,
                                memory_key_padding_mask=pad_mask,
                                pos=None,
                                query_pos=visual_fusion_pos
        ) 
        for i in range(len(vis)-1):
            _, _, hi, wi = vis[i].shape
            visual_fusion_pos = self.all_fusion_pos[i](vis[i])
            visual_fusion_pos = rearrange(visual_fusion_pos, 'b c h w -> (h w) b c', b=b)
            word_proj = self.all_fusion_proj[i](word) 
            word_proj = rearrange(word_proj, 'b l c -> l b c')
            fusion_input = rearrange(vis[i], 'b c h w -> (h w) b c', b=b)
            fusion_input = self.all_fusion_module[i](tgt=fusion_input,
                                memory=word_proj,
                                memory_key_padding_mask=pad_mask,
                                pos=None,
                                query_pos=visual_fusion_pos) 
            vis[i] = rearrange(fusion_input, '(h w) b c -> b c h w', h=hi, w=wi)

        # Decoder input
        query_input = rearrange(query_input, '(h w) b c -> b c h w', h=h, w=w)  

        # PixelDecoder
        pixel_input = []
        for i in range(len(vis)):
            pixel_input.append(vis[i])
        pixel_input[-1] = query_input
        pixel_output = self.pixel_decoder(pixel_input, word, pad_mask)  
        
        # QueryDecoder
        query_output = self.query_decoder(query_input, state)    


        _, _, h, w = pixel_output.shape
        if not self.cfg.aux_loss:
            pred = torch.bmm(query_output, pixel_output.flatten(2)) 
            pred = rearrange(pred, 'b l (h w) -> b l h w', h=h, w=w)   
        else:
            for l, q in enumerate(query_output):
                final_output = []
                pred = torch.bmm(query_output[l], pixel_output.flatten(2))
                pred = rearrange(pred, 'b l (h w) -> b l h w', h=h, w=w)
                final_output.append(pred)
                
        return pred.detach()
