import torch
import torch.nn as nn
import torch.nn.functional as F

class BayerUtils:
    @staticmethod
    def split_bayer(raw_img):
        """
        将 (B, 1, H, W) 的 Bayer 图拆分为 (B, 4, H/2, W/2)
        顺序: R, Gr, Gb, B (假设 RGGB 模式)
        """
        # 0::2 代表偶数索引，1::2 代表奇数索引
        r = raw_img[:, :, 0::2, 0::2]
        gr = raw_img[:, :, 0::2, 1::2]
        gb = raw_img[:, :, 1::2, 0::2]
        b = raw_img[:, :, 1::2, 1::2]
        return torch.cat([r, gr, gb, b], dim=1)

    @staticmethod
    def merge_bayer(bayer_4c):
        """
        将 (B, 4, H/2, W/2) 合并回 (B, 1, H, W)
        """
        B, C, H_sub, W_sub = bayer_4c.shape
        out = torch.zeros(B, 1, H_sub*2, W_sub*2, device=bayer_4c.device)
        
        out[:, :, 0::2, 0::2] = bayer_4c[:, 0:1, :, :] # R
        out[:, :, 0::2, 1::2] = bayer_4c[:, 1:2, :, :] # Gr
        out[:, :, 1::2, 0::2] = bayer_4c[:, 2:3, :, :] # Gb
        out[:, :, 1::2, 1::2] = bayer_4c[:, 3:4, :, :] # B
        
        return out