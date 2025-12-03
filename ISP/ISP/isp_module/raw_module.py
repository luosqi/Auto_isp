import torch
import torch.nn as nn   
import torch.nn.functional as F
from .raw_formate_change import BayerUtils


"""
ISP模块：黑电平校正
"""

class BlackLevelCorrection(nn.Module):

    """
    def auto_detect_parameters(self, image):
    # 基于图像直方图估算黑电平
    # 假设图像暗部像素代表黑电平
    black_estimate = torch.quantile(image.flatten(), 0.01)  # 取1%分位数
    
    # 根据最大值估算位深度
    max_pixel = torch.max(image)
    estimated_bit_depth = int(torch.log2(max_pixel + 1).ceil().item())
    
    return black_estimate.item(), estimated_bit_depth
    """
    def __init__(self, black_level=1024.0 ,bit_depth=14):
        super().__init__()
        self.black_level = black_level
        self.bit_depth = bit_depth
        self.max_value = 2 ** self.bit_depth - 1

    def forward(self, image):

        out = image - self.black_level
        out = torch.clamp(out, 0, self.max_value)
        return out
    
    """
    def forward(self, image, auto_detect=False):
    if auto_detect:
        black_level, bit_depth = self.auto_detect_parameters(image)
    else:
        black_level, bit_depth = self.black_level, self.bit_depth
    
    max_value = 2 ** bit_depth - 1
    out = image - black_level
    out = torch.clamp(out, 0, max_value)
    return out
    """

###########################################################################


"""
ISP 模块: Raw 域降噪
"""
class Raw_denoise(nn.Module):
    """
    Raw Domain Denoise (DNS)
    实现: Split Bayer -> 3x3 Conv (Blur) -> Blend -> Merge
    """
    def __init__(self):
        super().__init__()
        # 定义一个简单的 3x3 平滑核 (类似高斯)
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=torch.float32) / 16.0
        # 扩展为 (4, 1, 3, 3) 适应 4 个 Bayer 通道的分组卷积
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3).repeat(4, 1, 1, 1))

    def forward(self, raw_img, p):
        """
        p: (B, 1) 降噪强度
        """
        # 1. 拆分
        x = BayerUtils.split_bayer(raw_img) # (B, 4, H/2, W/2)
        
        # 2. 滤波
        # groups=4 保证 R 通道只和 R 通道卷积
        blurred = F.conv2d(x, self.kernel, padding=1, groups=4)
        
        # 3. 混合
        # p=0: 无降噪; p=1: 强降噪
        strength = torch.sigmoid(p).view(-1, 1, 1, 1)
        denoised_x = (1.0 - strength) * x + strength * blurred
        
        # 4. 合并
        return BayerUtils.merge_bayer(denoised_x)
    
    ###########################################################################
"""
ISP模块：可微分去马赛克
"""

class DifferentiableDemosaic(nn.Module):
    """
    基于固定卷积核的双线性插值去马赛克 (Bilinear Demosaicing)
    支持反向传播，适用于 RGGB 模式的 Bayer 阵列。
    """
    def __init__(self):
        super().__init__()
        
        # 定义卷积核权重
        # 1. G 通道的卷积核: 
        # 在 R/B 位置，G = (G_up + G_down + G_left + G_right) / 4
        # 在 G 位置，保持原值 (1)
        k_g = torch.tensor([[0, 0.25, 0],
                            [0.25, 1, 0.25],
                            [0, 0.25, 0]], dtype=torch.float32)
        
        # 2. R/B 通道的卷积核:
        # 在 G 位置(水平/垂直)，均值为 0.5 * (邻居)
        # 在对角线位置(异色)，均值为 0.25 * (四个角)
        # 中心点保持原值(1)
        k_rb = torch.tensor([[0.25, 0.5, 0.25],
                             [0.5, 1, 0.5],
                             [0.25, 0.5, 0.25]], dtype=torch.float32)

        # 注册为 Buffer，不参与梯度更新
        self.register_buffer('k_g', k_g.view(1, 1, 3, 3))
        self.register_buffer('k_rb', k_rb.view(1, 1, 3, 3))

    def forward(self, raw_img):
        """
        Args:
            raw_img: (B, 1, H, W) RGGB Bayer Pattern
        Returns:
            rgb_img: (B, 3, H, W) Linear RGB
        """
        B, C, H, W = raw_img.shape
        
        # 1. 构建掩码 (Masks) 标识 R, G, B 的位置
        # RGGB 模式坐标:
        # R: (0,0), (0,2)... -> y%2==0, x%2==0
        # Gr:(0,1), (0,3)... -> y%2==0, x%2==1
        # Gb:(1,0), (1,2)... -> y%2==1, x%2==0
        # B: (1,1), (1,3)... -> y%2==1, x%2==1
        
        # 生成网格坐标
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=raw_img.device), 
                                        torch.arange(W, device=raw_img.device), 
                                        indexing='ij')
        
        # 这里的 % 2 操作生成 0/1 掩码
        mask_r = (y_grid % 2 == 0) & (x_grid % 2 == 0)
        mask_b = (y_grid % 2 == 1) & (x_grid % 2 == 1)
        mask_g = ~(mask_r | mask_b) # 剩下的都是 G
        
        # 将掩码转换为 float 用于计算
        mask_r = mask_r.float().view(1, 1, H, W)
        mask_b = mask_b.float().view(1, 1, H, W)
        mask_g = mask_g.float().view(1, 1, H, W)
        
        # 2. 分离通道 (此时每个通道包含大量 0)
        raw_r = raw_img * mask_r
        raw_g = raw_img * mask_g
        raw_b = raw_img * mask_b
        
        # 3. 应用卷积插值
        # padding=1 保持尺寸不变
        
        # 计算全分辨率 R 通道
        # 输入是稀疏的 R (只有1/4有点)，通过卷积填补空缺
        # 注意：因为输入包含0，直接卷积后数值会偏小，需要修正？
        # 其实标准做法是：只要核设计对了（上面定义的 k_rb），直接卷积稀疏图即可还原插值
        r_out = F.conv2d(raw_r, self.k_rb, padding=1)
        
        # 计算全分辨率 B 通道
        b_out = F.conv2d(raw_b, self.k_rb, padding=1)
        
        # 计算全分辨率 G 通道
        g_out = F.conv2d(raw_g, self.k_g, padding=1)
        
        # 4. 拼接
        return torch.cat([r_out, g_out, b_out], dim=1)
    
############################################################################
"""
ISP模块：坏点矫正
"""

class DPC(nn.Module):
    """
    Differentiable Defective Pixel Correction (DPC)
    """
    def __init__(self, threshold=0.1):
        super().__init__()
        # 默认阈值，可以通过 Parameter 变为可学习
        self.base_threshold = threshold

    def forward(self, raw_img, p):
        """
        p: (B, 1) 控制检测坏点的灵敏度 (Sensitivity)
        """
        # 1. 拆分 Bayer 通道 (B, 4, H/2, W/2)
        # 这样每个通道内的 3x3 邻域都是同色像素
        x = BayerUtils.split_bayer(raw_img)
        
        # 2. 计算 3x3 邻域均值 (使用 AvgPool 模拟)
        # padding=1 保持尺寸不变
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        # 3. 计算当前像素与均值的差异
        diff = torch.abs(x - local_mean)
        
        # 4. 构建软掩码 (Soft Mask)
        # 如果 diff > threshold, mask -> 1 (认为是坏点，需要替换)
        # 如果 diff < threshold, mask -> 0 (认为是正常点，保留原值)
        # p 用来动态调整阈值：threshold' = base_threshold * (1 - p)
        # 这里的 scaling_factor 控制 Sigmoid 的陡峭程度，模拟 if-else
        sensitivity = torch.sigmoid(p).view(-1, 1, 1, 1)
        thresh = self.base_threshold * (1.0 - sensitivity * 0.5) 
        scaling_factor = 100.0 
        
        mask = torch.sigmoid((diff - thresh) * scaling_factor)
        
        # 5. 融合修复
        # Out = (1-mask)*Origin + mask*Mean
        corrected_x = (1.0 - mask) * x + mask * local_mean
        
        # 6. 合并回 Bayer 图
        return BayerUtils.merge_bayer(corrected_x)
    
############################################################################
"""
isp模块：raw域白平衡
"""

class Raw_WhiteBalance(nn.Module):
    def __init__(self, pattern='RGGB'):
        super().__init__()
        self.pattern = pattern
    def forward(self, image, p):
        B, C, H, W = image.shape
        mask = torch.zeros(B, 1, H, W).cuda()

        if self.pattern == 'RGGB':
            r_gain = p[:, 0].view(B, 1, 1, 1)
            gr_gain = p[:, 1].view(B, 1, 1, 1)
            gb_gain = p[:, 2].view(B, 1, 1, 1)
            b_gain = p[:, 3].view(B, 1, 1, 1)

            mask[:, :, ::2, ::2] = r_gain
            mask[:, :, ::2, 1::2] = gr_gain
            mask[:, :, 1::2, ::2] = gb_gain
            mask[:, :, 1::2, 1::2] = b_gain

        return image * mask
    
###########################################################################