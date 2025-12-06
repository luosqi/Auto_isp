
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rgb_formate_change import  rgb_to_hsv, rgb_to_lum, hsv_to_rgb

##########################################################################
"""
ISP模块：去饱和
"""

class Desaturation(nn.Module):
    def forward (self, image,parameters):
        
        lum = rgb_to_lum(image)

        lum_3c = lum.repeat(1, 3, 1, 1)

        p = parameters.view(-1, 1, 1, 1)

        return (1.0 - p)*image + p*lum_3c
    
##########################################################################
"""
ISP模块：色彩校正矩阵 (CCM)
"""

class ColorCorrectionMatrix(nn.Module):
    """色彩校正矩阵模块"""
    
    def forward(self, img, p):
        # p: (Batch_Size, 9) 
        B, C, H, W = img.shape
        matrix = p.view(B, 3, 3)
        
        # 归一化步骤: 
        # sum(dim=2) 计算每行的和。 +1e-8 防止除以0。
        # 结果使得 matrix 的每一行加起来都等于 1。
        row_sums = matrix.sum(dim=2, keepdim=True) + 1e-8
        matrix = matrix / row_sums
        
        # 准备进行矩阵乘法
        # 将图像 (B, 3, H, W) 展平为 (B, 3, H*W) 
        # 这样每个像素点就是一个列向量 (R, G, B)^T
        flat_img = img.view(B, 3, -1)
        
        # torch.bmm (Batch Matrix Multiplication): 批量矩阵乘法
        # (B, 3, 3) * (B, 3, HW) -> (B, 3, HW)
        corrected = torch.bmm(matrix, flat_img)
        
        # 恢复图像形状
        return corrected.view(B, 3, H, W)

##########################################################################
"""
isp模块：对比度增强
"""

class Contrast(nn.Module):
    def forward(self, image, contrast_factors):
        """
        应用对比度增强
        
        Args:
            image: 输入图像，形状为 (B, C, H, W)，值域[0, 1]
            contrast_factors: 对比度增强因子，形状为 (B, 1)，表示每张图像的对比度调整强度
            
        Returns:
            对比度增强后的图像
        """
        B, C, H, W = image.shape
        contrast_factors = contrast_factors.view(B, 1, 1, 1)  # 调整形状以便广播
        
        # 计算图像的均值
        mean = torch.mean(image, dim=(2, 3), keepdim=True)
        
        # 应用对比度调整
        contrasted = (image - mean) * contrast_factors + mean
        contrasted = torch.clamp(contrasted, 0.0, 1.0)
        
        return contrasted
    
##########################################################################
"""
isp模块：降噪
"""

class Denoise(nn.Module):
    def forward(self, image, noise_levels):
        """
        应用降噪处理
        
        Args:
            image: 输入图像，形状为 (B, C, H, W)，值域[0, 1]
            noise_levels: 降噪强度，形状为 (B, 1)，表示每张图像的噪声水平
            
        Returns:
            降噪后的图像
        """
        B, C, H, W = image.shape
        noise_levels = noise_levels.view(B, 1, 1, 1)  # 调整形状以便广播
        
        # 简单的均值滤波作为降噪示例
        kernel_size = 3
        padding = kernel_size // 2
        kernel = torch.ones((C, 1, kernel_size, kernel_size), device=image.device) / (kernel_size * kernel_size)
        
        blurred = F.conv2d(image, kernel, padding=padding, groups=C)
        
        denoised = image * (1 - noise_levels) + blurred * noise_levels
        denoised = torch.clamp(denoised, 0.0, 1.0)
        
        return denoised
    
##########################################################################
"""
ISP模块：曝光
"""

class Exposure(nn.Module):
    def forward(self, image, exposure_values):
        """
        应用曝光调整
        
        Args:
            image: 输入图像，形状为 (B, C, H, W)，值域[0, 1]
            exposure_values: 曝光调整值，形状为 (B, 1)，表示每张图像的曝光增益
            
        Returns:
            曝光调整后的图像
        """
        B, C, H, W = image.shape
        exposure_values = exposure_values.view(B, 1, 1, 1)  # 调整形状以便广播
        p = torch.power(2,exposure_values)
        adjusted = image * p
        adjusted = torch.clamp(adjusted, 0.0, 1.0)
        return adjusted

##########################################################################
#ISP模块：伽马校正
class Gamma(nn.Module):
    def forward(self, image, gamma_values):
        """
        应用伽马校正
        
        Args:
            image: 输入图像，形状为 (B, C, H, W)，值域[0, 1]
            gamma_values: 伽马值，形状为 (B, 3)，对应RGB通道
            
        Returns:
            伽马校正后的图像
        """
        
        #return torch.pow(image.clamp(image,1e-8), gamma_values.view(-1, 3, 1, 1))

        B, C, H, W = image.shape
        gamma_values = gamma_values.view(B, C, 1, 1) # 调整形状以便广播
        image = torch.clamp(image, 1e-8)  # 避免对0进行幂运算
        corrected = torch.pow(image, gamma_values)
        return corrected
    
##########################################################################
"""
ISP模块：图像锐化
"""

class SharpenBlur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[1, 1, 1],
                               [1, 5, 1],
                               [1, 1, 1]], dtype=torch.float32) / 13.0
        kernel = kernel.view(1,1,3,3).repeat(3,1,1,1)  # 适用于3通道图像
        self.kernel = nn.Parameter(kernel)#nn.Parameter表示该张量是模型的可学习参数
        
    def forward(self, image , parameters):
        blurred = F.conv2d(image, self.kernel, padding=1,groups=3)

        parameters = parameters.view(-1, 1, 1, 1)
        image = image * parameters + blurred * (1 - parameters)
        return image
    
###########################################################################
"""
isp模块：色调映射
""" 

class ToneMapping(nn.Module):
    def forward(self, image, parameters):
        B, C, H, W = image.shape
        L = 8
        slopes = parameters#slopes.shape = [B, L]

        # 计算样本的总斜率
        P_L = torch.sum(slopes, dim=1).view(B, 1, 1, 1)

        out = torch.zeros_like(image)#.zeros_like(image)的作用是创建一个与image相同形状的零张量，并赋给out变量。
        
        for i in range(L):
            slope_i = slopes[:, i].view(B, 1, 1, 1)
            contribution = torch.clamp( L* image -  i, 0.0, 1.0)#clamp函数的作用是将输入张量的值限制在指定的范围内
            out += contribution * slope_i
        
        return out/(P_L + 1e-8)

###########################################################################

"""
ISP模块：白平衡
"""

class WhiteBalance(nn.Module):
    def forward(self, image, gains):
        """
        应用白平衡调整
        
        Args:
            image: 输入图像，形状为 (B, C, H, W)，值域[0, 1]
            gains: 白平衡增益，形状为 (B, 3)，对应RGB通道
            
        Returns:
            白平衡调整后的图像
        """
        B, C, H, W = image.shape
        gains = gains.view(B, C, 1, 1)             #.view()函数可以将张量进行形状的转换
        balanced = image * gains
        balanced = torch.clamp(balanced, 0.0, 1.0)
        return balanced
    
##########################################################################
"""
ISP模块：饱和度调整
""" 

class Saturation(nn.Module):
    def forward (self, image, parameters):
        hsv = rgb_to_yuv(image)
        h, s, v = hsv[:,0:1], hsv[:,1:2], hsv[:,2:3]

        s_prime = s + (1.0 - s) * (0.5 - torch.abs(0.5 -v)) * 0.8

        img_prime = yuv_to_rgb(torch.cat([h, s_prime, v], dim=1))

        lum = torch.clamp(rgb_to_lum(image), min=1e-8)     #防止除0

        endance = 0.5*(1.0 + torch.cos( img_prime * math )) / lum

        p = p.view(-1, 1, 1, 1)

        return (1.0 - p)*image + p*image*endance

