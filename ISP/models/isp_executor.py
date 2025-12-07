import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import sys
sys.path.append('..')

from isp_module import (
    # Raw域
    BlackLevelCorrection, Raw_denoise, DifferentiableDemosaic, 
    DPC, Raw_WhiteBalance,
    # RGB域
    Desaturation, ColorCorrectionMatrix, Contrast, Denoise,
    Exposure, Gamma, SharpenBlur, ToneMapping, WhiteBalance, Saturation
)

class ISPExecutor(nn.Module):
    """
    ISP执行器：根据策略网络输出的动作，执行相应的ISP模块
    支持Raw域和RGB域的混合处理
    """
    def __init__(self, config_path='../config/isp_config.yaml'):
        super().__init__()
        
        # === 1. 初始化所有ISP模块 ===
        # Raw域模块 (ID 0-4)
        self.blc = BlackLevelCorrection()
        self.raw_dns = Raw_denoise()
        self.dpc = DPC()
        self.raw_wb = Raw_WhiteBalance()
        self.demosaic = DifferentiableDemosaic()
        
        # RGB域模块 (ID 5-14)
        self.desaturation = Desaturation()
        self.ccm = ColorCorrectionMatrix()
        self.contrast = Contrast()
        self.denoise = Denoise()
        self.exposure = Exposure()
        self.gamma = Gamma()
        self.sharpen_blur = SharpenBlur()
        self.tone_mapping = ToneMapping()
        self.white_balance = WhiteBalance()
        self.saturation = Saturation()
        
        # === 2. 动作映射表 ===
        self.action_map = {
            # Raw域
            0: ('blc', 0),           # BLC, 0参数
            1: ('raw_dns', 1),       # Raw降噪, 1参数
            2: ('dpc', 1),           # 坏点校正, 1参数
            3: ('raw_wb', 4),        # Raw白平衡, 4参数
            4: ('demosaic', 0),      # 去马赛克, 0参数（桥梁）
            
            # RGB域
            5: ('desaturation', 1),   # 去饱和, 1参数
            6: ('ccm', 9),            # CCM, 9参数
            7: ('contrast', 1),       # 对比度, 1参数
            8: ('denoise', 1),        # 降噪, 1参数
            9: ('exposure', 1),       # 曝光, 1参数
            10: ('gamma', 3),         # Gamma, 3参数
            11: ('sharpen_blur', 1),  # 锐化/模糊, 1参数
            12: ('tone_mapping', 8),  # 色调映射, 8参数
            13: ('white_balance', 3), # 白平衡, 3参数
            14: ('saturation', 1),    # 饱和度, 1参数
            
            15: ('stop', 0)           # 停止信号
        }
        
        # === 3. 参数切片索引（用于从35维参数向量提取） ===
        self.param_slices = self._build_param_slices()
        
    def _build_param_slices(self):
        """构建参数切片索引"""
        slices = {}
        start = 0
        
        for action_id, (name, n_params) in self.action_map.items():
            if n_params > 0:
                slices[action_id] = (start, start + n_params)
                start += n_params
            else:
                slices[action_id] = None
        
        return slices
    
    def execute_action(
        self, 
        image: torch.Tensor,
        action_id: int,
        params: torch.Tensor,
        is_raw: bool
    ) -> Tuple[torch.Tensor, bool]:
        """
        执行单个ISP操作
        
        Args:
            image: 输入图像 (B,C,H,W)，C=1(Raw) 或 C=3(RGB)
            action_id: 动作ID (0-15)
            params: 全部参数向量 (B, 35)
            is_raw: 当前是否处于Raw域
            
        Returns:
            output: 处理后的图像
            domain_changed: 域是否发生变化（Demosaic后从Raw变RGB）
        """
        domain_changed = False
        
        # 1. 获取动作信息
        module_name, n_params = self.action_map[action_id]
        
        # 2. 停止信号
        if module_name == 'stop':
            return image, False
        
        # 3. 提取当前动作的参数
        if n_params > 0:
            start, end = self.param_slices[action_id]
            action_params = params[:, start:end]
        else:
            action_params = None
        
        # 4. 执行对应模块
        if module_name == 'blc':
            if not is_raw:
                raise ValueError("BLC can only be applied in Raw domain")
            output = self.blc(image)
            
        elif module_name == 'raw_dns':
            if not is_raw:
                raise ValueError("Raw DNS can only be applied in Raw domain")
            output = self.raw_dns(image, action_params)
            
        elif module_name == 'dpc':
            if not is_raw:
                raise ValueError("DPC can only be applied in Raw domain")
            output = self.dpc(image, action_params)
            
        elif module_name == 'raw_wb':
            if not is_raw:
                raise ValueError("Raw WB can only be applied in Raw domain")
            output = self.raw_wb(image, action_params)
            
        elif module_name == 'demosaic':
            if not is_raw:
                raise ValueError("Demosaic can only be applied in Raw domain")
            output = self.demosaic(image)
            domain_changed = True  # 关键：域切换
            
        # RGB域模块
        elif module_name == 'desaturation':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.desaturation(image, action_params)
            
        elif module_name == 'ccm':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.ccm(image, action_params)
            
        elif module_name == 'contrast':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.contrast(image, action_params)
            
        elif module_name == 'denoise':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.denoise(image, action_params)
            
        elif module_name == 'exposure':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.exposure(image, action_params)
            
        elif module_name == 'gamma':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.gamma(image, action_params)
            
        elif module_name == 'sharpen_blur':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.sharpen_blur(image, action_params)
            
        elif module_name == 'tone_mapping':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.tone_mapping(image, action_params)
            
        elif module_name == 'white_balance':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.white_balance(image, action_params)
            
        elif module_name == 'saturation':
            if is_raw:
                raise ValueError("RGB module cannot be applied in Raw domain")
            output = self.saturation(image, action_params)
            
        else:
            raise ValueError(f"Unknown module: {module_name}")
        
        return output, domain_changed
    
    def forward(
        self,
        initial_image: torch.Tensor,
        action_sequence: torch.Tensor,
        param_sequence: torch.Tensor,
        is_initial_raw: bool = True
    ) -> torch.Tensor:
        """
        执行完整的ISP pipeline
        
        Args:
            initial_image: 初始图像 (B,C,H,W)
            action_sequence: 动作序列 (B,T)，T为最大步数
            param_sequence: 参数序列 (B,T,35)
            is_initial_raw: 初始图像是否为Raw
            
        Returns:
            final_image: 最终处理后的图像
        """
        B, T = action_sequence.shape
        current_image = initial_image
        is_raw = is_initial_raw
        
        for t in range(T):
            action_id = action_sequence[:, t]
            params = param_sequence[:, t, :]
            
            # 批处理：假设batch内所有样本执行相同动作
            # 实际训练时可能需要逐样本处理
            action_id_scalar = action_id[0].item()
            
            if action_id_scalar == 15:  # STOP
                break
            
            current_image, domain_changed = self.execute_action(
                current_image, action_id_scalar, params, is_raw
            )
            
            if domain_changed:
                is_raw = False
        
        return current_image


# === 测试代码 ===
if __name__ == '__main__':
    executor = ISPExecutor()
    
    # 测试Raw域处理
    raw_img = torch.rand(2, 1, 512, 512)  # 模拟Bayer图
    action_seq = torch.tensor([[0, 1, 4, 9, 15],   # BLC -> DNS -> Demosaic -> Exposure -> STOP
                                [2, 3, 4, 12, 15]]) # DPC -> WB -> Demosaic -> ToneMap -> STOP
    param_seq = torch.rand(2, 5, 35)
    
    output = executor(raw_img, action_seq, param_seq, is_initial_raw=True)
    print(f"✅ ISP Executor test passed!")
    print(f"   Input shape: {raw_img.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: (2, 3, 512, 512) - RGB output")