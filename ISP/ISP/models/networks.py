import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """
    共享特征提取骨干网 (Backbone)
    结构保持不变: 4层 Conv-BN-LRelu + FC
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.fc = nn.Linear(256 * 4 * 4, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PolicyNetwork(nn.Module):
    """
    混合域策略网络 (Mixed-Domain Policy Network)
    能够同时输出 Raw 域算子、Demosaic 算子和 RGB 域算子的决策。
    """
    def __init__(self):
        super().__init__()
        
        # === 1. 定义动作空间 ID 映射 ===
        # Raw Modules (来自 raw_module.py)
        # 0: BLC (无参)
        # 1: Raw_DNS (1参)
        # 2: DPC (1参)
        # 3: Raw_WB (4参)
        # 4: Demosaic (无参, 桥梁动作)
        
        # RGB Modules (来自 rgb_module.py)
        # 5: Desaturation (1参)
        # 6: CCM (9参)
        # 7: Contrast (1参)
        # 8: Denoise (1参)
        # 9: Exposure (1参)
        # 10: Gamma (3参)
        # 11: SharpenBlur (1参)
        # 12: ToneMapping (8参)
        # 13: WhiteBalance (3参)
        # 14: Saturation (1参)
        
        # 15: STOP (停止并输出)
        
        self.num_actions = 16
        
        # === 2. 计算参数总维度 ===
        # Raw: 0(BLC) + 1(DNS) + 1(DPC) + 4(WB) + 0(Demosaic) = 6
        # RGB: 1 + 9 + 1 + 1 + 1 + 3 + 1 + 8 + 3 + 1 = 29
        # Total = 35
        self.total_param_dim = 35
        
        # === 3. 输入通道定义 ===
        # Image (3) + History Mask (16) + Stage (1) + Domain Flag (1)
        # Domain Flag: 全0表示Raw域，全1表示RGB域
        input_channels = 3 + self.num_actions + 1 + 1
        
        self.backbone = FeatureExtractor(in_channels=input_channels)
        self.dropout = nn.Dropout(0.5)
        
        # 输出头
        self.action_head = nn.Linear(128, self.num_actions)
        self.param_head = nn.Linear(128, self.total_param_dim)

    def forward(self, img_preview, history_mask, stage_map, domain_map):
        """
        Args:
            img_preview: (B, 3, 64, 64) 
                - 如果当前是RGB状态，直接是RGB图。
                - 如果当前是Raw状态，必须先经过简单插值变成3通道伪RGB图。
            history_mask: (B, 16, 64, 64) 历史动作记录
            stage_map: (B, 1, 64, 64) 当前步数
            domain_map: (B, 1, 64, 64) 域指示器 (Raw=0, RGB=1)
        """
        # 拼接所有信息
        x = torch.cat([img_preview, history_mask, stage_map, domain_map], dim=1)
        
        features = self.backbone(x)
        features = self.dropout(features)
        
        action_logits = self.action_head(features)
        param_preds = self.param_head(features)
        
        return action_logits, param_preds

class ValueNetwork(nn.Module):
    """
    价值网络 (Critic)
    """
    def __init__(self):
        super().__init__()
        # Input: Image (3) + Stats (3) + Domain Flag (1)
        # Domain Flag 很有用，因为Raw图和RGB图的Value分布可能不同
        input_channels = 3 + 3 + 1
        self.backbone = FeatureExtractor(in_channels=input_channels)
        self.value_head = nn.Linear(128, 1)

    def forward(self, img_preview, img_stats, domain_map):
        x = torch.cat([img_preview, img_stats, domain_map], dim=1)
        features = self.backbone(x)
        return self.value_head(features)