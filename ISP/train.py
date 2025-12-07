import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from models.networks import PolicyNetwork, ValueNetwork
from models.environment import UnifiedRewardEnvironment
from models.isp_executor import ISPExecutor
from datasets.isp_dataset import ISPDataset


class AdaptiveISPTrainer:
    """
    AdaptiveISP训练器
    基于Actor-Critic算法实现强化学习训练
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 初始化网络
        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        self.isp_executor = ISPExecutor().to(self.device)
        self.env = UnifiedRewardEnvironment(
            mode=args.task_mode,
            yolo_size=args.input_size,
            lambda_c=args.lambda_c,
            lambda_r=args.lambda_r
        ).to(self.device)
        
        # 2. 优化器
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=3e-5)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=3e-5)
        
        # 3. 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.ExponentialLR(
            self.policy_optim, gamma=0.1**(1.0/(3*args.num_epochs))
        )
        self.value_scheduler = optim.lr_scheduler.ExponentialLR(
            self.value_optim, gamma=0.1**(1.0/(3*args.num_epochs))
        )
        
        # 4. 数据加载
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val')
        
        # 5. 日志
        self.writer = SummaryWriter(args.log_dir)
        self.global_step = 0
        
        # 6. RL超参数
        self.gamma = 0.99  # 折扣因子
        self.max_stages = 5  # 最大ISP阶段数
        
        print(f"🚀 Trainer initialized on {self.device}")
        print(f"   Task mode: {args.task_mode}")
        print(f"   Max stages: {self.max_stages}")
    
    def _build_dataloader(self, mode):
        dataset = ISPDataset(
            data_root=self.args.data_root,
            annotation_file=os.path.join(self.args.data_root, f'annotations/{mode}.json'),
            mode=mode,
            input_size=self.args.input_size,
            return_gt_rgb=(self.args.task_mode == 'quality')
        )
        
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(mode == 'train'),
            num_workers=4,
            pin_memory=True
        )
    
    def prepare_network_input(self, img, history_mask, stage, is_raw):
        """
        准备策略网络输入
        
        Args:
            img: (B, C, H, W) 当前图像
            history_mask: (B, 16) 历史动作记录
            stage: int 当前阶段
            is_raw: bool 是否为Raw域
        """
        B, C, H, W = img.shape
        
        # 1. 下采样图像到64x64用于策略网络
        img_preview = torch.nn.functional.interpolate(
            img, size=(64, 64), mode='bilinear', align_corners=False
        )
        
        # 2. History mask扩展到空间维度 (B, 16) -> (B, 16, 64, 64)
        history_map = history_mask.unsqueeze(-1).unsqueeze(-1).expand(B, 16, 64, 64)
        
        # 3. Stage map (B, 1, 64, 64)
        stage_map = torch.full((B, 1, 64, 64), stage / self.max_stages, 
                               device=self.device, dtype=torch.float32)
        
        # 4. Domain flag (B, 1, 64, 64)
        domain_map = torch.full((B, 1, 64, 64), 0.0 if is_raw else 1.0,
                                device=self.device, dtype=torch.float32)
        
        return img_preview, history_map, stage_map, domain_map
    
    def prepare_value_input(self, img, is_raw):
        """准备价值网络输入"""
        B = img.shape[0]
        
        # 下采样
        img_preview = torch.nn.functional.interpolate(
            img, size=(64, 64), mode='bilinear', align_corners=False
        )
        
        # 计算图像统计量（亮度、对比度、饱和度）
        if img.shape[1] == 3:  # RGB
            lum = 0.27 * img[:, 0] + 0.67 * img[:, 1] + 0.06 * img[:, 2]
            lum = lum.unsqueeze(1)
        else:  # Raw
            lum = img
        
        lum_mean = lum.mean(dim=[2, 3], keepdim=True).expand(B, 1, 64, 64)
        lum_std = lum.std(dim=[2, 3], keepdim=True).expand(B, 1, 64, 64)
        
        if img.shape[1] == 3:
            saturation = (img.max(dim=1, keepdim=True)[0] - 
                         img.min(dim=1, keepdim=True)[0])
            sat_mean = torch.nn.functional.interpolate(
                saturation, size=(64, 64), mode='bilinear', align_corners=False
            )
        else:
            sat_mean = torch.zeros(B, 1, 64, 64, device=self.device)
        
        img_stats = torch.cat([lum_mean, lum_std, sat_mean], dim=1)
        
        # Domain flag
        domain_map = torch.full((B, 1, 64, 64), 0.0 if is_raw else 1.0,
                                device=self.device, dtype=torch.float32)
        
        return img_preview, img_stats, domain_map

    def run_episode(self, batch, mode='train'):
        """
        运行一个episode
        自动根据环境模式选择合适的输入
        """
        images = batch['image'].to(self.device)
        B = images.shape[0]
    
        # === 根据任务模式准备数据 ===
        if self.args.task_mode == 'detection':
         # 检测模式需要targets
            targets = {
                'boxes': batch['targets']['boxes'].to(self.device),
                'labels': batch['targets']['labels'].to(self.device)
            }
            gt_img = None
    
        elif self.args.task_mode == 'quality':
            # 质量模式需要GT图像
            if 'gt_rgb' not in batch:
                raise ValueError("Quality mode requires 'gt_rgb' in batch")
            gt_img = batch['gt_rgb'].to(self.device)
            targets = None
    
        else:
            raise ValueError(f"Unknown task mode: {self.args.task_mode}")
    
        # 初始化
        current_img = images
        is_raw = (images.shape[1] == 1)
        history_mask = torch.zeros(B, 16, device=self.device)
    
        log_probs = []
        values = []
        rewards = []
        entropies = []
    
        # === 执行ISP Pipeline ===
        for stage in range(self.max_stages):
            # 1. 准备输入
            img_preview, history_map, stage_map, domain_map = \
                self.prepare_network_input(current_img, history_mask, stage, is_raw)
        
            # 2. 策略网络预测
            action_logits, all_params = self.policy_net(
                img_preview, history_map, stage_map, domain_map
            )
        
            # 3. 采样动作
            action_probs = torch.softmax(action_logits, dim=1)
        
            if mode == 'train':
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                entropy = action_dist.entropy()
            else:
                action = action_probs.argmax(dim=1)
                log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
                entropy = torch.zeros_like(log_prob)
        
            # 4. 停止检查
            if (action == 15).all():
                break
        
            # 5. 执行ISP
            next_img, domain_changed = self.isp_executor.execute_action(
                current_img, action[0].item(), all_params, is_raw
            )
        
            if domain_changed:
                is_raw = False
        
            # 6. 更新历史
            for b in range(B):
                history_mask[b, action[b]] = 1.0
        
            # 7. 计算奖励（关键：根据模式传入不同参数）
            reward, loss_next, truncated = self.env.get_reward(
                current_img, 
                next_img, 
                action, 
                history_mask,
                targets=targets,      # 检测模式使用
                gt_img=gt_img         # 质量模式使用
            )
        
            # 8. 价值评估
            val_preview, val_stats, val_domain = self.prepare_value_input(current_img, is_raw)
            state_value = self.value_net(val_preview, val_stats, val_domain)
        
            # 9. 记录
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            entropies.append(entropy)
        
            # 10. 更新状态
            current_img = next_img
        
            if truncated:
                break
    
        # === 计算损失 ===
        if len(rewards) == 0:
            return {'total_loss': torch.tensor(0.0)}
    
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
    
        returns = torch.tensor(returns, device=self.device)
    
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
        policy_loss = 0
        value_loss = 0
    
        for log_p, v, R, ent in zip(log_probs, values, returns, entropies):
            advantage = R - v.detach()
            policy_loss -= (log_p * advantage).mean()
            value_loss += F.mse_loss(v, R.unsqueeze(0))
        
            if mode == 'train' and self.args.entropy_coef > 0:
                policy_loss -= self.args.entropy_coef * ent.mean()
    
        total_loss = policy_loss + value_loss
    
        # === 反向传播 ===
        if mode == 'train':
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            total_loss.backward()
        
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        
            self.policy_optim.step()
            self.value_optim.step()
    
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_reward': torch.stack(rewards).mean().item(),
            'num_stages': len(rewards)
    }

def parse_args():
        parser = argparse.ArgumentParser(description='AdaptiveISP Training')
    
        # 数据
        parser.add_argument('--data_root', type=str, default='./data/LOD',
                           help='Root directory of dataset')
        parser.add_argument('--input_size', type=int, default=512,
                           help='Input image size')
    
        # 任务模式 (关键参数)
        parser.add_argument('--task_mode', type=str, default='detection',
                        choices=['detection', 'quality'],
                        help='Task mode: detection or quality')
    
        # 训练
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--val_interval', type=int, default=5)
        parser.add_argument('--save_interval', type=int, default=10)
    
        # RL超参数
        parser.add_argument('--lambda_c', type=float, default=0.05,
                           help='Time cost penalty coefficient')
        parser.add_argument('--lambda_r', type=float, default=2.0,
                           help='Reuse penalty coefficient')
        parser.add_argument('--entropy_coef', type=float, default=0.01,
                           help='Entropy regularization coefficient')
    
        # 检测模式专用
        parser.add_argument('--num_classes', type=int, default=80,
                           help='Number of detection classes (for COCO)')
        parser.add_argument('--use_simple_det_loss', action='store_true',
                           help='Use simplified detection loss')
    
        # 路径
        parser.add_argument('--save_dir', type=str, default='./checkpoints')
        parser.add_argument('--log_dir', type=str, default='./logs')
    
        # GPU
        parser.add_argument('--gpu', type=int, default=0)
    
        return parser.parse_args()


# === 使用示例 ===
if __name__ == '__main__':
    """
    使用方法：
    
    1. 训练检测模式（论文主要任务）：
       python train.py --task_mode detection \
                      --data_root ./data/LOD \
                      --batch_size 8 \
                      --num_epochs 100 \
                      --use_simple_det_loss
    
    2. 训练质量模式（对比实验）：
       python train.py --task_mode quality \
                      --data_root ./data/paired \
                      --batch_size 8 \
                      --num_epochs 100
    
    3. 调整惩罚系数：
       python train.py --task_mode detection \
                      --lambda_c 0.01 \  # 降低时间惩罚，允许更多模块
                      --lambda_r 5.0      # 增强重复使用惩罚
    """
    args = parse_args()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # 创建训练器
    trainer = AdaptiveISPTrainer(args)
    
    # 打印配置
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Task Mode:      {args.task_mode}")
    print(f"Data Root:      {args.data_root}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Num Epochs:     {args.num_epochs}")
    print(f"Lambda_c:       {args.lambda_c}")
    print(f"Lambda_r:       {args.lambda_r}")
    print("="*60 + "\n")
    
    # 开始训练
    trainer.train()