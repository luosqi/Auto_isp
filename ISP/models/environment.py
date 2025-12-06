from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from torchvision.models import ResNet18_Weights
except ImportError:  # 兼容旧版本 torchvision
    ResNet18_Weights = None  # type: ignore


class ObjectDetectionEvaluator(nn.Module):
    """
    目标检测评估器 (基于 YOLO)
    论文中使用 YOLO-v3 [cite: 70]。
    为了工程复现方便，这里提供一个通用的 Loss 计算接口。
    """
    def __init__(self, model_type='yolov3', input_size=512):
        super().__init__()
        self.input_size = input_size
        self.model_type = model_type
        
        # 1. 加载模型
        # 这里为了演示，我们使用 torchvision 的 FasterRCNN 或 Retinanet 作为占位符
        # 在实际项目中，建议使用 ultralytics (YOLOv5/8) 或专门的 YOLOv3 库
        # 关键是：我们需要模型的 Loss Function，而不是推理结果
        
        # 模拟一个 Feature Extractor + Head 用于计算梯度
        # 注意：实际部署时请替换为你具体的 YOLOv3 模型实例
        self.model = self._load_yolo_model()
        
        # 2. 冻结参数 (关键步骤 [cite: 217])
        # "pre-trained YOLOv3 model remains unaltered throughout the training process"
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

    def _load_yolo_model(self):
        """
        加载预训练模型。
        实际复现时，建议加载具体的权重文件 (.pt)
        """
        # 这里使用一个简单的预训练网络代替，实际请替换为：
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # 或者加载论文指定的 darknet yolov3
        print(f"Loading Frozen {self.model_type} for Reward Calculation...")
        # 仅作结构演示
        if ResNet18_Weights is not None:
            return torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        return torchvision.models.resnet18(pretrained=True)

    def resize_for_yolo(self, img):
        """
        将 ISP 输出 (任意尺寸) 调整为 YOLO 需要的尺寸 (如 512x512 )
        """
        return F.interpolate(img, size=(self.input_size, self.input_size), 
                             mode='bilinear', align_corners=False)

    def get_detection_loss(self, img, targets):
        """
        计算检测 Loss。这是 Reward 的核心来源。
        
        Args:
            img: (B, 3, H, W) ISP 处理后的图像
            targets: 目标检测的标签 (Bboxes, Classes)
        
        Returns:
            loss: scalar tensor (可微分)
        """
        # 1. 预处理 (Resize & Normalize)
        # YOLO 通常需要 [0, 1] 或 [0, 255]，且需要归一化
        img_resized = self.resize_for_yolo(img)
        
        # 2. 前向传播计算 Loss
        # 注意：这里是一个伪代码逻辑，因为不同 YOLO 库的 loss 调用方式不同
        # 如果是 ultralytics: loss, _ = model(img_resized, targets)
        
        # --- 模拟 Loss 用于代码跑通 ---
        # 假设我们希望图像特征接近某种高层语义（这里仅为占位逻辑）
        # 在真实复现中，这里必须是 model(img, targets)[0] (YOLO Loss)
        fake_prediction = self.model(img_resized) 
        # 模拟一个 Loss：希望预测结果尽可能强（仅用于测试管道连通性）
        dummy_loss = torch.mean(torch.abs(fake_prediction)) 
        
        return dummy_loss

class ImageQualityEvaluator(nn.Module):
    """
    图像质量评估器 (Image Quality Task)
    用于对比实验或当目标是优化画质时。
    参考论文  (Exposure 方法)
    """
    def __init__(self, metric='l1'):
        super().__init__()
        self.metric = metric

    def forward(self, pred_img, gt_img):
        """
        Args:
            pred_img: ISP 输出图
            gt_img: Ground Truth (如长曝光参考图)
        """
        if self.metric == 'l1':
            return F.l1_loss(pred_img, gt_img)
        elif self.metric == 'mse':
            return F.mse_loss(pred_img, gt_img)
        elif self.metric == 'psnr':
            mse = F.mse_loss(pred_img, gt_img)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            # 注意：Loss 应该是越小越好，所以返回 -PSNR 或 1/PSNR
            return -psnr 
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

class UnifiedRewardEnvironment(nn.Module):
    """
    统一环境管理器
    支持 'detection' 和 'quality' 两种模式切换，同时加入模块数量与图像异常截断规则
    """
    def __init__(self, mode='detection', yolo_size=512 , lambda_c=0.05, lambda_r=2.0):
        super().__init__()
        self.mode = mode
        self.lambda_c = lambda_c #时间惩罚系数
        self.lambda_r = lambda_r #复用惩罚系数
        
        # 初始化检测器
        if self.mode == 'detection':
            self.detector = ObjectDetectionEvaluator(input_size=yolo_size)
        
        # 初始化画质评估器
        self.quality_eval = ImageQualityEvaluator(metric='l1')

        costs_ms = torch.tensor([1.0,10.0,2.0,1.0,3.0,1.9,1.9,2.1,10.0,1.7,2.0,5.3,2.7,1.7,2.0])

        self.register_buffer('costs_table', costs_ms / costs_ms.max())
        self.max_modules = 5  # 限制可使用的最大模块数量

    def check_abnormal(self, img):
        """
        检查异常截断 (Truncated)
        需求: 防止图像全黑或全白 (cite: 异常截断)
        """
        # img shape: [B, 3, H, W]
        # 计算每张图的均值
        means = img.view(img.size(0), -1).mean(dim=1)
        
        # 阈值判定：太黑 (<0.02) 或 太亮 (>0.98)
        # 返回一个 Bool Tensor [Batch]
        is_abnormal = (means < 0.02) | (means > 0.98)
        return is_abnormal

    def get_reward(
        self,
        current_img: torch.Tensor,
        next_img: torch.Tensor,
        action_idx: Union[int, torch.Tensor],
        history_mask: Optional[torch.Tensor] = None,
        targets: Optional[Any] = None,
        gt_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        计算这一步的奖励 (Step Reward)
        公式 (4): r = D(s_t) - D(s_{t+1}) 
        意味着：如果误差降低了 (s_{t+1} < s_t)，奖励为正。

        Args:
            current_img: 当前状态下的图像 (B, 3, H, W)
            next_img: 执行动作后的图像 (B, 3, H, W)
            action_idx: 当前执行的动作索引 (int 或 LongTensor)
            history_mask: 历史动作掩码，记录已经执行过的动作
            targets: 检测模式下的目标标签
            gt_img: 画质模式下的参考图像

        Returns:
            final_reward: 奖励值 (标量 tensor)
            loss_next: 下一状态的任务损失 (用于反向传播)
            truncated: 是否触发异常截断 (超出模块限制或图像异常)

        Notes:
            - 当累计使用的处理模块数量超过 self.max_modules (默认 5) 时触发截断。
            - 若输出图像出现过暗/过亮异常，同样触发截断并施加惩罚。
        """
        device = next_img.device  
        dtype = next_img.dtype

        batch_size = next_img.size(0)
        base_reward = torch.zeros((), dtype=dtype, device=device)  
        loss_next = torch.zeros((), dtype=dtype, device=device)

        if self.mode == 'detection':
            assert targets is not None, "Detection mode requires targets (labels)"
            
            # 1. 计算处理前的 Loss
            with torch.no_grad():
                loss_prev = self.detector.get_detection_loss(current_img, targets)
            
            # 2. 计算处理后的 Loss (需要梯度，以便训练 ISP 参数)
            loss_next = self.detector.get_detection_loss(next_img, targets)
            
            # Reward = 之前的误差 - 现在的误差
            # 误差变小 -> Reward > 0
            base_reward = loss_prev - loss_next
            
            

        elif self.mode == 'quality':
            assert gt_img is not None, "Quality mode requires Ground Truth image"
            
            # 计算画质损失 (如 L1 Loss)
            with torch.no_grad():
                loss_prev = self.quality_eval(current_img, gt_img)
            
            loss_next = self.quality_eval(next_img, gt_img)
            
            base_reward = loss_prev - loss_next
            

        else:
            raise ValueError("Invalid mode")
        
        # A. 时间惩罚 (Time Cost Penalty)
        # 查表获取当前动作的归一化耗时
        # 兼容 action_idx 是 int 或 Tensor 的情况
        costs_table = self.costs_table.to(device=device)
        if isinstance(action_idx, int):
            current_cost = costs_table[action_idx]
        elif isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.to(device=device, dtype=torch.long)
            current_cost = costs_table.index_select(0, action_idx.view(-1)).mean()
        else:
            raise TypeError("action_idx must be int or torch.Tensor")

        current_cost = current_cost.to(dtype)
        time_penalty = current_cost * current_cost.new_tensor(self.lambda_c)
        
        # B. 重复使用惩罚 (Reuse Penalty)
        reuse_penalty = torch.zeros((), dtype=dtype, device=device)
        extra_penalty = torch.zeros((), dtype=dtype, device=device)
        if history_mask is not None:
            history_mask = history_mask.to(device=device, dtype=dtype)
            # 逻辑：如果 history_mask 在对应的 action_idx 位置是 1，则惩罚
            # 这里简化处理：假设 history_mask 已经包含了当前动作之前的状态
            
            # 如果是单张图片处理 (Batch=1 或 simple loop)
            if isinstance(action_idx, int):
                history_value = history_mask[..., action_idx]
                if torch.as_tensor(history_value, device=device).float().mean() > 0:
                    reuse_penalty = reuse_penalty + reuse_penalty.new_tensor(self.lambda_r)
            
            # 如果是 Batch 处理 (简单的平均惩罚逻辑，视具体 RL 实现而定)
            elif isinstance(action_idx, torch.Tensor):
                # 获取每个样本是否重复使用: gather history_mask 对应 action_idx 的值
                # history_mask shape: (B, Num_Actions)
                action_idx_view = action_idx.view(-1, 1).to(dtype=torch.long)
                if history_mask.ndim == 1:
                    selected = history_mask.index_select(0, action_idx_view.view(-1))
                else:
                    selected = history_mask.gather(1, action_idx_view)
                reuse_penalty = selected.float().mean() * reuse_penalty.new_tensor(self.lambda_r)

        # C. 模块使用数与图像异常检测 -> 截断逻辑
        truncated_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if history_mask is not None:
            mask_positive = history_mask > 0

            if mask_positive.ndim == 1:
                used_count = int(mask_positive.sum().item())
                if isinstance(action_idx, int) and 0 <= action_idx < mask_positive.shape[0]:
                    if not mask_positive[action_idx]:
                        used_count += 1
                if used_count > self.max_modules:
                    truncated_flags = torch.ones_like(truncated_flags, dtype=torch.bool)
            elif mask_positive.ndim == 2:
                modules_used = mask_positive.sum(dim=1).float()
                if isinstance(action_idx, torch.Tensor):
                    action_idx_long = action_idx.view(-1, 1).to(device=device, dtype=torch.long)
                    fresh_mask = ~(mask_positive.gather(1, action_idx_long))
                    modules_used = modules_used + fresh_mask.float().squeeze(1)
                modules_exceeded = modules_used > self.max_modules
                truncated_flags = truncated_flags | modules_exceeded.to(device=device)
            else:
                modules_used = mask_positive.view(batch_size, -1).sum(dim=1).float()
                modules_exceeded = modules_used > self.max_modules
                truncated_flags = truncated_flags | modules_exceeded.to(device=device)

        abnormal_flags = self.check_abnormal(next_img)
        truncated_flags = truncated_flags | abnormal_flags.to(device=device)

        if truncated_flags.any():
            # 触发截断时给予附加惩罚，鼓励策略提前停止或者避免异常
            penalty_scale = truncated_flags.float().mean().to(device=device, dtype=dtype)
            extra_penalty = (self.lambda_c + self.lambda_r) * penalty_scale

        truncated = bool(truncated_flags.any().item())
        
        # 3. 最终组合 (公式 9)
        # ---------------------------------------------------
        # r = Gain - P_time - P_reuse
        final_reward = base_reward - time_penalty - reuse_penalty - extra_penalty
        
        return final_reward, loss_next, truncated
