import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class YOLOv3Loss(nn.Module):
    """
    YOLOv3损失函数
    复现论文中使用的检测Loss，用于强化学习的奖励计算
    
    Loss包含三部分：
    1. 定位损失 (bbox regression)
    2. 置信度损失 (objectness)
    3. 分类损失 (class prediction)
    """
    def __init__(
        self,
        num_classes: int = 80,
        anchors: List[Tuple[int, int]] = None,
        img_size: int = 512,
        ignore_thresh: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh
        
        # YOLOv3使用3个尺度，每个尺度3个anchor
        if anchors is None:
            # COCO dataset的默认anchors (相对于416x416)
            # 按大小排序：小 -> 中 -> 大
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # 小目标 (52x52)
                [(30, 61), (62, 45), (59, 119)],     # 中目标 (26x26)
                [(116, 90), (156, 198), (373, 326)]  # 大目标 (13x13)
            ]
        else:
            self.anchors = anchors
        
        # 缩放到当前输入尺寸
        self.scaled_anchors = []
        for scale_anchors in self.anchors:
            scaled = [(w * img_size / 416, h * img_size / 416) 
                     for w, h in scale_anchors]
            self.scaled_anchors.append(scaled)
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        
    def forward(
        self, 
        predictions: List[torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            predictions: List of 3 tensors, shapes:
                - (B, 3, 13, 13, 5+num_classes)  # 大目标
                - (B, 3, 26, 26, 5+num_classes)  # 中目标
                - (B, 3, 52, 52, 5+num_classes)  # 小目标
            targets: Dict with keys:
                - 'boxes': (B, N, 4) normalized [x1, y1, x2, y2]
                - 'labels': (B, N) class indices
                
        Returns:
            total_loss: scalar tensor
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        
        total_loss = torch.zeros(1, device=device)
        
        # 遍历3个尺度
        for scale_idx, pred in enumerate(predictions):
            # pred shape: (B, 3, H, W, 5+C)
            B, num_anchors, grid_h, grid_w, _ = pred.shape
            
            # 构建该尺度的targets
            scale_targets = self._build_targets(
                pred, targets, self.scaled_anchors[scale_idx], 
                grid_h, grid_w, scale_idx
            )
            
            # 分离预测值
            pred_boxes = pred[..., :4]      # (B, 3, H, W, 4)
            pred_conf = pred[..., 4:5]      # (B, 3, H, W, 1)
            pred_cls = pred[..., 5:]        # (B, 3, H, W, C)
            
            # 获取mask
            obj_mask = scale_targets['obj_mask']      # (B, 3, H, W, 1)
            noobj_mask = scale_targets['noobj_mask']  # (B, 3, H, W, 1)
            
            # 1. 定位损失 (只计算有目标的格子)
            if obj_mask.sum() > 0:
                target_boxes = scale_targets['boxes']  # (B, 3, H, W, 4)
                
                # 使用MSE Loss (论文中的做法)
                box_loss = self.mse_loss(
                    pred_boxes[obj_mask.squeeze(-1)],
                    target_boxes[obj_mask.squeeze(-1)]
                )
                total_loss += box_loss
            
            # 2. 置信度损失
            # 2a. 有目标的格子：预测为1
            if obj_mask.sum() > 0:
                conf_loss_obj = self.bce_loss(
                    pred_conf[obj_mask],
                    torch.ones_like(pred_conf[obj_mask])
                )
                total_loss += conf_loss_obj
            
            # 2b. 无目标的格子：预测为0
            if noobj_mask.sum() > 0:
                conf_loss_noobj = self.bce_loss(
                    pred_conf[noobj_mask],
                    torch.zeros_like(pred_conf[noobj_mask])
                )
                # 无目标损失权重降低（论文中的处理）
                total_loss += 0.5 * conf_loss_noobj
            
            # 3. 分类损失 (只计算有目标的格子)
            if obj_mask.sum() > 0:
                target_cls = scale_targets['classes']  # (B, 3, H, W, C)
                cls_loss = self.bce_loss(
                    pred_cls[obj_mask.squeeze(-1)],
                    target_cls[obj_mask.squeeze(-1)]
                )
                total_loss += cls_loss
        
        # 归一化 (除以batch size)
        total_loss = total_loss / batch_size
        
        return total_loss
    
    def _build_targets(
        self,
        pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        anchors: List[Tuple[float, float]],
        grid_h: int,
        grid_w: int,
        scale_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        构建该尺度的训练目标
        
        Returns:
            targets_dict: Dict with keys:
                - 'obj_mask': (B, 3, H, W, 1) bool, 有目标的位置
                - 'noobj_mask': (B, 3, H, W, 1) bool, 无目标的位置
                - 'boxes': (B, 3, H, W, 4) 目标框坐标
                - 'classes': (B, 3, H, W, C) one-hot类别
        """
        device = pred.device
        B = pred.size(0)
        num_anchors = len(anchors)
        
        # 初始化
        obj_mask = torch.zeros(B, num_anchors, grid_h, grid_w, 1, 
                              dtype=torch.bool, device=device)
        noobj_mask = torch.ones(B, num_anchors, grid_h, grid_w, 1,
                               dtype=torch.bool, device=device)
        
        target_boxes = torch.zeros(B, num_anchors, grid_h, grid_w, 4, device=device)
        target_cls = torch.zeros(B, num_anchors, grid_h, grid_w, self.num_classes, 
                                device=device)
        
        # 处理每个样本
        for b in range(B):
            boxes = targets['boxes'][b]  # (N, 4) [x1, y1, x2, y2] normalized
            labels = targets['labels'][b]  # (N,)
            
            if len(boxes) == 0:
                continue
            
            # 转换为中心点格式
            boxes_cxcywh = self._xyxy_to_cxcywh(boxes)  # (N, 4) [cx, cy, w, h]
            
            # 缩放到grid尺寸
            boxes_cxcywh[:, 0] *= grid_w
            boxes_cxcywh[:, 1] *= grid_h
            boxes_cxcywh[:, 2] *= self.img_size
            boxes_cxcywh[:, 3] *= self.img_size
            
            # 为每个GT框分配anchor
            for box_idx, (box, label) in enumerate(zip(boxes_cxcywh, labels)):
                if label < 0:  # 忽略标签
                    continue
                
                cx, cy, w, h = box
                
                # 找到该框所在的grid cell
                grid_x = int(cx)
                grid_y = int(cy)
                
                if grid_x >= grid_w or grid_y >= grid_h:
                    continue
                
                # 计算与所有anchor的IoU，选择最佳anchor
                anchor_ious = []
                for anchor_w, anchor_h in anchors:
                    iou = self._bbox_iou_wh(w, h, anchor_w, anchor_h)
                    anchor_ious.append(iou)
                
                best_anchor_idx = np.argmax(anchor_ious)
                
                # 设置该位置为有目标
                obj_mask[b, best_anchor_idx, grid_y, grid_x, 0] = True
                noobj_mask[b, best_anchor_idx, grid_y, grid_x, 0] = False
                
                # 设置目标值
                # bbox: 相对于grid cell的偏移
                tx = cx - grid_x
                ty = cy - grid_y
                tw = torch.log(w / anchors[best_anchor_idx][0] + 1e-16)
                th = torch.log(h / anchors[best_anchor_idx][1] + 1e-16)
                
                target_boxes[b, best_anchor_idx, grid_y, grid_x] = \
                    torch.tensor([tx, ty, tw, th], device=device)
                
                # class: one-hot
                target_cls[b, best_anchor_idx, grid_y, grid_x, int(label)] = 1.0
        
        return {
            'obj_mask': obj_mask,
            'noobj_mask': noobj_mask,
            'boxes': target_boxes,
            'classes': target_cls
        }
    
    @staticmethod
    def _xyxy_to_cxcywh(boxes):
        """[x1,y1,x2,y2] -> [cx,cy,w,h]"""
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        return torch.stack([cx, cy, w, h], dim=1)
    
    @staticmethod
    def _bbox_iou_wh(w1, h1, w2, h2):
        """计算两个框的IoU (仅基于宽高)"""
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter_area = inter_w * inter_h
        
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / (union_area + 1e-16)


class YOLOv3Detector(nn.Module):
    """
    YOLOv3检测器包装类
    用于RL训练中计算检测loss
    """
    def __init__(
        self,
        num_classes: int = 80,
        img_size: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 加载预训练的YOLOv3
        try:
            # 尝试使用ultralytics的实现
            from ultralytics import YOLO
            self.model = YOLO('yolov3.pt')
            self.use_ultralytics = True
            print("✅ Loaded YOLOv3 from ultralytics")
        except:
            # 备选：使用torchvision的Faster R-CNN作为替代
            print("⚠️ ultralytics not available, using Faster R-CNN as fallback")
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.use_ultralytics = False
        
        # 冻结backbone（论文要求）
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("🔒 YOLOv3 backbone frozen")
        
        self.model.eval()
        
        # Loss计算器
        self.loss_fn = YOLOv3Loss(
            num_classes=num_classes,
            img_size=img_size
        )
    
    def forward(self, images: torch.Tensor, targets: Dict = None):
        """
        Args:
            images: (B, 3, H, W) RGB图像
            targets: 训练时的GT标注
            
        Returns:
            如果targets=None: 返回预测结果
            如果targets!=None: 返回loss
        """
        if targets is None:
            # 推理模式
            with torch.no_grad():
                return self.model(images)
        else:
            # 训练模式：计算loss
            # 这里我们需要获取中间特征来计算YOLO loss
            return self._compute_loss(images, targets)
    
    def _compute_loss(self, images, targets):
        """
        计算YOLOv3检测loss
        
        注意：由于不同YOLO实现差异，这里提供简化版本
        实际项目中建议直接使用ultralytics的loss计算
        """
        if self.use_ultralytics:
            # ultralytics的YOLO可以直接计算loss
            results = self.model.train()
            # 需要调用模型的loss计算接口
            # 这里简化处理：使用模型的forward返回
            outputs = self.model.model(images)
            
            # 提取3个尺度的预测
            # 注意：实际实现需要根据具体模型结构调整
            predictions = self._extract_predictions(outputs)
            
            # 计算loss
            loss = self.loss_fn(predictions, targets)
            return loss
        else:
            # 使用Faster R-CNN的loss
            # 转换targets格式
            target_list = []
            for b in range(len(targets['boxes'])):
                target_list.append({
                    'boxes': targets['boxes'][b] * self.img_size,  # 反归一化
                    'labels': targets['labels'][b]
                })
            
            loss_dict = self.model(images, target_list)
            # Faster R-CNN返回loss字典
            total_loss = sum(loss for loss in loss_dict.values())
            return total_loss
    
    def _extract_predictions(self, outputs):
        """
        从模型输出中提取3个尺度的预测
        这个函数需要根据具体的YOLO实现调整
        """
        # 占位实现
        # 实际使用时需要根据加载的模型结构修改
        if isinstance(outputs, (list, tuple)):
            return outputs[:3]  # 假设前3个是不同尺度
        else:
            # 如果是单个tensor，需要手动split
            # 这里返回简化版本
            return [outputs, outputs, outputs]


# === 简化版本：直接使用检测结果作为loss ===
class SimpleDetectionLoss(nn.Module):
    """
    简化的检测loss计算
    当无法获取YOLOv3内部loss时，使用检测结果反推loss
    
    思路：检测效果越好 -> loss越小 -> reward越大
    """
    def __init__(self, img_size=512):
        super().__init__()
        self.img_size = img_size
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 检测结果列表，每个元素是一个样本的检测框
                格式: List[Dict] with keys 'boxes', 'scores', 'labels'
            targets: GT标注
            
        Returns:
            loss: 基于检测精度计算的伪loss
        """
        device = targets['boxes'].device
        batch_size = targets['boxes'].size(0)
        
        total_loss = torch.zeros(1, device=device)
        
        for b in range(batch_size):
            gt_boxes = targets['boxes'][b]  # (N, 4)
            gt_labels = targets['labels'][b]  # (N,)
            
            if len(predictions) > b:
                pred_boxes = predictions[b]['boxes']  # (M, 4)
                pred_scores = predictions[b]['scores']  # (M,)
                pred_labels = predictions[b]['labels']  # (M,)
                
                # 计算匹配得分
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    # 计算IoU矩阵
                    ious = self._box_iou(pred_boxes, gt_boxes)
                    
                    # 对每个GT，找最佳匹配
                    max_ious, _ = ious.max(dim=0)
                    
                    # Loss = 1 - average_max_iou
                    # (IoU越高，loss越低)
                    loss = 1.0 - max_ious.mean()
                else:
                    # 没有预测或没有GT
                    loss = torch.tensor(1.0, device=device)
            else:
                loss = torch.tensor(1.0, device=device)
            
            total_loss += loss
        
        return total_loss / batch_size
    
    @staticmethod
    def _box_iou(boxes1, boxes2):
        """计算IoU矩阵"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / (union + 1e-6)
        return iou