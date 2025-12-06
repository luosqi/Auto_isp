import torch

def rgb_to_lum(img):
    """
    计算亮度 (Luminance)
    
    公式: L = 0.27*R + 0.67*G + 0.06*B
    这是标准的亮度转换公式，反映了人眼对绿色最敏感，蓝色最不敏感的特性。
    
    参数:
        img: (B, 3, H, W) 输入图像张量
    返回:
        lum: (B, 1, H, W) 单通道亮度张量
    """
    # 切片操作 img[:, 0:1, :, :] 保持了维度为 (B, 1, H, W)，方便后续广播计算
    return 0.27 * img[:, 0:1, :, :] + 0.67 * img[:, 1:2, :, :] + 0.06 * img[:, 2:3, :, :]


def rgb_to_hsv(img):
    """
    可微的 RGB -> HSV 转换
    
    难点: 通常 RGB转HSV 包含大量的 if-else 判断（例如：如果 R 最大，H=...；如果 G 最大，H=...）。
    这种逻辑在神经网络中会导致梯度断裂或效率低下。
    
    解决方案: 使用 torch.eq 生成掩码 (mask)，并行计算所有情况，最后组合结果。
    """
    eps = 1e-8 # 极小值，防止分母为 0 导致 NaN (Not a Number) 错误
    
    # max/min: 在通道维度 (dim=1) 寻找最大最小值
    # keepdim=True 保持 (B, 1, H, W) 形状，防止维度压缩
    max_val, _ = img.max(dim=1, keepdim=True)
    min_val, _ = img.min(dim=1, keepdim=True)
    d = max_val - min_val # 色度差 (Chroma)
    
    # 1. 计算 V (Value/Brightness)
    v = max_val
    
    # 2. 计算 S (Saturation)
    # 如果 max_val 为 0 (黑色)，S 应该为 0。使用 eps 避免除以 0。
    s = d / (max_val + eps)
    
    # 3. 计算 H (Hue)
    # 提取 RGB 分量
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    h = torch.zeros_like(v) # 初始化 H 张量
    
    # 创建掩码 (Mask): 对应位置如果为 True，说明该像素该通道是最大值
    mask_r = (max_val == r) 
    mask_g = (max_val == g)
    mask_b = (max_val == b)
    
    # 并行计算不同情况下的 H 值
    # 逻辑: if R is max, h = (g - b) / d
    h[mask_r] = (g[mask_r] - b[mask_r]) / (d[mask_r] + eps)
    # 逻辑: if G is max, h = 2 + (b - r) / d
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / (d[mask_g] + eps)
    # 逻辑: if B is max, h = 4 + (r - g) / d
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / (d[mask_b] + eps)
    
    # 将 H 归一化到 [0, 1] 范围
    h = (h / 6.0) % 1.0
    
    # torch.cat: 在通道维度拼接 H, S, V
    return torch.cat([h, s, v], dim=1)

def hsv_to_rgb(hsv):
    """
    可微的 HSV -> RGB 转换
    
    逻辑: 将色相环分为 6 个扇区 (0-6)，根据 H 落在哪个扇区决定 RGB 的组合方式。
    """
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    c = v * s # Chroma
    x = c * (1 - torch.abs((h * 6) % 2 - 1)) # 第二大分量
    m = v - c # 匹配值 (添加到 RGB 以调整亮度)
    
    zero = torch.zeros_like(c)
    
    # 将 H 映射到 0-6 区间
    h6 = (h * 6)
    r, g, b = zero.clone(), zero.clone(), zero.clone()
    
    # 使用掩码处理分段函数 (Piecewise function)
    # 0 <= H < 1: (C, X, 0)
    mask = (h6 < 1)
    r[mask], g[mask], b[mask] = c[mask], x[mask], zero[mask]
    
    # 1 <= H < 2: (X, C, 0)
    mask = (h6 >= 1) & (h6 < 2)
    r[mask], g[mask], b[mask] = x[mask], c[mask], zero[mask]
    
    # ... 后续扇区依此类推 ...
    mask = (h6 >= 2) & (h6 < 3)
    r[mask], g[mask], b[mask] = zero[mask], c[mask], x[mask]
    
    mask = (h6 >= 3) & (h6 < 4)
    r[mask], g[mask], b[mask] = zero[mask], x[mask], c[mask]
    
    mask = (h6 >= 4) & (h6 < 5)
    r[mask], g[mask], b[mask] = x[mask], zero[mask], c[mask]
    
    mask = (h6 >= 5)
    r[mask], g[mask], b[mask] = c[mask], zero[mask], x[mask]
    
    rgb = torch.cat([r+m, g+m, b+m], dim=1)
    
    # torch.clamp: 确保数值严格在 [0, 1] 之间，防止数值漂移
    return torch.clamp(rgb, 0, 1)