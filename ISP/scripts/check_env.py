import torch
import rawpy
import cv2
from bayes_opt import BayesianOptimization

print("✅ DreamISP 环境配置成功！")

# 打印版本信息
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# GPU检测
if torch.cuda.is_available():
    print(f"🚀 GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 当前运行在 CPU 模式")

# 简单算例测试
x = torch.rand(3, 3)
print("\n测试矩阵计算:\n", torch.mm(x, x))
