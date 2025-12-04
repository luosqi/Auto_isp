import os # 文件操作
import cv2
import rawpy # 读取RAW文件
import imageio # 保存图像
from tqdm import tqdm  # 进度条显示

RAW_DIR = "data/raw/SIDD/0001_DNG"
OUT_DIR = "data/linear/SIDD/0001_DNG"
os.makedirs(OUT_DIR, exist_ok=True)

def process_raw(raw_path):
    with rawpy.imread(raw_path) as raw: 
        rgb = raw.postprocess( #raw.postprocess()的作用：将RAW数据转换为RGB图像，具体参数包括使用相机白平衡、输出位数、是否自动亮度调整、 gamma 值
            use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True,
            gamma=(1,1)
        )
    return rgb

def main():
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.DNG') or f.endswith('.ARW')]
    for f in tqdm(raw_files, desc="Processing RAW files"):
        out_name = os.path.splitext(f)[0] + "_16bit.png"
        rgb = process_raw(os.path.join(RAW_DIR, f))
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(os.path.join(OUT_DIR, out_name), rgb_bgr)
    print(" RAW→16-bit PNG 转换完成")

if __name__ == "__main__":
    main()
