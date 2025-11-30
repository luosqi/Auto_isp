import cv2
import numpy as np
import os
from tqdm import tqdm

SRC_DIR = "data/linear/SIDD/0001_DNG"
OUT_DIR = "data/preview/SIDD/0001_DNG(1)"
os.makedirs(OUT_DIR, exist_ok=True) # 创建目录

def main():
    for name in tqdm(os.listdir(SRC_DIR), desc="Generating 8-bit previews"):
        if not name.lower().endswith(".png"): continue  #不区分大小写
        path = os.path.join(SRC_DIR, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
        img = img / 65535.0  
        img_srgb = np.clip(img ** (1/1.5), 0, 1)  # gamma校正
        img_srgb = (img_srgb * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, name.replace("_16bit", "_srgb")), img_srgb)
    print(" 8-bit 预览图生成完成")

if __name__ == "__main__":
    main()