"""
图像读写与预处理
"""

import torch as th
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "data/content"
STYLE_DIR = ROOT / "data/style"
OUT_DIR = ROOT / "outputs"
MASK_DIR = ROOT / "data/mask"
OUT_DIR.mkdir(parents = True, exist_ok = True)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 读入图片，裁剪成size*size转化为张量
def load_image(path : str | Path, size : int) -> th.Tensor :
  img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
  img = transforms.Resize(size)(img)
  img = transforms.CenterCrop(size)(img)
  img = np.array(img, dtype = np.float32) / 255
  img = th.from_numpy(img)
  img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
  return img

# 读入掩膜并进行羽化，类型和格式与img对齐
def load_mask(path : str | Path, size : int, feather : int = 0) -> th.Tensor :
  img = ImageOps.exif_transpose(Image.open(path)).convert("L")
  img = transforms.Resize(size)(img)
  img = transforms.CenterCrop(size)(img)
  if feather > 0 :
    img = img.filter(ImageFilter.GaussianBlur(feather))
  img = np.array(img, dtype = np.float32) / 255
  img = th.from_numpy(img).unsqueeze(0).unsqueeze(0)
  return img

# 把张量转化为PIL类型的图片
def to_pil(x : th.Tensor) -> Image.Image :
  x = x.detach().clamp(0, 1)
  if x.dim() == 4 :
    x = x.squeeze(0)
  x = x.permute(1, 2, 0).numpy()
  x = (x * 255).round().astype("uint8")
  x = Image.fromarray(x)
  return x

# 用imagenet的数据来标准化
def normalize(x : th.Tensor) -> th.Tensor :
  return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)