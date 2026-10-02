"""
加载并冻结 VGG19
"""

import torch as th
from pathlib import Path
from torchvision.models import vgg19

LAYERS = {
  "relu1_1" : 1, "relu2_1" : 6, "relu3_1" : 11, 
  "relu4_1" : 20, "relu4_2" : 22, "relu5_1" : 29
}
CONTENT_LAYER = "relu4_2"
STYLE_LAYERS = ("relu1_1","relu2_1","relu3_1","relu4_1","relu5_1")
MAX_LAYER_IDX = max(LAYERS.values())
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models/vgg19-dcbb9e9d.pth"

# 加载预训练模型的feature层并冻结参数
def load_vgg(weights_path : str | Path = DEFAULT_WEIGHTS) -> th.nn.Sequential :
  model = vgg19(weights = None).features
  state = th.load(weights_path)
  state = {k[len("features.") :] : v 
           for k, v in state.items()
           if k.startswith("features.")
          }
  missing, unexpected = model.load_state_dict(state, strict = False)
  if missing or unexpected :
    raise RuntimeError(f"权重不匹配 missing = {missing} unexpected = {unexpected}")
  model.eval()
  for p in model.parameters() :
    p.requires_grad_(False)
  return model

# 手动做一次前向传播，返回指定层（ReLU 激活）的结果
def extract_features(vgg, x : th.Tensor, names : str | tuple[str, ...]) -> dict[str, th.Tensor] :
  if isinstance(names, str) :
    names = [names]
  wanted = {LAYERS[n] : n for n in names}
  out = {}
  for idx, layer in enumerate(vgg) :
    x = layer(x)
    if idx in wanted :
      out[wanted[idx]] = x
    if idx >= MAX_LAYER_IDX :
      break
  return out