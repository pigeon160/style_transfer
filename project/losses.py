"""
计算Gram矩阵与三个损失
"""

import torch as th
import torch.nn.functional as F

# 计算gram矩阵
def gram_matrix(feat : th.Tensor) -> th.Tensor:
  if feat.dim() == 4 :
    assert feat.shape[0] == 1
    feat = feat.squeeze(0)
  c, h, w = feat.shape
  flat = feat.reshape(c, h * w)
  return flat @ flat.T / (c * h * w)

# 计算存在mask情况下的gram矩阵
def masked_gram(feat : th.Tensor, mask : th.Tensor) -> th.Tensor :
  if feat.dim() == 4 :
    assert feat.shape[0] == 1
    feat = feat.squeeze(0)
  c, h, w = feat.shape
  m = F.adaptive_avg_pool2d(mask, (h, w))[0]
  flat = (feat * m).reshape(c, h * w)
  return flat @ flat.T / (c * m.sum()).clamp(min = 1e-6)

# 计算内容损失
def content_loss(feat_gen : th.Tensor, feat_target : th.Tensor) -> th.Tensor :
  return F.mse_loss(feat_gen, feat_target)

# 计算mask存在情况下的内容损失
def masked_content_loss(
    feat_gen : th.Tensor, 
    feat_target : th.Tensor, 
    mask : th.Tensor) -> th.Tensor :
  c, h, w = feat_gen.shape
  m = F.adaptive_avg_pool2d(mask, (h, w))[0]
  num = ((feat_gen - feat_target) ** 2 * m).sum()
  num = num / (m.sum() * c).clamp(min = 1e-6)
  return num

# 计算风格损失
def style_loss(
    feats_gen : dict[str, th.Tensor], 
    style_gram_targets : dict[str, th.Tensor], 
    weights : dict[str, float] | None = None) -> th.Tensor :
  total = 0.0
  for name, target in style_gram_targets.items() :
    w = 1.0 if weights is None else weights.get(name, 1.0)
    total += w * F.mse_loss(gram_matrix(feats_gen[name]), target)
  return total

# 计算存在mask情况下的风格损失
def masked_style_loss(
    feats_gen : dict[str, th.Tensor], 
    style_gram_targets : dict[str, th.Tensor], 
    mask : th.Tensor, 
    weights : dict[str, float] | None = None) -> th.Tensor :
  total = 0.0
  for name, target in style_gram_targets.items() :
    w = 1.0 if weights is None else weights.get(name, 1.0)
    total += w * F.mse_loss(masked_gram(feats_gen[name], mask), target)
  return total

# 计算总分变损失
def tv_loss(x : th.Tensor) -> th.Tensor :
  dh = th.diff(x, dim = 2).abs().mean()
  dw = th.diff(x, dim = 3).abs().mean()
  return dh + dw

# 计算mask存在情况下的总分变损失
def masked_tv_loss(x : th.Tensor, mask : th.Tensor) -> th.Tensor : 
  c = x.shape[1]
  dh = th.diff(x, dim = 2).abs()
  dw = th.diff(x, dim = 3).abs()
  mh, mw = mask[:, :, 1:, :], mask[:, :, :, 1:]
  dh = (dh * mh).sum() / (mh.sum() * c).clamp(min = 1e-6)
  dw = (dw * mw).sum() / (mw.sum() * c).clamp(min = 1e-6)
  return dh + dw