"""
主循环，风格迁移
"""
import csv
import time
import dataio
import torch as th
from pathlib import Path
import model as model_mod
import losses as loss_mod

SIZE, STEPS = 256, 80
ALPHA, BETA, GAMMA = 1.0, 5e4, 1.0
LR = 0.01
CONTENT_IMG = dataio.CONTENT_DIR / "daisy.jpg"
STYLE_IMG = dataio.STYLE_DIR / "starry_night.jpg"
OUT_IMG = dataio.OUT_DIR / "stylized.png"
OUT_LOG = dataio.OUT_DIR / "loss.csv"
FEATHER = 6
USE_MASK = True
MASK = dataio.MASK_DIR / "daisy_mask.png"
SNAP_DIR = dataio.OUT_DIR / "snapshots"
SNAPSHOT_STEPS = (1, 3, 5, 10, 20, 40, 60, 80)
SEED = 42

th.set_num_threads(4)
th.manual_seed(SEED)

# 读图+载入预训练模型
content = dataio.load_image(CONTENT_IMG, SIZE)
style = dataio.load_image(STYLE_IMG, SIZE)
vgg = model_mod.load_vgg()
names = (model_mod.CONTENT_LAYER, *model_mod.STYLE_LAYERS)

# 得到目标内容矩阵和gram矩阵
with th.no_grad() :
  c_target = model_mod.extract_features(
    vgg, 
    dataio.normalize(content), 
    [model_mod.CONTENT_LAYER]
  )[model_mod.CONTENT_LAYER][0].detach()
  s_feat = model_mod.extract_features(
    vgg, 
    dataio.normalize(style), 
    model_mod.STYLE_LAYERS
  )
  s_targets = {k : loss_mod.gram_matrix(v).detach() for k, v in s_feat.items()}

# 初始化生成图和优化器,读入掩膜
generated = content.clone().requires_grad_(True)
if USE_MASK :
  region_mask = dataio.load_mask(MASK, SIZE, FEATHER)
  content_mask = 1.0 - region_mask
optimizer = th.optim.LBFGS(
  [generated], 
  lr = 1.0, 
  max_iter = 20, 
  history_size = 10, 
  line_search_fn = "strong_wolfe"
)
history = []

# 先对原图做一次前向传播并算损失
with th.no_grad() :
  f0 = model_mod.extract_features(
    vgg, 
    dataio.normalize(generated), 
    names
  )
  l_c0 = (loss_mod.content_loss(f0[model_mod.CONTENT_LAYER][0], c_target) 
          if not USE_MASK else 
          loss_mod.masked_content_loss(f0[model_mod.CONTENT_LAYER][0], c_target, content_mask))
  l_s0 = (loss_mod.style_loss(f0, s_targets)
          if not USE_MASK else
          loss_mod.masked_style_loss(f0, s_targets, region_mask))
  l_t0 = (loss_mod.tv_loss(generated)
          if not USE_MASK else
          loss_mod.masked_tv_loss(generated, region_mask))
  history.append({
    "step" : 0, 
    "total" : float((ALPHA * l_c0 + BETA * l_s0 + GAMMA * l_t0).detach()), 
    "content" : float(l_c0.detach()), 
    "style" : float(l_s0.detach()), 
    "tv" : float(l_t0.detach()), 
    "sec" : 0.0
  })

# 为优化器定义一个可调用的闭包
parts = {}
def closure() :
  with th.no_grad() :
    generated.clamp_(0, 1)
  optimizer.zero_grad(set_to_none = True)
  feats = model_mod.extract_features(
    vgg, 
    dataio.normalize(generated), 
    names
  )
  l_c = (loss_mod.content_loss(feats[model_mod.CONTENT_LAYER][0], c_target) 
         if not USE_MASK else 
         loss_mod.masked_content_loss(feats[model_mod.CONTENT_LAYER][0], c_target, content_mask))
  l_s = (loss_mod.style_loss(feats, s_targets)
         if not USE_MASK else
         loss_mod.masked_style_loss(feats, s_targets, region_mask))
  l_t = (loss_mod.tv_loss(generated)
         if not USE_MASK else
         loss_mod.masked_tv_loss(generated, region_mask))
  loss = l_c * ALPHA + l_s * BETA + l_t * GAMMA
  loss.backward()
  parts.update({
    "content" : l_c.detach(), 
    "style" : l_s.detach(), 
    "tv" : l_t.detach(), 
  })
  return loss

# 开始循环迭代
snapshots = {}
t0 = time.time()
for step in range(1, STEPS + 1) :
  loss = optimizer.step(closure)
  l_c, l_s, l_t = parts["content"], parts["style"], parts["tv"]
  loss_fix = float((ALPHA * l_c + BETA * l_s + GAMMA * l_t).detach())
  history.append({
      "step" : step,  
      "total" : loss_fix, 
      "content" : float(l_c.detach()), 
      "style" : float(l_s.detach()), 
      "tv" : float(l_t.detach()), 
      "sec" : time.time() - t0
    })
  if step in SNAPSHOT_STEPS :
    result = (generated if not USE_MASK else 
              (region_mask * generated + (1 - region_mask) * content).clamp(0, 1))
    snapshots[step] = dataio.to_pil(result)
    print(f"step : {step} | loss : {loss_fix:.2f} | loss_content : {l_c:.2e} | loss_style : {l_s:2e} | loss_tv : {l_t:2e}")

# 成果保存
result = (generated if not USE_MASK else 
          (region_mask * generated + (1 - region_mask) * content).clamp(0, 1))
dataio.to_pil(result).save(OUT_IMG)
with open(OUT_LOG, "w", newline = "") as f :
  w = csv.DictWriter(f, fieldnames = list(history[0].keys()))
  w.writeheader()
  w.writerows(history)
SNAP_DIR.mkdir(parents = True, exist_ok = True)
for s, im in sorted(snapshots.items()) :
  im.save(SNAP_DIR / f"step_{s:04d}.png")