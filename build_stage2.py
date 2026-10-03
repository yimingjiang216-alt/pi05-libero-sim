# -*- coding: utf-8 -*-
# 生成阶段2 notebook: LIBERO + lerobot/pi05_libero_base
import json, io

CELLS = []

def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(keepends=True)})

md("""# π0.5 驱动 LIBERO 仿真 (Kaggle T4, 阶段 2)
权重: `lerobot/pi05_libero_base` (LIBERO 微调版, 官方 LeRobot 移植)
环境: LIBERO (robosuite 1.4.1 + MuJoCo) — libero_spatial 任务
流程: 官方预处理管线 (状态 token 化 + 指令 tokenize) → π0.5 出 7 维动作 → robosuite 执行
产出: 每集 MP4 视频 + 成功率统计""")

code("""# GPU 自检
import os
os.environ["MUJOCO_GL"] = "egl"          # T4 离屏渲染
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", gpu.stdout.strip() or "无")
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显存: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))""")

code("""!pip install -q -U "lerobot[dataset]" "transformers>=5.5,<5.6" 2>&1 | tail -1
import importlib
print("lerobot:", importlib.util.find_spec("lerobot") is not None)""")

code("""# 安装 LIBERO: 用官方发行包 hf-libero (自带 bddl/init 文件, 锁定 robosuite==1.4.0 + mujoco 3.x)
!pip install -q "hf-libero" 2>&1 | tail -3
import importlib
print("libero:", importlib.util.find_spec("libero") is not None)
print("robosuite:", importlib.util.find_spec("robosuite") is not None)
import mujoco
print("mujoco:", mujoco.__version__)""")

code("""# LIBERO 环境冒烟: libero_spatial 第 0 个任务
import os, yaml
# 首次导入会 input() 问数据集路径 —— 非交互环境会报错, 预先生成配置绕过
import libero as _l  # 定位包路径
_libero_root = os.path.join(os.path.dirname(_l.__file__), "libero")
_cfg_dir = os.path.expanduser("~/.libero")
os.makedirs(_cfg_dir, exist_ok=True)
_cfg = {
    "benchmark_root": _libero_root,
    "bddl_files": os.path.join(_libero_root, "bddl_files"),
    "init_states": os.path.join(_libero_root, "init_files"),
    "datasets": os.path.join(_libero_root, "../datasets"),
    "assets": os.path.join(_libero_root, "assets"),
}
yaml.dump(_cfg, open(os.path.join(_cfg_dir, "config.yaml"), "w"))
print("libero config 预置完成:", os.path.join(_cfg_dir, "config.yaml"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("任务:", task.language)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
for k, v in obs.items():
    if hasattr(v, "shape"):
        print("  %-28s %s %s" % (k, v.shape, v.dtype))
print("=== LIBERO 环境就绪 (动作维度 7, 由 pi05_libero_base 配置固定) ===")""")

code("""# 加载 pi05_libero_base (meta 设备构造 + 逐张量 fp16 流式加载, T4 16GB 可容纳)
from huggingface_hub import snapshot_download
import torch, shutil, json
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import lerobot.policies.pi05.processor_pi05 as _pp05
from lerobot.processor import RobotProcessorPipeline

# 关键修复: 训练用 lerobot 0.4.1, 其 prepare 步骤会把 8 维状态补零到 max_state_dim=32
# 再离散化进 prompt; 0.6.1 删掉了补零 -> prompt 里状态 token 数不对 -> 模型失效 (0/3 根因)
# 这里给 0.6.1 的类打回 0.4.1 行为
import torch.nn.functional as _F
from copy import deepcopy as _deepcopy
from lerobot.lerobot_types import TransitionKey as _TK
from lerobot.utils.constants import OBS_STATE as _OBS_STATE
import numpy as _np
_orig_prepare_call = _pp05.Pi05PrepareStateTokenizerProcessorStep.__call__
def _prepare_call_with_pad(self, transition):
    obs = transition.get(_TK.OBSERVATION, {})
    s = obs.get(_OBS_STATE)
    if s is not None and s.shape[-1] < self.max_state_dim:
        obs = dict(obs)
        obs[_OBS_STATE] = _F.pad(s, (0, self.max_state_dim - s.shape[-1]))
        transition[_TK.OBSERVATION] = obs
    return _orig_prepare_call(self, transition)
_pp05.Pi05PrepareStateTokenizerProcessorStep.__call__ = _prepare_call_with_pad
print("prepare 步骤已补 0.4.1 的 pad 行为 (8 -> 32 维)")

REPO = "bigbangoslab/pi05_libero_openpiconfig_on_lerobot_chunk30"
# lerobot 格式的 LIBERO 微调 pi05: lerobot/libero 数据 30k 步 batch128 2xH200,
# type=pi05 与已装版本兼容, 自带归一化统计量 (pre/post step safetensors)
print("downloading", REPO, "...")
local = snapshot_download(REPO)

local_dir = "/root/pi05_libero"
os.makedirs(local_dir, exist_ok=True)
for fn in os.listdir(local):
    if fn != "model.safetensors" and os.path.isfile(os.path.join(local, fn)):
        shutil.copy(os.path.join(local, fn), os.path.join(local_dir, fn))
cfgd = json.load(open(os.path.join(local_dir, "config.json")))
cfgd["device"] = "cpu"; cfgd["dtype"] = "float32"
json.dump(cfgd, open(os.path.join(local_dir, "config.json"), "w"))

# 预/后处理器: 官方 tokenizer 是 gated 仓库, 换同词表(256k)的 Gemma 镜像
for jfn in ("policy_preprocessor.json", "policy_postprocessor.json"):
    jfp = os.path.join(local_dir, jfn)
    if not os.path.exists(jfp):
        continue
    ppd = json.load(open(jfp))
    changed = False
    for step in ppd.get("steps", []):
        c = step.get("config", {})
        if "tokenizer_name" in c:
            c["tokenizer_name"] = "unsloth/gemma-7b-it"
            changed = True
    if changed:
        json.dump(ppd, open(jfp, "w"))
        print(jfn, "tokenizer 替换 -> unsloth/gemma-7b-it")
pre = RobotProcessorPipeline.from_pretrained(local_dir, "policy_preprocessor.json")
print("预处理器步骤:", [type(s).__name__ for s in pre.steps])
# 后处理器: 动作反归一化 (之前漏掉, 导致动作尺度错误 -> 0/3)
post = RobotProcessorPipeline.from_pretrained(local_dir, "policy_postprocessor.json")
print("后处理器步骤:", [type(s).__name__ for s in post.steps])

# 官方加载: bf16 checkpoint 7.5GB, T4 15.6GB 直接放得下, 无需 meta 设备技巧
# (此前的 fp32 流式加载是为 14.5GB 的 pi05_base 设计的, 对本权重反而引入静默错位风险)
policy = PI05Policy.from_pretrained(local)
policy = policy.to("cuda", dtype=torch.float16).eval()
print("π0.5 就绪 | dtype:", next(policy.parameters()).dtype,
      "| GPU mem: %.1f GB" % (torch.cuda.memory_allocated() / 1e9))""")

code("""# 冒烟推理: 用真环境观测跑一步, 确认接口
from robosuite.utils.transform_utils import quat2axisangle
import numpy as np

def make_batch(obs, instruction):
    def img(x):  # HWC uint8 -> 翻转180° -> CHW float 0-1 (官方 LiberoProcessorStep 同款)
        x = x[::-1, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(x).unsqueeze(0)
    # 状态布局与训练数据 (lerobot/libero stats 证实): eef_pos(3)+axisangle(3)+gripper(2)
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"][:2],
    ]).astype(np.float32)
    return {
        "observation.images.image": img(obs["agentview_image"]),
        "observation.images.image2": img(obs["robot0_eye_in_hand_image"]),
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": [instruction],
    }

policy.reset()
_raw = make_batch(obs, task.language)
print("状态向量 (eef+aa+gripper):", _raw["observation.state"][0].tolist())

# 诊断1: 归一化统计是否真的加载 (QUANTILES 需要 q01/q99)
_norm_step = pre.steps[2]
_st = getattr(_norm_step, "stats", None)
if _st:
    print("normalizer stats 特征:", list(_st.keys()))
    print("  state q01:", [round(float(x), 3) for x in _st["observation.state"]["q01"]])
else:
    print("!!! normalizer 没有加载任何 stats !!!")
_unn = post.steps[0]
_ust = getattr(_unn, "stats", None)
print("unnormalizer stats:", "有" if _ust else "!!! 没有 !!!")

batch = pre(_raw)
# 诊断2: prompt 应为 "Task: ..., State: 32个数;" + 换行 + "Action: "
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained("unsloth/gemma-7b-it")
_ids = batch.get("observation.language.tokens")
if _ids is not None:
    _txt = _tok.decode(_ids[0][:80])
    print("prompt 开头:", repr(_txt[:150]))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    a_raw = policy.select_action(batch)
print("归一化动作:", a_raw[0].float().cpu().tolist())
a = post({"action": a_raw})["action"]
print("反归一化动作:", a[0].float().cpu().tolist())
print("=== π0.5 + LIBERO 接口打通 ===")""")

code("""# 离线一致性测试: 用训练数据集的 demo 观测喂模型, 对比预测动作 vs demo 真实动作
# 相关性高 => 推理栈(权重/预处理/精度)正确, 问题在环境交互; 不相关 => 权重或精度问题
import pandas as pd, imageio.v2 as imageio, urllib.request

_BASE = "https://huggingface.co/datasets/lerobot/libero/resolve/main/"
urllib.request.urlretrieve(_BASE + "data/chunk-000/file-315.parquet", "/kaggle/working/d315.parquet")
urllib.request.urlretrieve(_BASE + "videos/observation.images.image/chunk-000/file-028.mp4", "/kaggle/working/v028.mp4")
urllib.request.urlretrieve(_BASE + "videos/observation.images.image2/chunk-000/file-028.mp4", "/kaggle/working/v028w.mp4")

_ep = pd.read_parquet("/kaggle/working/d315.parquet")
_ep = _ep[_ep.episode_index == 1304]           # task 30 demo, 113 帧
_states = np.stack(_ep["observation.state"].values).astype(np.float32)   # (T,8) 训练布局
_actions = np.stack(_ep["action"].values).astype(np.float32)             # (T,7)
_INSTR = "pick up the black bowl next to the cookie box and place it on the plate"
print("demo 帧:", len(_ep), "| 指令:", _INSTR)

_r = imageio.get_reader("/kaggle/working/v028.mp4")
_rw = imageio.get_reader("/kaggle/working/v028w.mp4")
_T0, _FPS = 319.2, 10

def _ds_frame(reader, t):
    return reader.get_data(int(round(t * _FPS)))          # 数据集帧已是正立约定, 不再翻转

policy.reset()
preds, truths = [], []
for t in range(0, 60, 6):
    policy.reset()   # 每步强制重新推理, 取动作块首动作与 demo 对齐
    o = _ds_frame(_r, _T0 + t); w = _ds_frame(_rw, _T0 + t)
    b = {
        "observation.images.image": torch.from_numpy(o.transpose(2,0,1).astype(np.float32)/255.0).unsqueeze(0),
        "observation.images.image2": torch.from_numpy(w.transpose(2,0,1).astype(np.float32)/255.0).unsqueeze(0),
        "observation.state": torch.from_numpy(_states[t]).unsqueeze(0),
        "task": [_INSTR],
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        a = policy.select_action(pre(b))
    a = post({"action": a})["action"][0].float().cpu().numpy()
    preds.append(a); truths.append(_actions[t])
    print("t=%2d 预测: %s" % (t, np.round(a, 3).tolist()))
    print("      真实: %s" % np.round(_actions[t], 3).tolist())

preds = np.array(preds); truths = np.array(truths)
_l2 = np.linalg.norm(preds - truths, axis=1)
print("每步 L2 误差:", np.round(_l2, 3).tolist())
print("平均 L2: %.3f | 真实动作幅度均值: %.3f" % (_l2.mean(), np.linalg.norm(truths, axis=1).mean()))""")

code("""# 跑 3 集完整任务 (libero_spatial task 0): 固定初始状态 + 开局 10 步 no-op + 录视频
import imageio, time

N_EP, MAX_STEPS, N_WAIT = 3, 280, 10
init_states = suite.get_task_init_states(0)
dummy = np.zeros(7, dtype=np.float64)
dummy[-1] = -1.0  # 官方 get_libero_dummy_action 同款
results = []
for ep in range(N_EP):
    env.reset()
    obs = env.set_init_state(init_states[ep % len(init_states)])
    policy.reset()
    frames, t0 = [], time.time()
    success = False
    for t in range(MAX_STEPS):
        if t < N_WAIT:
            action = dummy
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                a = policy.select_action(pre(make_batch(obs, task.language)))
            a = post({"action": a})["action"]  # 反归一化到真实动作尺度
            action = a.squeeze(0).float().cpu().numpy()
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1, ::-1].copy())
        if env.check_success():
            success = True
            break
    dt = time.time() - t0
    results.append(dict(ep=ep, steps=t + 1, success=success, seconds=round(dt, 1)))
    print("ep%d: success=%s steps=%d 用时%.0fs" % (ep, success, t + 1, dt))
    imageio.mimsave("/kaggle/working/pi05_libero_ep%d.mp4" % ep, frames, fps=20)

sr = sum(r["success"] for r in results) / len(results)
print("=" * 40)
print("成功率: %d/%d (%.0f%%)" % (sum(r["success"] for r in results), len(results), sr * 100))
json.dump(dict(task=task.language, suite="libero_spatial", task_id=0,
               model=REPO, max_steps=MAX_STEPS, protocol="init_state+10wait",
               results=results, success_rate=sr),
          open("/kaggle/working/pi05_libero_results.json", "w"), ensure_ascii=False, indent=2)
try:
    env.close()
except Exception:
    pass
print("=== 阶段 2 完成 ===")""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

out = r"C:\Users\蒋惊人\Downloads\pi05_kaggle\pi05_libero_sim.ipynb"
json.dump(nb, io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written", out, len(CELLS), "cells")
