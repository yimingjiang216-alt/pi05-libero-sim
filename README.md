# pi05-libero-sim — 在 Kaggle 免费 T4 上跑通 π0.5 驱动 LIBERO 机械臂

开源 π0.5（VLA 模型）在 LIBERO 基准任务上闭环驱动机器人：**libero_spatial task 0 三集全部成功（3/3，87/88/90 步，每集约 40 秒）**，单卡 T4（16GB，免费配额），无需本地 GPU。

任务指令：*pick up the black bowl between the plate and the ramekin and place it on the plate*

## 结果

| 集 | 结果 | 步数（上限 280） | 用时 |
|---|---|---|---|
| ep0 | ✅ 成功 | 87 | 42s |
| ep1 | ✅ 成功 | 88 | 35s |
| ep2 | ✅ 成功 | 90 | 36s |

成功视频见 [`results/`](results/)（`pi05_libero_ep0.mp4` 等，agentview 相机视角），原始指标在 [`results/pi05_libero_results.json`](results/pi05_libero_results.json)。

## 系统架构

```
π0.5 (VLA, 3B)                       LIBERO / robosuite / MuJoCo
┌─────────────────────┐   动作(7维)  ┌──────────────────────────┐
│ 双相机 256×256 图像  │ ──────────▶ │ OSC 控制器 → Panda 机械臂  │
│ 8 维本体状态         │             │ 20Hz 物理仿真             │
│ 自然语言任务指令     │ ◀────────── │ agentview + eye-in-hand   │
└─────────────────────┘   观测       └──────────────────────────┘
```

- **权重**：[bigbangoslab/pi05_libero_openpiconfig_on_lerobot_chunk30](https://huggingface.co/bigbangoslab/pi05_libero_openpiconfig_on_lerobot_chunk30)（lerobot/libero 数据集 3 万步微调，bf16，7.5GB；作者自评 spatial 94.4%）
- **环境**：hf-libero（robosuite 1.4.0 + MuJoCo 3.x + bddl），EGL 离屏渲染
- **推理**：LeRobot PyTorch PI05Policy，fp16（T4 无 bf16），官方 `from_pretrained` 加载
- **评测协议**：固定任务初始状态 + 开局 10 步 no-op 等待 + 官方预处理管线（状态 8→32 维补零后离散化为 256-bin token 拼进 prompt）+ 动作分位数反归一化

## 调试历程：从 0/3 到 3/3

比跑通更有价值的是排错过程，每一步都用数据定位：

**第 1 步：通用底座跑不出任务。** 先用 `lerobot/pi05_libero_base`（名字带 libero 但实际是未微调底座，仅 I/O 形状按 LIBERO 配置），0/3。换真正微调过的社区权重。

**第 2 步：换权重后仍 0/3 → 离线对拍定位。** 闭环失败的原因可能在模型侧，也可能在环境交互侧，盲跑无法分辨。于是做离线一致性测试：从训练数据集（lerobot/libero）取 demo 的观测喂模型，对比模型预测动作与 demo 记录的真实动作——**平均 L2 误差 1.58 vs 动作幅度 1.30，基本不相关**，证明问题在模型执行侧而非环境侧。

**第 3 步：二分排查模型侧。** 预处理链路逐项核对（状态布局、归一化统计加载、prompt 格式）全部正确后，锁定权重加载环节：我此前为 14.5GB 的 fp32 底座权重手写的 meta 设备流式加载器，对 bf16 检查点会**静默错位**（buffer 零填充 / tied embedding 恢复策略不适配）。bf16 权重只有 7.5GB，T4 直接放得下——删掉整套自制加载器，改用官方 `from_pretrained` + fp16，立即恢复。

**附带修复**：LeRobot 0.6.1 的状态离散化步骤丢了 0.4.1（训练版本）的 8→32 维补零行为，用 monkeypatch 打回，保证 prompt 里状态 token 数量与训练一致。

## 复现

```bash
# 1. 生成本地 notebook（或直接使用仓库里的 pi05_libero_sim.ipynb）
python build_stage2.py

# 2. 推到 Kaggle 跑（需要 Kaggle 账号 + API token，T4 GPU + 联网）
kaggle kernels push -p .
```

全程约 25 分钟：安装 ~8min（lerobot + hf-libero）、下载权重 ~2min、离线一致性测试 ~2min、3 集 rollout ~5min。

## 文件结构

```
├── build_stage2.py           # notebook 构建脚本（所有实验逻辑都在这里）
├── pi05_libero_sim.ipynb     # 生成的 Kaggle notebook（v49，即 3/3 版本）
└── results/
    ├── pi05_libero_results.json
    └── pi05_libero_ep{0,1,2}.mp4
```

## 相关项目

- [mini-vla-pi0](https://github.com/yimingjiang216-alt/mini-vla-pi0)：π0 系 policy 侧的迷你复现（自训 3B 迷你 VLA + 3D 导航任务）
- [rgbd-slam](https://github.com/yimingjiang216-alt/rgbd-slam)：RGB-D SLAM 回环失效的量化归因与修复
