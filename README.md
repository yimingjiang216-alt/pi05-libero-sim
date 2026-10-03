# pi05-libero-sim — 开源 π0.5 驱动 LIBERO 机械臂仿真实验

在 Kaggle 免费 T4（16GB）上，用开源 π0.5 视觉-语言-动作模型（LIBERO 微调权重）闭环驱动 LIBERO 基准的 Panda 机械臂，完成桌面取物任务。libero_spatial 第 0 号任务 3 次运行全部成功。

任务指令：*pick up the black bowl between the plate and the ramekin and place it on the plate*

## 一、全景：三个组件各管一件事

整个实验链由三层组成，分工是：**π0.5 出动作，LIBERO 出题判卷，MuJoCo 算物理**。

| 组件 | 角色 | 管什么 |
|---|---|---|
| **π0.5**（模型） | 决策者 | 看两路相机画面 + 机器人状态 + 一句指令，输出 7 维动作 |
| **LIBERO**（benchmark） | 出题、判卷 | 130 道任务定义"摆什么、摆在哪、怎样算成功"，每一步判一次 |
| **robosuite**（机器人框架） | 执行层 | Panda 机械臂、OSC 控制器（动作 → 关节力矩）、两台虚拟相机 |
| **MuJoCo**（物理引擎） | 考场 | 每 2ms 一次碰撞 → 受力 → 积分，并渲染相机画面 |

模型这层是可替换的：换成别的 VLA、甚至写死的脚本，只要接口不变（吃观测、吐 7 维动作），链路照常工作。

## 二、一轮决策循环的数据链

控制频率 20Hz（每 50ms 一步）。每一歩发生的事，按顺序：

```
① 取观测
   agentview 图 256×256 + 腕部相机 256×256   (LIBERO 渲染上下颠倒 → 双向翻转 180°)
   状态 8 维 = 末端位置(3) + 姿态轴角(3) + 夹爪开度(2)
   指令 = "pick up the black bowl between the plate and the ramekin ..."

② 状态进 prompt（π0.5 的特点：状态不走连续向量，走文本）
   8 维状态 --分位数归一化到[-1,1]--> --线性离散化 256 档--> --补零到 32 维-->
   拼进文本: "Task: <指令>, State: <32 个档位数>;" --> Gemma 分词器

③ π0.5 前向
   两张图 + prompt ──▶ PaliGemma 主干(3B) ──▶ 动作专家(300M) 跨注意力
   ──flow matching 从噪声积分 10 步──▶ 动作块 30 个，取前 10 个排队执行
   ──分位数反归一化──▶ 真实动作尺度

④ 执行
   7 维动作(末端位姿增量 6 + 夹爪 1) ──▶ robosuite OSC 控制器
   ──▶ 关节力矩 ──▶ MuJoCo 物理(碰撞/受力/积分) ──▶ 相机重新渲染

⑤ 判卷
   LIBERO 检查任务成功条件 (check_success)
   成功 → 停；未成功 → 回到 ①，最多 280 步
```

前 10 步发零动作让场景沉降（官方评测协议），动作块机制意味着模型每推理一次管 10 步，推理不是每步都发生的。

### 与官方 openpi 方案的差别

官方跑法是**模型和仿真分成两个进程**：服务端 openpi（JAX，权重 12.4GB，RTX 3090 级）常驻，客户端跑 LIBERO，中间用本机 websocket 传"两张图 + 状态 → 10 个动作"。

本实验在 Kaggle 单个 notebook 里把两件事放进一个进程：LeRobot 的 PyTorch 版 π0.5 以 fp16 加载（7.5GB 权重，T4 16GB 直接容纳），推理结果在进程内直接喂给 robosuite，不需要 websocket。观测/动作的语义与官方一致，差异只在工程形态：

| | 官方 openpi 方案 | 本实验 |
|---|---|---|
| 模型实现 | openpi（JAX） | LeRobot（PyTorch） |
| 权重 | pi05_libero 官方微调版 12.4GB | 社区同数据微调版 7.5GB（bf16→fp16） |
| 硬件 | RTX 3090 | Kaggle 免费 T4 |
| 进程/通信 | 双进程 + websocket | 单进程直连 |
| 成功率 | 作者自评 spatial 94.4% | 任务 #0 实测 3/3 |

## 三、实验环境

| 项 | 配置 |
|---|---|
| 算力 | Kaggle Notebook，Tesla T4 16GB，单卡 |
| 权重 | [bigbangoslab/pi05_libero_openpiconfig_on_lerobot_chunk30](https://huggingface.co/bigbangoslab/pi05_libero_openpiconfig_on_lerobot_chunk30)（lerobot/libero 数据集微调 30000 步，batch 128，bf16，7.5GB；作者自评 libero_spatial 成功率 94.4%） |
| 仿真 | hf-libero 0.1.4（robosuite 1.4.0 + MuJoCo 3.x + bddl 1.0.1），EGL 离屏渲染 |
| 模型库 | LeRobot 0.6.1 + transformers 5.5，fp16 推理（T4 无 bf16 计算） |
| 分词器 | unsloth/gemma-7b-it（与 PaliGemma 同为 256k 词表，绕过官方 tokenizer 的访问限制） |

## 四、实验过程

### 步骤 1：搭仿真环境

pip 安装 hf-libero（自带全部任务的 BDDL 与初始状态文件），预写 `~/.libero/config.yaml` 跳过首次运行的交互式路径配置，创建 `OffScreenRenderEnv`（相机 256×256，EGL），reset 一次核对观测键名与维度。

### 步骤 2：加载模型与官方预处理管线

`PI05Policy.from_pretrained()` 加载 bf16 权重，转 fp16 上 T4（显存 8.3GB）。预/后处理管线用权重仓库自带的 json 构造，分词器名替换为同词表 Gemma 镜像。两处对齐训练版本（lerobot 0.4.1）的适配：

- 0.6.1 的状态离散化步骤不再把 8 维状态补零到 32 维，而训练时是补零后离散化的——monkeypatch 恢复该行为，保证 prompt 中状态 token 数量一致（8 个真实值 + 24 个零对应的中间档位 128）；
- 动作输出必须经后处理管线的分位数反归一化才是真实动作尺度。

### 步骤 3：构造观测（对齐训练分布）

- 图像：按官方预处理约定双向翻转 180°，转 CHW、除 255（VISUAL 归一化模式为 IDENTITY，无需再变换）；
- 状态：`[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]`——与训练数据 stats 逐维核对过（第 3 维 0.35~3.67 为朝下夹爪的轴角主分量，第 6/7 维是 ±0.04 的镜像手指对）。

### 步骤 4：离线校验

闭环运行前，从训练数据集取一条演示轨迹，把其中若干帧的观测喂给模型，比较输出动作与该帧记录的真实动作的数量级与分布。flow matching 对噪声初值敏感，单步动作本就不与演示逐点重合，该校验用于确认管线正确性（归一化统计已加载、prompt 拼装格式正确），不作为精度指标。

### 步骤 5：闭环运行

每集：`env.reset()` → 载入固定初始状态 `env.set_init_state()` → 前 10 步零动作沉降 → 之后逐步走"取观测 → 预处理 → 推理 → 反归一化 → 执行"，每步 `env.check_success()` 判定，命中即停。

## 五、结果

| 集 | 结果 | 步数（上限 280） | 用时 |
|---|---|---|---|
| ep0 | 成功 | 87 | 42s |
| ep1 | 成功 | 88 | 35s |
| ep2 | 成功 | 90 | 36s |

任务对应演示数据的平均长度约 113 步（10Hz 记录），模型在 20Hz 执行下用 87~90 步完成，节奏合理。成功过程视频见 [`results/`](results/)（agentview 视角），原始指标在 [`results/pi05_libero_results.json`](results/pi05_libero_results.json)。

局限：样本量仅 3 集（同任务不同初始状态），不足以估计该任务的真实成功率；未覆盖其余 129 个任务；fp16 与训练所用 bf16 存在数值差异，未单独评估其影响。

## 六、方法与出处

| 组件 | 出处 |
|---|---|
| π0 模型 | Physical Intelligence, arXiv:2410.24164；官方实现 Physical-Intelligence/openpi |
| π0.5 模型 | Physical Intelligence, arXiv:2504.16054 |
| LIBERO 基准 | Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.03310 |
| robosuite | Zhu et al., arXiv:2009.12293；OSC 控制器源自 Khatib 1987 操作空间公式 |
| MuJoCo | Todorov et al., *MuJoCo: A Physics Engine for Model-Based Control*, IROS 2012 |
| LeRobot PyTorch 实现 | github.com/huggingface/lerobot |
| PaliGemma 主干 | Beyer et al., *PaliGemma: A Versatile 3B VLM for Transfer*, arXiv:2407.07726 |

## 七、复现

```bash
# 生成本地 notebook（或直接使用仓库里的 pi05_libero_sim.ipynb）
python build_stage2.py
# 推到 Kaggle 运行（需 Kaggle 账号 + API token，T4 GPU + 联网）
kaggle kernels push -p .
```

单次运行约 25 分钟：安装约 8 分钟、权重下载约 2 分钟、离线校验约 2 分钟、3 集 rollout 约 5 分钟。

## 八、文件结构

```
├── build_stage2.py           # notebook 构建脚本（全部实验逻辑）
├── pi05_libero_sim.ipynb     # 生成的 Kaggle notebook
└── results/
    ├── pi05_libero_results.json
    └── pi05_libero_ep{0,1,2}.mp4
```

## 相关仓库

- [mini-vla-pi0](https://github.com/yimingjiang216-alt/mini-vla-pi0)：π0 架构的从零迷你实现（2.38M 参数，CPU 训练）——本仓库是同一架构在真实开源权重、真实操作基准上的验证
- [vision-worldmodel-projects](https://github.com/yimingjiang216-alt/vision-worldmodel-projects)：动作条件视频世界模型
