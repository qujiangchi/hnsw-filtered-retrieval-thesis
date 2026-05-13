# AutoDL 部署指南

AutoDL 是国内性价比最高的 GPU 租赁平台，学生常用。

---

## 一、注册与开实例

1. 访问 https://www.autodl.com 注册账号
2. 充值（支持支付宝/微信，充 20-30元 足够）
3. 点击 **算力市场** → 选择地区（推荐 **西北B区/内蒙A区**，便宜且资源多）
4. **筛选条件**：
   - GPU: RTX 4090 或 RTX 3090
   - 内存: ≥128GB (如果选4090跑完整MS MARCO)
   - 或内存≥64GB (如果选3090跑50% MS MARCO)
5. **镜像选择**：
   - **PyTorch 2.x / Python 3.10 / CUDA 12.x**（推荐）
   - 不要选 TensorFlow 镜像
6. 点击 **立即创建**

---

## 二、连接实例

### 方式1：SSH（推荐）

创建实例后，在控制台找到 **SSH指令**，类似：
```bash
ssh -p 12345 root@connect.westb.seetacloud.com
# 密码: xxxxxxxxxxxx
```

### 方式2：JupyterLab

点击 **JupyterLab** 进入 Web 终端，但操作不如 SSH 方便。

---

## 三、一键部署（2条命令）

SSH 连接后：

```bash
# 1. 把项目clone到数据盘（数据盘空间大，系统盘可能不够）
cd /root/autodl-tmp
git clone https://github.com/YOUR_USERNAME/filtered-vector-lab.git
cd filtered-vector-lab

# 2. 运行AutoDL专用部署脚本
bash scripts/autodl_setup.sh
```

然后等待完成（约 2-4 小时）。

---

## 四、AutoDL 环境特点

| 项目 | 说明 |
|------|------|
| 数据盘 | `/root/autodl-tmp/` 空间大（通常100GB+），代码和数据放这里 |
| 系统盘 | `/` 较小（约30GB），不要在这里存大数据 |
| 预装环境 | 已有 PyTorch、CUDA、conda，不需要重装 |
| 网络 | 国内线路，访问 Hugging Face 可能需要镜像 |
| 关机 | 点"关机"后数据保留，不计费；"释放"后数据清空 |
| 网盘 | `/root/autodl-pub/` 是网盘，可跨实例共享文件 |

---

## 五、手动分步部署（如果一键脚本失败）

### Step 1: 配置 Hugging Face 镜像加速

```bash
# 在 ~/.bashrc 中添加
export HF_ENDPOINT=https://hf-mirror.com
source ~/.bashrc
```

### Step 2: 安装缺失依赖

```bash
# AutoDL 已有 PyTorch 和 CUDA，只需装实验相关包
pip install datasets sentence-transformers faiss-gpu pandas scipy tqdm -q
```

### Step 3: 下载并编码数据

```bash
cd /root/autodl-tmp/filtered-vector-lab
python3 scripts/download_real_datasets.py \
  --msmarco-ratio 0.5 \
  --nq-ratio 0.2 \
  --enron-ratio 1.0 \
  --max-queries 5000
```

### Step 4: 运行实验

```bash
python3 scripts/run_real_experiments.py
```

---

## 六、省钱技巧

1. **用关机代替释放**：实验跑完先点"关机"，数据保留，下次开机继续。只有"开机"状态才计费。
2. **不要选北京/上海节点**：价格贵 30-50%，选西北/内蒙节点。
3. **3090 够用了**：如果跑 50% MS MARCO + 20% NQ，64GB内存 + 3090 足够，1.5元/小时。
4. **编码和实验分开**：如果 4090 被抢光了，可以先租 3090 编码，再租 4090 跑实验（数据放网盘共享）。

---

## 七、常见问题

### Q1: Hugging Face 下载很慢或失败

AutoDL 国内访问 Hugging Face 可能不稳定。脚本已自动配置 `hf-mirror.com` 镜像。如果仍失败：
```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
python3 scripts/download_real_datasets.py
```

### Q2: 显存不足 (CUDA out of memory)

编码时 batch_size 太大：
```bash
python3 scripts/download_real_datasets.py --batch-size 64
```

### Q3: 数据盘满了

检查空间：`df -h`
如果 `/root/autodl-tmp` 满了，可以删旧数据或扩容。

### Q4: 实例被抢占（共享实例）

如果选的是"共享GPU"实例，可能被强制关机。**建议选独占实例**（标注"独占"的）。

---

## 八、实验完成后

```bash
# 1. 保存结果到网盘（跨实例持久化）
cp -r results /root/autodl-pub/filtered-vector-lab-results/

# 2. 或下载到本地
# 在本地终端执行：
# scp -P 12345 -r root@connect.westb.seetacloud.com:/root/autodl-tmp/filtered-vector-lab/results ./

# 3. 关机（保留数据，停止计费）
# 在AutoDL控制台点击 "关机"
```

---

*最后更新: 2026-05-13*
