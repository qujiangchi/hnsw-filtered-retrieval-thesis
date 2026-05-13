# 云服务器租用指南（CloudLab不可用时的替代方案）

CloudLab资源紧张抢不到节点时，以下是国内/国外最实惠的替代方案。

---

## 一、成本总览（6小时实验）

| 方案 | 配置 | 6h成本 | 可跑规模 | 推荐度 |
|------|------|--------|---------|--------|
| **阿里云抢占式** | 16核128GB | **~7元** | 完整MS MARCO + 68% NQ | ⭐⭐⭐⭐⭐ |
| 阿里云按量 | 16核128GB | ~27元 | 同上 | ⭐⭐⭐⭐ |
| 腾讯云按量 | 16核128GB | ~25元 | 同上 | ⭐⭐⭐⭐ |
| AWS学生机 | 16核128GB | **$0**（有额度） | 同上 | ⭐⭐⭐⭐⭐ |
| GCP学生机 | 16核128GB | **$0**（有额度） | 同上 | ⭐⭐⭐⭐⭐ |

> 结论：**优先阿里云抢占式**（7元搞定）；有.edu邮箱的**优先AWS/GCP学生免费额度**。

---

## 二、方案A：阿里云抢占式（最省钱，7元搞定）

### 1. 创建实例

1. 登录 https://ecs.console.aliyun.com/
2. 点击 **创建实例**
3. 配置选择：
   - **付费方式**：抢占式实例
   - **地域**：任意（建议选离你近的，如华北2/华东1）
   - **实例规格**：`内存型 r7` → `ecs.r7.4xlarge`（16核128GB）
   - **镜像**：**Ubuntu 22.04 64位**
   - **系统盘**：ESSD 100GB（默认40GB可能不够）
   - **带宽**：按流量 100Mbps（下载数据用，实验时几乎不耗流量）
   - **安全组**：放行 **SSH(22)** 端口
4. 点击 **确认下单**

> 抢占式实例价格约1.2元/小时，但可能被回收。6小时实验被回收概率极低。如果被回收，数据会保留在云盘，换实例重新挂载即可。

### 2. SSH连接并部署

```bash
ssh root@YOUR_EIP

# 以下命令在服务器上执行
apt update && apt install -y git python3-pip python3-venv

git clone https://github.com/YOUR_USERNAME/filtered-vector-lab.git
cd filtered-vector-lab

# 一键部署
bash scripts/cloudlab_setup.sh
```

### 3. 实验完成后释放实例（重要！）

```bash
# 先复制结果回本地
scp -r root@YOUR_EIP:~/filtered-vector-lab/results ./

# 然后在阿里云控制台 → 实例 → 释放
# 否则会持续计费！
```

---

## 三、方案B：腾讯云按量付费（25元，最稳定）

### 1. 创建实例

1. 登录 https://console.cloud.tencent.com/cvm
2. 点击 **新建**
3. 配置：
   - **计费模式**：按量计费
   - **地域**：任意
   - **机型**：标准型 S5 → **4XLARGE32**（16核128GB）
   - **镜像**：Ubuntu 22.04 LTS
   - **系统盘**：云硬盘 100GB
   - **带宽**：按流量计费 100Mbps
   - **安全组**：放行22端口
4. 点击 **立即购买**

### 2. 部署命令

与阿里云完全一致，参考方案A第2步。

---

## 四、方案C：AWS 学生免费额度（$0，需.edu邮箱）

### 1. 申请 AWS Educate

1. 访问 https://aws.amazon.com/education/awseducate/
2. 用 `.edu` 邮箱注册，通常可获得 **$100-150** 免费额度

### 2. 创建实例

1. 登录 AWS Console → EC2 → Launch Instance
2. 配置：
   - **Name**: ann-experiment
   - **AMI**: Ubuntu 22.04 LTS
   - **Instance type**: `r5.4xlarge`（16核128GB）
   - **Key pair**: 创建或选择已有
   - **Security group**: 允许 SSH (22)
   - **Storage**: 100GB gp3
3. Launch

### 3. 部署

```bash
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
# 后续与阿里云完全一致
```

### 4. 完成后 Stop 实例（不要Terminate，保留数据）

Stop后不计费，下次Start继续用。

---

## 五、方案D：Google Cloud 学生额度（$300，需.edu）

1. 访问 https://cloud.google.com/edu 申请 $300 额度
2. 创建 VM：
   - Machine type: `n2-highmem-16`（16核128GB）
   - Boot disk: Ubuntu 22.04, 100GB
3. 部署同上

---

## 六、方案E：恒创科技/UCloud（国内小众，更便宜）

如果上述都觉得贵：

- **恒创科技** https://www.henghost.com/ 经常有首单优惠
- **UCloud** https://www.ucloud.cn/ 学生机有时有大内存促销
- **Vultr** https://www.vultr.com/ 海外节点，按小时计费，但大内存实例较少

---

## 七、抢占式实例防回收技巧

如果被回收，数据不会丢失（云盘保留），但需要重新开实例、挂载云盘。为避免麻烦：

```bash
# 在实验过程中，每隔30分钟自动保存结果到本地/对象存储
while true; do
  scp results/normalized/all_thesis_experiments.csv user@backup_host:~/backup/
  sleep 1800
done &
```

或者直接用 **nohup + 后台** 让实验在断开SSH后继续跑：

```bash
nohup bash scripts/cloudlab_setup.sh > setup.log 2>&1 &
# 然后可以断开SSH，过几小时再连回来检查
tail -f setup.log
```

---

## 八、一键部署脚本（通用，任何Ubuntu云主机都能跑）

脚本 `scripts/cloudlab_setup.sh` 在任意Ubuntu 22.04云主机上都能直接运行，已包含：
- 安装git/python/venv
- pip install 所有依赖
- 下载并编码三个数据集
- 运行全部实验
- 生成图表

只需要确保：**实例内存≥64GB、磁盘≥100GB、有外网访问**。

---

## 九、论文中如何写实验环境

```
实验在阿里云ECS实例上进行，配置为Intel Xeon Platinum处理器（16核），
128GB DDR4内存，100GB ESSD云盘，运行Ubuntu 22.04 LTS。
由于完整NQ数据集索引需约150GB内存，对NQ采用68%随机采样，
MS MARCO和Enron采用完整集合。所有算法在相同硬件条件下横向对比。
```

---

*最后更新: 2026-05-13*
