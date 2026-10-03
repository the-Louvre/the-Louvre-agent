# Agent3 双模型 Ubuntu 服务器部署与迁移说明

本文用于把已经在教学服务器验证通过的 AIDE 与 IML-ViT Worker 迁移到小组 Ubuntu GPU 服务器。默认保持现有 HTTP 契约不变，因此迁移后 Agent3 主控只需更新两个 Worker URL。

## 1. 已验证架构

```text
Windows / Web 主控（FastAPI :8000）
├── AIDE Worker（Ubuntu :8101） -> 整图 real/fake 概率
└── IML-ViT Worker（Ubuntu :8102） -> 像素级概率 Mask
```

两个 Worker 必须使用隔离的 Python 环境。已验证硬件为 RTX 4090D 24GB；若两模型同时常驻，建议小组服务器至少提供 24GB 显存。磁盘还需容纳两个 Conda 环境、AIDE 约 3.4GB 权重、IML-ViT 约 351MB 权重和上游源码，建议预留至少 30GB。

## 2. 需要上传到服务器的文件

### 2.1 本项目 Worker 文件

从本仓库上传以下目录，不能只上传 `app.py`：

```text
product-agent-demo/model_workers/aide/
├── app.py
├── requirements.txt
└── start_worker.sh

product-agent-demo/model_workers/iml_vit/
├── app.py
├── requirements.txt
└── start_worker.sh
```

建议放到服务器：

```text
/home/<service-user>/agent3-worker/model_workers/aide/
/home/<service-user>/agent3-worker/model_workers/iml_vit/
```

### 2.2 上游源码

```text
AIDE：https://github.com/shilinyan99/AIDE
IML-ViT：https://github.com/SunnyHaze/IML-ViT
```

建议目录：

```text
/home/<service-user>/agent3-models/AIDE-main/
/home/<service-user>/agent3-models/IML-ViT-main/
```

上游源码不需要复制进本项目 Git 仓库。若服务器无法访问 GitHub，可在可信电脑下载官方 ZIP，核对来源后上传并解压。

### 2.3 模型权重

```text
AIDE：GenImage_train.pth（约 3.4GB，来自 AIDE 官方 Model Zoo）
IML-ViT：iml-vit_checkpoint.pth（约 351MB）
```

建议目录：

```text
/home/<service-user>/agent3-models/checkpoints/aide/GenImage_train.pth
/home/<service-user>/agent3-models/checkpoints/iml_vit/iml-vit_checkpoint.pth
```

已验证的 IML-ViT 权重 SHA-256：

```text
9da3d79187fc21841e2b4aa235364187aacb6de6ead6dfd9be2f77ffc278d52d
```

AIDE 权重迁移前应在原服务器执行 `sha256sum GenImage_train.pth`，把结果记录到小组的发布记录，并在新服务器再次核对。权重文件不得上传到 GitHub；使用前还应由小组确认上游权重许可与分发条件。

## 3. 推荐服务器目录

```text
/home/<service-user>/
├── agent3-worker/
│   └── model_workers/
│       ├── aide/{app.py,requirements.txt,start_worker.sh}
│       └── iml_vit/{app.py,requirements.txt,start_worker.sh}
└── agent3-models/
    ├── AIDE-main/
    ├── IML-ViT-main/
    └── checkpoints/
        ├── aide/GenImage_train.pth
        └── iml_vit/iml-vit_checkpoint.pth
```

## 4. 环境准备

先确认 NVIDIA 驱动和 GPU：

```bash
nvidia-smi
```

### 4.1 AIDE 环境

已验证组合：Python 3.10、PyTorch 2.0.1+cu118、torchvision 0.15.2+cu118、NumPy 1.24.4。

```bash
conda create -n aide python=3.10 -y
conda activate aide
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.24.4 "setuptools<81"
```

AIDE 官方 `requirements.txt` 内含会覆盖 GPU 环境的旧版 `torch`、`torchvision` 和 `numpy` 固定项。生成临时过滤清单，再安装其余依赖：

```bash
grep -Ev '^(torch|torchvision|numpy)==' /home/<service-user>/agent3-models/AIDE-main/requirements.txt > /tmp/aide-requirements-no-core.txt
python -m pip install -r /tmp/aide-requirements-no-core.txt
python -m pip install -r /home/<service-user>/agent3-worker/model_workers/aide/requirements.txt
```

验证：

```bash
python -c "import numpy,torch; print(numpy.__version__,torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

### 4.2 IML-ViT 环境

当前已验证环境为 Python 3.8、PyTorch 2.4.1+cu121，CUDA 可用。迁移时优先从当前可运行服务器导出环境清单，以减少上游依赖变化：

```bash
conda env export -n imlvit --no-builds > imlvit-environment.yml
```

在小组服务器创建环境后，安装 IML-ViT 上游依赖及 Worker 依赖：

```bash
conda env create -f imlvit-environment.yml
conda activate imlvit
python -m pip install -r /home/<service-user>/agent3-models/IML-ViT-main/requirements.txt
python -m pip install -r /home/<service-user>/agent3-worker/model_workers/iml_vit/requirements.txt
```

如果 CUDA/驱动版本与教学服务器不同，应按小组服务器驱动选择 PyTorch 官方 wheel，并重新执行严格权重加载和一次 GPU 前向测试，不能只看 `torch.cuda.is_available()`。

## 5. 需要修改的路径和接口配置

Worker 脚本已支持环境变量覆盖，迁移时无需改 Python 源码。设置如下变量：

| 变量 | 用途 | 小组服务器示例 |
|---|---|---|
| `AIDE_REPO` | AIDE 官方源码目录 | `/home/team/agent3-models/AIDE-main` |
| `AIDE_CHECKPOINT` | AIDE 权重 | `/home/team/agent3-models/checkpoints/aide/GenImage_train.pth` |
| `AIDE_CONDA_SH` | Conda 初始化脚本 | `/opt/conda/etc/profile.d/conda.sh` |
| `AIDE_CONDA_ENV` | Conda 环境名 | `aide` |
| `AIDE_PORT` | 内部监听端口 | `8101` |
| `IML_VIT_REPO` | IML-ViT 官方源码目录 | `/home/team/agent3-models/IML-ViT-main` |
| `IML_VIT_CHECKPOINT` | IML-ViT 权重 | `/home/team/agent3-models/checkpoints/iml_vit/iml-vit_checkpoint.pth` |
| `IML_VIT_CONDA_SH` | Conda 初始化脚本 | `/opt/conda/etc/profile.d/conda.sh` |
| `IML_VIT_CONDA_ENV` | Conda 环境名 | `imlvit` |
| `IML_VIT_PORT` | 内部监听端口 | `8102` |

启动示例：

```bash
AIDE_REPO=/home/team/agent3-models/AIDE-main \
AIDE_CHECKPOINT=/home/team/agent3-models/checkpoints/aide/GenImage_train.pth \
bash /home/team/agent3-worker/model_workers/aide/start_worker.sh
```

```bash
IML_VIT_REPO=/home/team/agent3-models/IML-ViT-main \
IML_VIT_CHECKPOINT=/home/team/agent3-models/checkpoints/iml_vit/iml-vit_checkpoint.pth \
bash /home/team/agent3-worker/model_workers/iml_vit/start_worker.sh
```

主控侧只改以下地址，不改 `/api/v1/agent3/*` 业务接口：

```text
AIDE_WORKER_URL=http://<group-server-host>:8101
IML_VIT_WORKER_URL=http://<group-server-host>:8102
AGENT3_MODEL_TIMEOUT_SECONDS=60
```

若通过反向代理或端口映射暴露，填写映射后的地址。AIDE 首次推理实测约 18 秒，双模型任务实测约 21 秒；共享服务器建议先使用 60 秒超时，再根据压测调整。

## 6. 必须保持的 Worker 接口

### 6.1 公共健康检查

```http
GET /health
```

返回：

```json
{"status":"ready","model_version":"...","device":"cuda"}
```

### 6.2 AIDE 推理

```http
POST /internal/v1/predict
Content-Type: multipart/form-data

image=<binary>
request_id=<string>
```

成功响应必须包含：

```json
{
  "status": "ok",
  "model_version": "aide-genimage-v1",
  "ai_probability": 0.0,
  "real_probability": 1.0,
  "latency_ms": 0
}
```

### 6.3 IML-ViT 推理

请求路径和 multipart 字段与 AIDE 相同。成功响应必须包含：

```json
{
  "status": "ok",
  "model_version": "iml-vit-v1",
  "mask_encoding": "png_gray16_base64",
  "mask_width": 1024,
  "mask_height": 1024,
  "probability_mask": "<base64>",
  "latency_ms": 0
}
```

若小组服务器改变路径、鉴权头或响应字段，必须同步修改：

```text
app/agent3/detectors/aide_client.py
app/agent3/detectors/iml_vit_client.py
```

优先保持上述契约不变；服务器迁移本身不应触发业务接口变更。

## 7. 端口与安全

- `8101`、`8102` 当前无应用层鉴权，只能放在小组内网、VPN 或受限安全组中。
- 禁止把 Worker 无限制公开到公网。
- 若必须跨公网访问，应在反向代理层增加 TLS、访问控制、请求体大小限制和速率限制。
- 防火墙只允许 Agent3 主控主机访问两个 Worker 端口。
- 用户上传图片和运行 Artifact 不应写入共享日志或长期保存。

## 8. 迁移验收顺序

1. 核对源码来源、权重文件名、大小和 SHA-256。
2. 分别验证两个 Conda 环境的 CUDA 张量运算。
3. 严格加载权重；IML-ViT 必须 `missing=0`、`unexpected=0`。
4. 启动 Worker，分别检查两个 `/health`。
5. 分别用单张图片调用两个 `/internal/v1/predict`。
6. 在主控设置新 `AIDE_WORKER_URL` 与 `IML_VIT_WORKER_URL`。
7. 确认 Agent3 `/health` 显示两个真实模型版本。
8. 用真实图、整图 AI 图、局部编辑图各跑至少一张，并确认无 `errors`。
9. 旧服务器保留到新服务器连续运行和回归测试通过后再下线。

## 9. 迁移后仍需完成的工作

迁移成功只代表工程链路可用，不代表检测效果已校准。准确率、阈值、融合语义、压缩退化和误报率应按 `agent3-issue-drafts.md` 中的阶段 E Issue 单独验收。
