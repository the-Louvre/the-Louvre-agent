# Agent3 第一版实现指导书

版本：`1.0.0`
状态：开发基线（接口冻结）
适用项目：`product-agent-demo`

## 1. 规范说明

本文中的“必须”是第一版验收要求，“禁止”表示不得实现，“应该”表示原则上需要实现。若修改冻结字段，必须先同步修改本文档、契约测试和调用方。

Agent3 第一版必须：

1. 使用 AIDE 判断整图 AIGC 概率。
2. 使用 IML-ViT 生成像素级疑似篡改概率图。
3. 生成二值 Mask、区域框和叠加图。
4. 通过固定规则融合两个模型。
5. 输出结构化 `VisualReviewResult`。
6. 将模型结论标记为 L6 辅助证据。
7. 支持单模型失败降级，并保留模型、阈值和输入版本信息。

第一版禁止：

- 宣称图片已被“确定”由 AI 生成。
- 将 IML-ViT 的篡改区域直接称为 AI 生成区域。
- 使用评论推断图片真伪或判断产品功效。
- 覆盖 Agent1 的产品身份结果。
- 依赖按次收费的外部检测 API。
- 在模型未提供证据时编造包装、光影或皮肤异常。

## 2. 模型选型

| 模型 | 固定职责 | 输出 | 业务字段 |
|---|---|---|---|
| AIDE | 整图 AIGC 检测 | real/fake 概率 | `ai_probability` |
| IML-ViT | 通用图片篡改定位 | 像素概率图 | `tamper_probability`、Mask、区域 |

固定版本标识：

```text
aide-v1
iml-vit-v1
fusion-rules-v1
```

两个项目均采用 MIT License。模型权重必须在部署阶段下载到本地；正常推理不得把用户图片发送至第三方。

## 3. 运行架构

```text
主控 Agent / 浏览器
    -> Agent3 API（现有 FastAPI，默认 8000）
         -> AIDE Worker（本地内部服务，默认 8101）
         -> IML-ViT Worker（本地内部服务，默认 8102）
    -> 后处理 -> 规则融合 -> Artifact -> VisualReviewResult
```

两个模型必须使用隔离的运行环境。AIDE 官方参考环境为 Python 3.10、PyTorch 2.0.1、CUDA 11.8；IML-ViT 官方参考环境为 Python 3.8、PyTorch 1.11、CUDA 11.7。两套模型依赖禁止直接加入 Web 应用的同一个 `requirements.txt`。

## 4. 建议目录

```text
product-agent-demo/
├── app/
│   ├── main.py
│   └── agent3/
│       ├── __init__.py
│       ├── router.py
│       ├── schemas.py
│       ├── service.py
│       ├── repository.py
│       ├── fusion.py
│       ├── postprocess.py
│       ├── artifacts.py
│       ├── config.py
│       └── detectors/
│           ├── base.py
│           ├── aide_client.py
│           └── iml_vit_client.py
├── model_workers/
│   ├── aide/{app.py,adapter.py,requirements.txt}
│   └── iml_vit/{app.py,adapter.py,requirements.txt}
├── models/{aide,iml-vit}/
├── runtime/agent3/
└── tests/
    ├── test_agent3_api.py
    ├── test_agent3_schemas.py
    ├── test_agent3_fusion.py
    ├── test_agent3_postprocess.py
    └── test_agent3_degradation.py
```

`models/`、`runtime/agent3/` 必须进入 `.gitignore`。第三方仓库不得复制到 `app/` 中。

## 5. 外部 API 总则

接口前缀固定为 `/api/v1/agent3`。JSON 使用 UTF-8；时间使用 UTC ISO 8601，例如 `2026-09-22T08:30:15.123Z`。所有概率为 `[0,1]` 的 JSON number，禁止返回百分数字符串、NaN 或 Infinity。

### 5.1 创建任务

```http
POST /api/v1/agent3/reviews
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 限制 |
|---|---|---:|---|
| `image` | binary | 是 | PNG/JPEG/WebP；1 B～10 MiB |
| `task_id` | string | 是 | `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` |
| `product_context` | JSON string | 否 | 最大 16 KiB；见 6.2 |

校验要求：

- 必须使用文件头验证真实格式，不能只信任扩展名或 MIME。
- 解码后宽高分别为 `64～12000` 像素，总像素不超过 `40,000,000`。
- 动态图片只处理第一帧，并写入 `uncertainties`。
- 同一 `task_id` 可以创建多个任务，每次生成新的 `review_id`。

成功返回 `202 Accepted`：

```json
{
  "review_id": "vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC",
  "task_id": "task_001",
  "status": "queued",
  "created_at": "2026-09-22T08:30:15.123Z",
  "status_url": "/api/v1/agent3/reviews/vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC"
}
```

### 5.2 查询任务

```http
GET /api/v1/agent3/reviews/{review_id}
```

`review_id` 必须符合 `^vr_[A-Za-z0-9]{20,32}$`。任务存在统一返回 `200`；不存在返回 `404`。

处理中响应：

```json
{
  "review_id": "vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC",
  "task_id": "task_001",
  "status": "running",
  "stage": "model_inference",
  "progress": 0.55,
  "created_at": "2026-09-22T08:30:15.123Z",
  "updated_at": "2026-09-22T08:30:18.456Z"
}
```

完成后必须返回第 6.3 节的 `VisualReviewResult`。

### 5.3 获取 Artifact

```http
GET /api/v1/agent3/reviews/{review_id}/artifacts/{artifact_name}
```

`artifact_name` 白名单：

```text
probability-mask.png
binary-mask.png
overlay.png
```

成功返回 `200 image/png` 并设置 `Cache-Control: private, no-store`。不存在或非白名单名称返回 `404`。禁止把 `artifact_name` 当作任意路径使用。

### 5.4 健康检查

```http
GET /api/v1/agent3/health
```

```json
{
  "status": "ready",
  "models": {
    "aide": {"status": "ready", "model_version": "aide-v1", "latency_ms": 12},
    "iml_vit": {"status": "ready", "model_version": "iml-vit-v1", "latency_ms": 15}
  },
  "fusion_version": "fusion-rules-v1"
}
```

顶层 `status` 只能是：`ready`（均可用）、`degraded`（一个可用）、`unavailable`（均不可用）。

## 6. 严格数据契约

### 6.1 枚举

```text
ReviewStatus = queued | running | completed | degraded | failed
ReviewStage = validating | queued | model_inference | postprocessing |
              fusion | storing_artifacts | completed
SourceType = likely_ai_generated | suspected_ai_assisted_edit |
             edited_photo | no_anomaly_detected | inconclusive
RiskLevel = high | medium | low | unknown
ModelStatus = ok | error | timeout | unavailable
RegionType = suspected_manipulation
EvidenceLevel = L6
```

### 6.2 ProductContext

```json
{
  "product_id": "P001",
  "brand": "理肤泉",
  "product_name": "B5多效修复霜",
  "confidence": 0.91,
  "reference_image_urls": []
}
```

`product_id`、`brand` 最大 128 字符，`product_name` 最大 256 字符，`confidence` 为 `[0,1]`。允许未知字段，以兼容 Agent1 扩展。第一版必须忽略 `reference_image_urls`，禁止主动下载远程 URL，避免 SSRF。

### 6.3 VisualReviewResult

```json
{
  "schema_version": "1.0.0",
  "review_id": "vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC",
  "task_id": "task_001",
  "status": "completed",
  "stage": "completed",
  "source_type": "suspected_ai_assisted_edit",
  "risk": "high",
  "confidence": 0.81,
  "input": {
    "sha256": "8f14e45fceea167a5a36dedd4bea2543b28f9f2d3dce5e7f3f5f68f8f4bfa123",
    "mime_type": "image/png",
    "width": 1080,
    "height": 1440,
    "animated": false
  },
  "model_results": {
    "aide": {
      "status": "ok",
      "ai_probability": 0.86,
      "real_probability": 0.14,
      "model_version": "aide-v1",
      "latency_ms": 423,
      "error": null
    },
    "iml_vit": {
      "status": "ok",
      "tamper_probability": 0.73,
      "mask_area_ratio": 0.18,
      "model_version": "iml-vit-v1",
      "latency_ms": 618,
      "error": null
    }
  },
  "regions": [{
    "region_id": "region_001",
    "type": "suspected_manipulation",
    "bbox": [120, 220, 360, 480],
    "normalized_bbox": [0.111111, 0.152778, 0.333333, 0.333333],
    "score": 0.83,
    "area_ratio": 0.12,
    "reason_code": "high_tamper_probability"
  }],
  "artifacts": {
    "probability_mask_url": "/api/v1/agent3/reviews/vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC/artifacts/probability-mask.png",
    "binary_mask_url": "/api/v1/agent3/reviews/vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC/artifacts/binary-mask.png",
    "overlay_url": "/api/v1/agent3/reviews/vr_01J8E2C4Y2N7QKJ5R5T3P9A8BC/artifacts/overlay.png"
  },
  "evidence": [{
    "evidence_id": "ev_visual_001",
    "level": "L6",
    "source": "AIDE",
    "finding_code": "global_aigc_probability_high",
    "finding": "整图AIGC概率较高",
    "confidence": 0.86
  }],
  "uncertainties": [
    "篡改区域不等同于确定的AI生成区域",
    "截图、压缩和二次编辑可能影响检测结果"
  ],
  "errors": [],
  "fusion_version": "fusion-rules-v1",
  "created_at": "2026-09-22T08:30:15.123Z",
  "completed_at": "2026-09-22T08:30:20.123Z",
  "duration_ms": 5000
}
```

### 6.4 数据不变量

1. `schema_version` 第一版固定为 `1.0.0`。
2. `completed/degraded` 必须具有 `source_type`、`risk`、`confidence`。
3. `failed` 必须为 `source_type=inconclusive`、`risk=unknown`。
4. 所有概率、区域 score 和 area ratio 必须在 `[0,1]`。
5. `real_probability + ai_probability` 误差不得超过 `1e-4`。
6. `bbox` 固定为原图像素整数 `[x,y,width,height]`，不得越界。
7. `normalized_bbox` 同格式，分别除以宽高并保留六位小数。
8. `regions` 按 score 降序，最多 20 个；无区域必须返回 `[]`。
9. 只有成功生成的 Artifact 才返回 URL，缺失项为 `null`。
10. 模型非 `ok` 时概率必须为 `null`，同时给出脱敏 error。
11. 禁止在响应中返回 API Key、绝对用户路径或堆栈。

## 7. 错误结构

请求级错误：

```json
{
  "error": {
    "code": "invalid_image",
    "message": "无法解码上传的图片",
    "retryable": false,
    "details": {}
  }
}
```

| HTTP | code | 含义 | 可重试 |
|---:|---|---|---:|
| 400 | `invalid_request` | 表单或 JSON 不合法 | 否 |
| 400 | `invalid_image` | 无法解码 | 否 |
| 422 | `validation_error` | 字段缺失、格式或取值范围不合法 | 否 |
| 413 | `image_too_large` | 文件或尺寸超限 | 否 |
| 415 | `unsupported_media_type` | 格式不支持 | 否 |
| 404 | `review_not_found` | 任务不存在 | 否 |
| 429 | `queue_full` | 队列满 | 是 |
| 500 | `internal_error` | 内部错误 | 是 |
| 503 | `models_unavailable` | 两个模型均不可用 | 是 |

POST 成功后发生的模型错误必须写入任务结果，不再改变 POST 的状态码。任务内部错误格式：

```json
{
  "component": "aide",
  "code": "model_timeout",
  "message": "AIDE推理超时",
  "retryable": true
}
```

## 8. 模型 Worker 内部 API

内部接口只监听回环地址或受保护的容器网络。

### 8.1 AIDE Worker

```http
POST http://127.0.0.1:8101/internal/v1/predict
Content-Type: multipart/form-data
```

请求包含 `image` 和 `request_id`。成功响应：

```json
{
  "request_id": "vr_...",
  "status": "ok",
  "model_version": "aide-v1",
  "ai_probability": 0.86,
  "real_probability": 0.14,
  "latency_ms": 423
}
```

AIDE 适配器必须读取模型标签映射，禁止假定数组位置；logits 必须 softmax，已有概率不得重复 softmax。模型在服务启动时加载，并使用 evaluation/inference 模式。

### 8.2 IML-ViT Worker

```http
POST http://127.0.0.1:8102/internal/v1/predict
Content-Type: multipart/form-data
```

```json
{
  "request_id": "vr_...",
  "status": "ok",
  "model_version": "iml-vit-v1",
  "mask_encoding": "png_gray16_base64",
  "mask_width": 1080,
  "mask_height": 1440,
  "probability_mask": "iVBORw0KGgoAAA...",
  "latency_ms": 618
}
```

`probability_mask` 是无 `data:` 前缀的单通道 16-bit PNG Base64。像素 0 对应概率 0，65535 对应概率 1。Worker 必须记录 resize/padding 并逆变换到用户可见原图尺寸，禁止直接拉伸导致坐标错位。

Worker 通用错误：

```json
{
  "request_id": "vr_...",
  "status": "error",
  "model_version": "aide-v1",
  "error": {"code": "inference_failed", "message": "模型推理失败", "retryable": true}
}
```

默认单 Worker 超时 30 秒，超时允许自动重试一次。两个 Worker 必须并行调用。

## 9. 图片处理规范

固定顺序：

1. 流式读取，并在读取时执行 10 MiB 限制。
2. 计算原始字节 SHA-256。
3. 通过 magic bytes 验证格式。
4. 解码第一帧并获取尺寸。
5. 保存未经重编码的原始字节。
6. 生成应用 EXIF Orientation 的工作副本。
7. 转换为 sRGB/RGB；透明区域合成白色背景。
8. 分别进入两个模型的官方预处理。

禁止在计算 SHA-256 前重编码、覆盖原图，或让两个模型共享不符合各自要求的固定 resize。所有 Mask 和 bbox 以应用 Orientation 后、用户实际看到的图片坐标为准。

## 10. IML-ViT 后处理

默认配置：

```json
{
  "pixel_threshold": 0.50,
  "min_region_area_ratio": 0.005,
  "max_regions": 20,
  "morphology_kernel": 3,
  "overlay_alpha": 0.45,
  "top_percent_for_image_score": 0.05
}
```

处理步骤：

1. 将 16-bit PNG 解码为 `[0,1]` 浮点图 `M`。
2. Mask 尺寸必须与工作图一致，否则作为模型错误，不静默拉伸。
3. `binary = M >= 0.50`。
4. 执行一次 `3×3` 闭运算，再执行一次 `3×3` 开运算。
5. 查找 8 邻域连通区域。
6. 删除面积小于全图 `0.5%` 的区域。
7. 区域 `score` 为区域内 `M` 的平均值。
8. `area_ratio = region_pixels / image_pixels`。
9. 按 score 降序，最多保留 20 个。
10. 最终二值 Mask 只包含保留区域。

图像级指标：

```text
mask_area_ratio = 最终二值 Mask 像素数 / 总像素数
tamper_probability = M 中最高 5% 像素的平均值
```

生成物：

- `probability-mask.png`：`round(M × 255)` 的 8-bit 灰度图。
- `binary-mask.png`：仅包含 0、255。
- `overlay.png`：原图上以 0.45 透明度叠加红色 Mask。

## 11. 融合规则

```json
{
  "aide_high": 0.75,
  "aide_low": 0.35,
  "tamper_high": 0.60,
  "local_area_min": 0.02,
  "global_area_min": 0.60,
  "rules_version": "fusion-rules-v1"
}
```

设 `P_ai=AIDE.ai_probability`、`P_tamper=tamper_probability`、`area=mask_area_ratio`。两个模型成功时按顺序匹配，命中即停止：

| 顺序 | 条件 | source_type | risk |
|---:|---|---|---|
| 1 | `P_ai>=0.75 and area>=0.60` | `likely_ai_generated` | high |
| 2 | `P_ai>=0.75 and P_tamper>=0.60 and 0.02<=area<0.60` | `suspected_ai_assisted_edit` | high |
| 3 | `P_ai>=0.75` | `likely_ai_generated` | medium |
| 4 | `P_ai<=0.35 and P_tamper>=0.60 and area>=0.02` | `edited_photo` | medium |
| 5 | `P_ai<=0.35 and P_tamper<0.60 and area<0.02` | `no_anomaly_detected` | low |
| 6 | 其他 | `inconclusive` | unknown |

置信度保留六位小数：

```text
likely_ai_generated       = P_ai
suspected_ai_assisted_edit= 0.6*P_ai + 0.4*P_tamper
edited_photo              = 0.7*P_tamper + 0.3*(1-P_ai)
no_anomaly_detected       = 0.5*(1-P_ai) + 0.5*(1-P_tamper)
inconclusive              = 0.5
```

融合后的 `confidence` 禁止命名为 `ai_probability`。

### 11.1 降级

| AIDE | IML-ViT | status | 允许结论 |
|---|---|---|---|
| 成功 | 失败 | degraded | `P_ai>=0.75` 时 `likely_ai_generated/medium`，否则 `inconclusive`；无区域 |
| 失败 | 成功 | degraded | `P_tamper>=0.60 and area>=0.02` 时 `edited_photo/medium`，否则 `inconclusive`；不得声称 AIGC |
| 失败 | 失败 | failed | `inconclusive/unknown/0.5` |

## 12. 证据表达

允许的 finding code：

```text
global_aigc_probability_high
global_aigc_probability_low
localized_manipulation_detected
no_localized_manipulation_detected
model_results_conflict
model_result_degraded
```

允许表达“疑似”“概率较高”“建议人工复核”。禁止表达“已证明由 AI 生成”“红色区域确定是 AI 生成的”“未发现异常所以绝对真实”。

每个完成或降级结果至少包含一条限制：模型属于辅助证据；篡改不等同于 AI 生成；压缩、裁剪和截图可能影响结果。

## 13. 状态机与队列

合法转换：

```text
queued -> running -> completed | degraded | failed
queued -> failed
```

终态不得回退。阶段进度：`validating=0.05`、`queued=0.10`、`model_inference=0.20～0.65`、`postprocessing=0.75`、`fusion=0.85`、`storing_artifacts=0.95`、`completed=1.0`。

第一版可以使用进程内 `asyncio.Queue`，默认最大 16、并发 1。任务异常不得终止消费者循环；应用重启后未完成任务标记为 `failed`，错误码 `worker_restarted`。

## 14. 存储

```text
runtime/agent3/{review_id}/
├── original.bin
├── input.json
├── result.json
└── artifacts/
    ├── probability-mask.png
    ├── binary-mask.png
    └── overlay.png
```

JSON 必须使用临时文件加原子替换写入。所有路径由服务端根据 `review_id` 生成。默认保留 24 小时；清理逻辑只能操作经过校验且位于 `runtime/agent3/` 下的目录。日志只记录 SHA-256 前 12 位，不记录图片和 Base64。

## 15. 配置

```text
AGENT3_ENABLED=true
AGENT3_RUNTIME_DIR=runtime/agent3
AGENT3_MAX_UPLOAD_BYTES=10485760
AGENT3_MAX_QUEUE_SIZE=16
AGENT3_CONCURRENCY=1
AIDE_WORKER_URL=http://127.0.0.1:8101
IML_VIT_WORKER_URL=http://127.0.0.1:8102
AGENT3_MODEL_TIMEOUT_SECONDS=30
AGENT3_ARTIFACT_TTL_HOURS=24
```

阈值必须集中在 `config.py` 或一个版本化 JSON 文件，不得散落在路由与测试中。

## 16. 实现边界

`schemas.py` 至少定义：

```text
ProductContext, InputMetadata, ModelError, AideModelResult,
ImlVitModelResult, VisualRegion, ArtifactLinks, VisualEvidence,
TaskError, ReviewAccepted, ReviewProgress, VisualReviewResult, HealthResult
```

除 `ProductContext(extra="ignore")` 外，其余模型建议设置 `extra="forbid"`。概率使用 `Field(ge=0,le=1)`，字符串和数组必须设置上限。

`detectors/base.py`：

```python
class AigcDetector(Protocol):
    async def predict(self, image_bytes: bytes, request_id: str) -> AideModelResult: ...

class TamperLocalizer(Protocol):
    async def predict(self, image_bytes: bytes, request_id: str) -> RawMaskResult: ...
```

`service.py`：

```python
class VisualReviewService:
    async def create_review(...) -> ReviewAccepted: ...
    async def get_review(review_id: str) -> ReviewProgress | VisualReviewResult: ...
    async def process_review(review_id: str) -> None: ...
```

路由只负责 HTTP 校验和响应转换，禁止直接调用模型、执行图像后处理或编写融合条件。

在 `router.py` 定义：

```python
router = APIRouter(prefix="/api/v1/agent3", tags=["agent3"])
```

在现有 `app/main.py` 中注册：

```python
from .agent3.router import router as agent3_router
app.include_router(agent3_router)
```

不得破坏已有 `/api/lab`、`/api/model/test` 和 `/api/health`。

## 17. 测试矩阵

### 17.1 Schema/API

必须验证概率越界、非法枚举、负数 bbox、`regions=null`、非 ok 模型携带概率等情况均失败；合法 PNG/JPEG 返回 202；伪装文本返回 400/415；超限返回 413；非法 ID 返回 422；不存在任务和非法 Artifact 返回 404；路径穿越不能读取任意文件；队列满返回 429。

### 17.2 融合参数化测试

| P_ai | P_tamper | area | 预期类型 |
|---:|---:|---:|---|
| 0.90 | 0.90 | 0.80 | likely_ai_generated |
| 0.90 | 0.80 | 0.20 | suspected_ai_assisted_edit |
| 0.90 | 0.20 | 0.00 | likely_ai_generated |
| 0.10 | 0.80 | 0.20 | edited_photo |
| 0.10 | 0.20 | 0.00 | no_anomaly_detected |
| 0.55 | 0.80 | 0.20 | inconclusive |
| 0.55 | 0.20 | 0.00 | inconclusive |

边界值 `0.35/0.75/0.60/0.02/0.60 area` 必须分别测试。

### 17.3 后处理与降级

必须验证小区域删除、多区域排序、20 个上限、bbox 边界、top 5% 均值、16-bit PNG 编解码误差不超过 `1/65535+1e-6`。还必须覆盖单 Worker 超时、双 Worker 超时、非法 JSON、Mask 尺寸错误和 Artifact 写入失败。

### 17.4 真实模型冒烟

每个模型至少用一张真实图、一张完整 AI 图和一张带已知局部编辑 Mask 的图运行。冒烟测试只证明可加载、输出合法和 Artifact 可生成，不代表准确率。

### 17.5 阶段 A Fake Worker 的结果解释

阶段 A 的 Fake Worker 只用于契约和状态机联调，不产生真实检测结论。使用普通 PNG 运行时，预期可以得到：

- `aide-fake-v1` 返回低 AIGC 概率；
- `iml-vit-fake-v1` 返回全零概率图；
- `probability-mask.png` 和 `binary-mask.png` 为全黑；
- `overlay.png` 与输入图像视觉一致；
- `source_type=no_anomaly_detected`、`risk=low`。

这组结果只能证明上传、推理桩、后处理、融合和 Artifact 链路正常，不能证明图片真实或没有 AIGC。阶段 A 不得用空 Mask 评估检测准确率；只有阶段 B/C 接入真实权重后，才允许进行模型效果评估。

## 18. 评测与阈值校准

最低建议数据量：真实产品图、完整 AI 图、AI 局部重绘图、传统 PS/拼接图各 50 张。局部修改图必须有人工核验的 Ground Truth Mask。

报告 AIDE 的 Precision、Recall、F1、真实图假阳性率；IML-ViT 的 Pixel F1、IoU、图片级 F1；融合结果的五分类混淆矩阵。还要分开报告原图、平台压缩、截图和缩放图。阈值只在 calibration 集上确定，禁止用最终 test 集反复调参。

## 19. 安全与隐私

- 默认完全本地处理图片。
- 禁止记录请求体、图片 Base64、密钥、模型权重路径。
- 捕获损坏图片异常和解压炸弹警告。
- 禁止通过本地路径或远程 URL 提交图片。
- 第一版不处理 SVG、PDF、GIF 动画和视频。
- Artifact 只能从对应 review 的白名单文件返回。

## 20. 实施顺序

### A. 契约与 Fake Worker

建立 Schema、状态机、四个外部接口、两个测试桩和契约测试。普通合法图片返回全零 Fake Mask 属于预期行为。完成标志：无真实模型时可以跑通上传、轮询、融合结果和 Artifact，并能明确区分“接口通过”和“检测有效”。

### B. IML-ViT

建立隔离环境，固定 checkpoint 及 SHA-256，实现 Worker、坐标逆变换、16-bit Mask、后处理和三种 Artifact。完成标志：已知局部修改样本产生尺寸正确的 Mask 和 bbox，并能用真实模型结果替换 Fake Worker 而不改变业务 API。

#### B 阶段中期总结（2026-09-23）

阶段 B-2 已完成“真实 IML-ViT Worker 到 Windows Agent3”的端到端联调；阶段 B 的效果校准尚未开始。

已完成：

| 项目 | 状态 | 已验证证据 |
|---|---|---|
| Ubuntu 隔离环境 | 完成 | `imlvit`，Python 3.8.20、PyTorch 2.4.1+cu121、`cuda=True` |
| IML-ViT 官方源码与 checkpoint | 完成 | CASIAv2 checkpoint，`iml-vit_checkpoint.pth`，351 MiB，SHA-256：`9da3d79187fc21841e2b4aa235364187aacb6de6ead6dfd9be2f77ffc278d52d` |
| 权重加载 | 完成 | `missing: 0`、`unexpected: 0` |
| GPU 前向推理 | 完成 | 输出 Mask 尺寸 `(1, 1, 1024, 1024)` |
| Ubuntu Worker | 完成 | `GET /health` 返回 `ready`；`POST /internal/v1/predict` 返回 16-bit PNG Base64 Mask |
| Windows Remote Client | 完成 | `RemoteImlVitLocalizer` 解码 16-bit Mask，且 `trust_env=False` 避免把内部调用误送到本机代理 |
| 教学平台端口映射 | 完成 | 内部 `8102` 映射至受限网络的外部端口 |
| Agent3 端到端任务 | 完成 | 真实 IML-ViT 推理约 971 ms；生成 Probability Mask、Binary Mask、Overlay、区域框和 L6 Evidence |

已验证任务 `vr_b44a8472fdc8484d8683e69f` 的真实模型结果：原图 `1487×1058`，`tamper_probability=0.693204`、`mask_area_ratio=0.024947`，输出两个局部疑似篡改区域，任务总耗时约 1.86 秒。该结果仅证明模型链路和坐标映射有效，不代表该截图确定被编辑或由 AI 生成。

阶段 B 冒烟样本记录：

| 样本 | review_id | 已知情况 | IML-ViT 结果 | 结论 |
|---|---|---|---|---|
| `8caa49256111a83cc335683be011f347.jpg` | `vr_d0dcf60ed49943618ecaef8b` | 真实、未编辑产品图 | `tamper=0.443526`、`area=0`、无区域 | 负样本通过：未产生达到阈值的局部区域 |
| `negative_AIGC.png` | `vr_bb91d191e3b346d7b3ef78cb` | 使用 image2 基于原图进行局部修改 | `tamper=0.349784`、`area=0.017629`、1 个区域 | 局部编辑部分通过：Overlay 与真实修改位置部分重合，但信号低于融合阈值 |
| `3.png` | `vr_d08234c7a9744653ab33a56c` | 从零生成的完整 AI 产品图 | `tamper=0.998308`、`area=0.111935`、2 个区域 | 强响应通过：检测到高置信局部异常；IML-ViT 仍不能单独证明整图为 AI 生成 |

**阶段 B 工程验收结论：通过。**真实 IML-ViT 已从远程 GPU Worker 接入 Agent3，三类冒烟图片均返回尺寸正确的 Mask、区域与 Artifact，且无运行错误。完整 AI 图在当前 `aide-fake-v1` 存在时被融合为 `edited_photo`，这符合第一版的职责边界：IML-ViT 是通用篡改定位模型，不是整图 AIGC 分类器。阶段 C 接入真实 AIDE 后才可对整图 AIGC 概率进行有效融合。

尚未完成：

- 使用有 Ground Truth 的局部编辑图验证 Mask/区域的 IoU、Pixel F1；
- 记录 checkpoint 的下载日期与来源；SHA-256 已固定为 `9da3d79187fc21841e2b4aa235364187aacb6de6ead6dfd9be2f77ffc278d52d`；
- 校准阈值并评估截图、平台压缩和缩放造成的假阳性；
- 接入真实 AIDE（阶段 C），当前 `aide-fake-v1` 不能用于真实融合结论；
- 为 Worker 加入鉴权或限制端口映射访问范围。

#### B-2 每次开机后的远程启动流程

前提：Ubuntu 教学服务器上保留以下文件，且 Windows 本地项目已包含本仓库的 B-2 代码。

```text
/home/<service-user>/agent3-models/IML-ViT-main/
├── checkpoints/iml-vit_checkpoint.pth
└── iml_vit_model.py

/home/<service-user>/agent3-worker/model_workers/iml_vit/
├── app.py
├── requirements.txt
└── start_worker.sh
```

1. 在 Ubuntu 教学服务器打开终端，进入 Worker 目录并启动：

```bash
cd /home/<service-user>/agent3-worker/model_workers/iml_vit
bash start_worker.sh
```

启动成功必须看到：

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:8102
```

该终端必须保持运行。若 `start_worker.sh` 尚未上传，可临时使用：

```bash
cd /home/<service-user>/agent3-worker/model_workers/iml_vit && IML_VIT_REPO=/home/<service-user>/agent3-models/IML-ViT-main IML_VIT_CHECKPOINT=/home/<service-user>/agent3-models/checkpoints/iml_vit/iml-vit_checkpoint.pth python -m uvicorn app:app --host 0.0.0.0 --port 8102
```

2. 在 Ubuntu 的第二个终端中验证 Worker。教学服务器自动注入的 `127.0.0.1:7897` 代理可能拦截本地 HTTP 请求，因此必须临时清除代理变量：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy curl -4 http://127.0.0.1:8102/health
```

预期：

```json
{"status":"ready","model_version":"iml-vit-v1","device":"cuda"}
```

3. 在教学平台配置端口转发。必须配置为：

```text
内部端口：8102
外部地址：<worker-host>:<iml-vit-port>
```

不得把旧代理映射 `7897 → 8055` 当作 Worker 映射。每次重启教学实例后，应重新确认该转发规则仍存在。

4. 在 Windows PowerShell 验证映射：

```powershell
curl.exe --noproxy "*" --max-time 10 http://<worker-host>:<iml-vit-port>/health
```

5. 在 Windows 项目中启动 Agent3。推荐直接运行仓库中的脚本：

```powershell
Set-Location 'E:\the-Louvre-agent\the-Louvre-agent\product-agent-demo'
powershell -ExecutionPolicy Bypass -File .\scripts\start-agent3-with-imlvit.ps1 -AideWorkerUrl "http://<worker-host>:<aide-port>" -ImlVitWorkerUrl "http://<worker-host>:<iml-vit-port>"
```

脚本会设置：

```text
IML_VIT_WORKER_URL=http://<worker-host>:<iml-vit-port>
AIDE_WORKER_URL=http://<worker-host>:<aide-port>
AGENT3_MODEL_TIMEOUT_SECONDS=30
IML_VIT_MODEL_VERSION=iml-vit-v1
AIDE_MODEL_VERSION=aide-genimage-v1
```

并使用项目 `.venv` 启动 `http://127.0.0.1:8000`。运行后验证：

```powershell
curl.exe http://127.0.0.1:8000/api/v1/agent3/health
```

预期 `models.iml_vit.model_version` 为 `iml-vit-v1`；若显示 `iml-vit-fake-v1`，说明 Windows 进程未读取 `IML_VIT_WORKER_URL`，应关闭并按第 5 步重启。

6. 关闭顺序：先停止 Windows Agent3（`Ctrl+C`），再停止 Ubuntu Worker（`Ctrl+C`），最后在教学平台移除不再需要的外部端口映射。Worker 当前无鉴权，端口映射仅应在联调期间开放。

### C. AIDE

阶段 C 已完成真实 AIDE 整图 AIGC Worker 的部署和 Agent3 接入。它与 IML-ViT 使用完全隔离的环境，不得把 AIDE 依赖写入 Windows 主控的 `requirements.txt`。

#### C.1 已验收的服务器环境与文件位置

```text
Conda 环境：/opt/conda/envs/aide （Python 3.10）
PyTorch：2.0.1+cu118
NumPy：1.24.4
GPU：NVIDIA GeForce RTX 4090 D
官方源码：/home/<service-user>/agent3-models/AIDE-main
官方权重：/home/<service-user>/agent3-models/checkpoints/aide/GenImage_train.pth
Worker：/home/<service-user>/agent3-worker/model_workers/aide
```

权重来自 AIDE 官方 Model Zoo 的 `GenImage_train.pth`。它是整图二分类模型；官方标签映射是 `0_real -> 0`、`1_fake -> 1`，因此 Worker 必须将 `softmax(logits)[1]` 返回为 `ai_probability`。

#### C.2 每次教学服务器重启后的启动流程

在两个独立 Ubuntu 终端中分别运行；两个终端都必须保持打开。

```bash
# 终端 1：AIDE（脚本会自行激活 aide 环境）
bash ~/agent3-worker/model_workers/aide/start_worker.sh
```

```bash
# 终端 2：IML-ViT
conda activate imlvit
cd ~/agent3-worker/model_workers/iml_vit
IML_VIT_REPO=/home/<service-user>/agent3-models/IML-ViT-main \
IML_VIT_CHECKPOINT=/home/<service-user>/agent3-models/checkpoints/iml_vit/iml-vit_checkpoint.pth \
python -m uvicorn app:app --host 0.0.0.0 --port 8102
```

本次教学平台端口映射为：

```text
<worker-host>:<aide-port> -> 容器 8101 -> AIDE
<worker-host>:<iml-vit-port> -> 容器 8102 -> IML-ViT
```

先在 Windows 验证两个服务：

```powershell
curl.exe --noproxy "*" --max-time 20 http://<worker-host>:<aide-port>/health
curl.exe --noproxy "*" --max-time 20 http://<worker-host>:<iml-vit-port>/health
```

然后从项目根目录运行：

```powershell
.\scripts\start-agent3-with-imlvit.ps1 -AideWorkerUrl "http://<worker-host>:<aide-port>" -ImlVitWorkerUrl "http://<worker-host>:<iml-vit-port>"
```

`GET /api/v1/agent3/health` 必须显示 `aide-genimage-v1` 与 `iml-vit-v1`；任何一个显示 fake 版本都表示 Windows 主控未读取对应 Worker URL，需要停止并重新启动。

#### C.3 阶段 C 端到端验收记录

已知从零生成图片 `images/3.png` 的任务 `vr_52820d464cbe4653a66ec284` 完成且无 errors：

| 模型 | 结果 |
|---|---|
| AIDE | `ai_probability=0.940639`，`real_probability=0.059361`，`latency_ms=18128` |
| IML-ViT | `tamper_probability=0.998308`，`mask_area_ratio=0.111935`，`latency_ms=20303` |
| 融合 | `risk=high`，`confidence=0.963707` |

该样本的 `source_type` 是 `suspected_ai_assisted_edit`，原因是当前 `fusion-rules-v1` 将高 AIDE 概率与局部 IML-ViT 区域合并为“疑似 AI 辅助编辑”。这证明服务链路和模型输出均正常，但不能据单个样本认为标签策略已经校准；应在阶段 E 用独立的真实图、整图 AI 图与局部编辑图验证并调整规则。

### D. 融合与主控

并行调用 Worker，实现 `fusion-rules-v1`、L6 Evidence、降级规则，并向现有 FastAPI 注册 router。主控必须直接读取 JSON，不通过自然语言解析。

### E. 校准与展示

建立四类测试集，校准阈值，测试压缩退化；前端展示 overlay、区域框、模型分数和限制说明。

## 21. 完成定义

- [ ] 四个外部 API 完全符合本文契约。
- [ ] AIDE 与 IML-ViT 使用真实权重完成推理。
- [ ] 双模型正常时并行运行。
- [ ] 单模型失败返回 `degraded` 且不越权下结论。
- [ ] 双模型失败返回 `failed/inconclusive`。
- [ ] Mask、bbox、overlay 使用用户可见图坐标。
- [ ] 所有模型证据为 L6。
- [ ] 无“确定由 AI 生成”等过度表述。
- [ ] Schema、API、融合、后处理、降级测试全部通过。
- [ ] 项目原有测试保持通过。
- [ ] 权重、用户图、Artifact、密钥均未进入 Git。
- [ ] README 记录两个 Worker 与 Agent3 API 的启动方式。
- [ ] 记录每个 checkpoint 的来源、文件名、SHA-256、许可证。

## 22. V2 预留

V2 可加入 Agent1 官方图包装比对、OCR 文字修改定位、前后图配准、FUSED、学习型融合、GPU 批处理和数据库持久化。V2 可以增加字段，但不得改变 V1 字段含义；破坏性修改必须提升 `schema_version` 主版本。
