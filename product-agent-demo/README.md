# 百炼产品识别 Agent Demo

这是一个真实 API 接入的本地 Host Demo：上传一张或多张产品/社交媒体截图后，调用百炼兼容接口识别产品，再通过配置好的检索服务做资料检索，并在画布中展示三个 Agent 的可审计进度。批量最多保留 8 张图片，同时最多运行 3 张，避免一次性打满上游。代码不生成 mock 识别结果或假来源；缺少密钥会直接显示配置错误。

## 运行

```bash
cd product-agent-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export DASHSCOPE_API_KEY='在本机配置，不要提交到 Git'
# 如果使用百炼 Managed Agents / MCP 的工作空间地址，再配置 BAILIAN_WORKSPACE_ID 和 BAILIAN_MCP_URL
uvicorn app.main:app --reload --port 8000
```

浏览器打开 <http://127.0.0.1:8000>，在右侧抽屉填写模型连接和检索配置，上传一张或多张图片后点击“开始识别与检索”。API Key 只保存在当前页面内存中，不写入 localStorage 或仓库。

## 结构

- `app/lab.py`：三 Agent 并行识别、声明核验、图片核查、流式检索和结构化报告。
- `app/model_client.py`：DashScope/Qwen 兼容接口调用。
- `app/mcp_client.py`：Streamable HTTP MCP 调用边界，支持百炼官方/自定义 MCP。
- `app/orchestrator.py`：产品识别、证据检索、报告生成的事件编排。
- `app/main.py`：上传接口、SSE 事件流和本地 Host。
- `static/lab.html` + `static/product-workbench.js`：画布、图片素材列表、配置抽屉、批量进度和结果导出。
- `tests/`：模型输出边界、流式检索、批量状态和 HTTP 入口测试。

## 声明契约

视觉识别结果同时提供兼容字段 `claims` 和结构化字段 `claims_structured`：

- `claims` 是最多 8 条、每条最多 180 字的展示文本，保留给旧消费者。
- `claims_structured` 是供 `review` Agent 和后续 2B/2C 使用的声明对象。每条声明包含稳定的 `claim_id`、`input_id`、原文、规范化文本、声明类型、父级关系、文本片段、产品上下文、条件字段、解析状态和不确定原因。
- 条件字段包括功效指标、数值/单位、时间、人群、使用条件、实验样本量和背书实体。字段未出现时为 `null`，不会根据上下文补写。
- 专家、机构、论文和检测报告引用保存在 `conditions.endorsements`，并保留对应原文片段。

`input_id` 由前端每张图片的 UUID 传入；同一张图片重试时复用该 ID。没有传入 ID 的 API 调用会根据图片二进制或查询内容生成确定性 SHA-256 命名空间。服务端不会直接信任模型提供的 `claim_id`，而是根据输入 ID、声明原文、文本片段、声明类型和重复序号生成系统 ID；模型 ID 仅保存在 `upstream_claim_id`。

复合声明会保留一个父声明，并为拆分出的原子声明设置 `parent_claim_id`。只有上游提供了 `region_id` 或文本位置时才保留，否则使用 `null`。

声明预算超限不会静默丢失。识别结果中的 `claim_coverage` 会报告候选数量、已处理数量、预算排除项、无法解析数量和状态：`complete`、`partial`、`unparsed` 或 `empty`。例如：

```json
{
  "claim_coverage": {
    "candidate_count": 10,
    "processed_count": 8,
    "excluded_count": 2,
    "excluded": [{"reason": "budget_exceeded"}],
    "unparsed_count": 0,
    "status": "partial"
  }
}
```

`/api/lab/run` 的 `product_identified` SSE 事件包含 `input_id`、`claims`、`claims_structured` 和 `claim_coverage`。三个 Agent 共享这份识别结果；`review` Agent 必须沿用现有 `claim_id`，报告中的未知 `claim_id` 会被过滤并记录为证据缺口。旧字符串输入会自动适配为结构化声明，因此旧消费者无需立即迁移。

单条结构化声明示例：

```json
{
  "claim_id": "input-demo:claim:...",
  "original_text": "受试者30人，连续使用28天，细纹指标改善20%",
  "normalized_text": "细纹指标改善20%",
  "claim_type": "efficacy",
  "parent_claim_id": null,
  "input_id": "input-demo",
  "region_id": null,
  "conditions": {
    "efficacy_metric": {"value": "细纹", "original_text": "细纹指标"},
    "value": {"value": 20, "unit": "%", "original_text": "改善20%"},
    "time": {"value": 28, "unit": "天", "original_text": "连续使用28天"},
    "audience": null,
    "usage_condition": null,
    "sample_size": {"value": 30, "unit": "人", "original_text": "受试者30人"},
    "endorsements": []
  },
  "parse_status": "parsed",
  "uncertainty_reasons": []
}
```

OCR 不清晰或字段无法确认时，声明会进入 `partially_parsed` 或 `unparsed`，并在 `uncertainty_reasons` 中说明原因；不会补写确定数值。该契约只负责声明解析和追踪，证据检索与最终支持判定仍属于后续 2B/2C。

运行测试（在 `product-agent-demo` 目录内）：

```bash
python -m unittest discover -s tests
node --test tests/workspace-state.test.mjs
```

## Agent3 第一版阶段 A

阶段 A 已提供本地 Fake Worker，用于在真实 AIDE/IML-ViT 权重接入前完成接口联调。启动 Web 应用后，Agent3 会自动使用进程内的 `aide-fake-v1` 和 `iml-vit-fake-v1`，不需要外部 API Key：

```bash
uvicorn app.main:app --reload --port 8000
```

接口入口：

```text
POST /api/v1/agent3/reviews
GET  /api/v1/agent3/reviews/{review_id}
GET  /api/v1/agent3/reviews/{review_id}/artifacts/{artifact_name}
GET  /api/v1/agent3/health
```

阶段 A 的契约、状态机、Fake Worker、融合规则和验收条件见 [`design/agent3-v1-implementation-guide.md`](design/agent3-v1-implementation-guide.md)。真实模型接入阶段会将 Fake Worker 替换为独立的 AIDE/IML-ViT Worker，不改变上述业务接口。

## Agent3 阶段 B-2：Ubuntu IML-ViT Worker

阶段 B-2 将 IML-ViT 放在具备 CUDA 的 Ubuntu 服务器上运行，Windows 主控通过 HTTP 调用它。服务器端使用独立的 `imlvit` Conda 环境，模型权重默认放在：

```text
~/the-louvre-imlvit/IML-ViT/checkpoints/iml-vit_checkpoint.pth
```

在服务器上进入官方源码环境后启动 Worker：

```bash
cd product-agent-demo/model_workers/iml_vit
python -m pip install -r requirements.txt
export IML_VIT_REPO="${HOME}/agent3-models/IML-ViT-main"
export IML_VIT_CHECKPOINT="$IML_VIT_REPO/checkpoints/iml-vit_checkpoint.pth"
uvicorn app:app --host 0.0.0.0 --port 8102
```

Windows 主控配置 Worker 地址（若服务器通过 SSH 隧道映射到本机，则使用 `http://127.0.0.1:8102`；直接内网访问则使用服务器地址）：

```text
IML_VIT_WORKER_URL=http://127.0.0.1:8102
```

未设置 `IML_VIT_WORKER_URL` 时，Agent3 会继续使用阶段 A Fake Worker，便于离线运行契约测试。Worker 的内部接口为 `POST /internal/v1/predict`，只应暴露在受保护的内部网络中。

当前教学服务器部署可以直接使用：

```text
Ubuntu Worker 启动脚本：model_workers/iml_vit/start_worker.sh
Windows Agent3 启动脚本：scripts/start-agent3-with-imlvit.ps1
```

完整的远程启动、端口映射、验证与关闭流程见 [`design/agent3-v1-implementation-guide.md`](design/agent3-v1-implementation-guide.md) 的“B-2 每次开机后的远程启动流程”。

## Agent3 阶段 C：Ubuntu AIDE Worker

阶段 C 已将真实 AIDE `GenImage_train.pth` 权重接入为整图 AIGC 检测 Worker。它使用独立的 Python 3.10 / PyTorch 2.0.1 CUDA 环境，返回 `ai_probability` 与 `real_probability`；AIDE 不提供局部掩膜，局部区域仍由 IML-ViT 负责。

服务器端文件：

```text
源码：~/aide-deploy/AIDE-main
权重：~/aide-deploy/checkpoints/GenImage_train.pth
Worker：~/agent3-worker/model_workers/aide
```

Windows 同时接入本次教学服务器的两个真实 Worker：

```powershell
.\scripts\start-agent3-with-imlvit.ps1 -AideWorkerUrl "http://<worker-host>:<aide-port>" -ImlVitWorkerUrl "http://<worker-host>:<iml-vit-port>"
```

每次服务器重启后都必须恢复两个 Ubuntu Worker 与端口映射。完整的启动、验证、端口对应关系和已知测试结果见实施指导书的“C. AIDE”章节。

Agent3 相关交付文档：

- [`design/agent3-v1-implementation-guide.md`](design/agent3-v1-implementation-guide.md)：V1 接口、规则、阶段和验收基线；
- [`design/agent3-ubuntu-dual-model-deployment.md`](design/agent3-ubuntu-dual-model-deployment.md)：双模型迁移到小组 Ubuntu GPU 服务器；
- [`design/agent3-github-submission.md`](design/agent3-github-submission.md)：GitHub 文件清单、排除项与个人贡献声明；
- [`design/agent3-issue-drafts.md`](design/agent3-issue-drafts.md)：已完成工作和阶段 E 校准/效果验收 Issue 草案。

## 当前边界

MCP 工具名称和入参由百炼控制台实际开通的服务决定，因此通过页面配置或环境变量提供；没有配置检索服务时页面会保留识别结果并明确显示证据缺口，不会生成虚假的证据。社交媒体截图会被标记为 UGC，文案中的功效和参数只作为待核验声明。官方事实只接受监管来源或检索提供方明确标记为 `official`、`registration`、`authority` 的来源；没有可信等级的普通网页仍会保留为可点击的待核验来源。后续接入产品登记、成分库、淘宝/天猫评论时，为每个来源实现一个 `EvidenceProvider`，返回统一的 `EvidenceItem`，报告只读取结构化证据。下一阶段安排见 [`design/optimization-roadmap.md`](design/optimization-roadmap.md)。
