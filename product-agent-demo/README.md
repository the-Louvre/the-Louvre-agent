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

运行测试（在 `product-agent-demo` 目录内）：

```bash
python -m unittest discover -s tests
node --test tests/workspace-state.test.mjs
```

## 当前边界

MCP 工具名称和入参由百炼控制台实际开通的服务决定，因此通过页面配置或环境变量提供；没有配置检索服务时页面会保留识别结果并明确显示证据缺口，不会生成虚假的证据。社交媒体截图会被标记为 UGC，文案中的功效和参数只作为待核验声明。官方事实只接受监管来源或检索提供方明确标记为 `official`、`registration`、`authority` 的来源；没有可信等级的普通网页仍会保留为可点击的待核验来源。后续接入产品登记、成分库、淘宝/天猫评论时，为每个来源实现一个 `EvidenceProvider`，返回统一的 `EvidenceItem`，报告只读取结构化证据。下一阶段安排见 [`design/optimization-roadmap.md`](design/optimization-roadmap.md)。
