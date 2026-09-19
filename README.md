# the-Louvre-agent

## 产品识别与资料检索工作台

可运行原型位于 [`product-agent-demo/`](product-agent-demo/)。它提供真实模型/API 配置、社交媒体截图的 UGC 标记、图片批量并行、三个 Agent 的流式进度、结构化证据报告和结果导出。

```bash
cd product-agent-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

打开 <http://127.0.0.1:8000>，在右侧配置抽屉填写模型 API Key 和检索服务。Key 只留在当前页面内存中，不应提交到 Git。代码、测试和后续优化路线见 [`product-agent-demo/README.md`](product-agent-demo/README.md) 与 [`product-agent-demo/design/optimization-roadmap.md`](product-agent-demo/design/optimization-roadmap.md)。
