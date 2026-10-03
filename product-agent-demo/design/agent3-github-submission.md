# Agent3 GitHub 提交清单与贡献声明

本文用于准备 Agent3 第一版的 GitHub commit / Pull Request。目标是只提交 Agent3 的源代码、测试和文档，不提交模型权重、运行产物、用户图片或其他功能的未完成改动。

## 1. 建议提交的文件

### 1.1 Agent3 主控代码（全部提交）

```text
product-agent-demo/app/agent3/__init__.py
product-agent-demo/app/agent3/config.py
product-agent-demo/app/agent3/fusion.py
product-agent-demo/app/agent3/postprocess.py
product-agent-demo/app/agent3/repository.py
product-agent-demo/app/agent3/router.py
product-agent-demo/app/agent3/schemas.py
product-agent-demo/app/agent3/service.py
product-agent-demo/app/agent3/detectors/__init__.py
product-agent-demo/app/agent3/detectors/base.py
product-agent-demo/app/agent3/detectors/fake.py
product-agent-demo/app/agent3/detectors/aide_client.py
product-agent-demo/app/agent3/detectors/iml_vit_client.py
```

### 1.2 Ubuntu 模型 Worker（全部提交）

```text
product-agent-demo/model_workers/aide/app.py
product-agent-demo/model_workers/aide/requirements.txt
product-agent-demo/model_workers/aide/start_worker.sh
product-agent-demo/model_workers/iml_vit/app.py
product-agent-demo/model_workers/iml_vit/requirements.txt
product-agent-demo/model_workers/iml_vit/start_worker.sh
```

这些文件只包含适配和服务封装，不包含 AIDE、IML-ViT 官方源码及模型权重。

### 1.3 启动脚本、测试和文档（全部提交）

```text
product-agent-demo/scripts/start-agent3-with-imlvit.ps1
product-agent-demo/tests/test_agent3_api.py
product-agent-demo/tests/test_agent3_degradation.py
product-agent-demo/tests/test_agent3_fusion.py
product-agent-demo/tests/test_agent3_iml_vit_client.py
product-agent-demo/tests/test_agent3_postprocess.py
product-agent-demo/tests/test_agent3_schemas.py
product-agent-demo/design/agent3-v1-implementation-guide.md
product-agent-demo/design/agent3-github-submission.md
product-agent-demo/design/agent3-ubuntu-dual-model-deployment.md
product-agent-demo/design/agent3-issue-drafts.md
```

### 1.4 需要提交的既有文件改动

```text
product-agent-demo/.env.example
product-agent-demo/.gitignore
product-agent-demo/README.md
product-agent-demo/requirements.txt
```

`product-agent-demo/app/main.py` 只暂存以下两个 Agent3 相关改动：

```python
from .agent3.router import router as agent3_router
app.include_router(agent3_router)
```

该文件目前还包含 `.mjs` MIME 修复，那不是 Agent3 改动；若要保持 PR 纯净，应通过 `git add -p` 分块暂存，而不是整文件暂存。

## 2. 默认不提交的内容

```text
product-agent-demo/images/
product-agent-demo/runtime/agent3/
product-agent-demo/binary-mask.png
product-agent-demo/probability-mask.png
product-agent-demo/overlay.png
任何 *.pth、*.pt、*.ckpt、*.bin 模型权重
AIDE 和 IML-ViT 的完整上游源码副本
.env、API Key、服务器账号、SSH Key
```

`tests/test_workbench_http.py` 当前改动属于工作台静态模块 MIME 修复，不应混入仅包含 Agent3 的提交。测试图片只有在来源、版权和隐私都明确，并转为小尺寸匿名 fixture 后才可提交。

## 3. 推荐暂存方式

不要执行 `git add .`。在仓库根目录按以下边界暂存：

```powershell
git add product-agent-demo/app/agent3
git add product-agent-demo/model_workers
git add product-agent-demo/scripts/start-agent3-with-imlvit.ps1
git add product-agent-demo/tests/test_agent3_*.py
git add product-agent-demo/design/agent3-*.md
git add product-agent-demo/.env.example product-agent-demo/.gitignore product-agent-demo/README.md product-agent-demo/requirements.txt
git add -p product-agent-demo/app/main.py
```

分块暂存 `app/main.py` 时，只选择 Agent3 import 和 `include_router` 所在 hunk。暂存后必须执行：

```powershell
git diff --cached --check
git diff --cached --stat
git diff --cached
```

## 4. 提交前验收

- [ ] `python -m unittest discover -s tests -p "test_agent3*.py"` 全部通过。
- [ ] `GET /api/v1/agent3/health` 显示期望的真实或 Fake 模型版本。
- [ ] 未暂存权重、用户图片、Artifact、`.env` 或服务器凭据。
- [ ] AIDE 与 IML-ViT 的上游项目和许可证在 PR 中注明。
- [ ] PR 明确“模型结果属于辅助证据”，不宣称能够确定图片来源。
- [ ] 已说明阶段 E 的校准和效果验收尚未完成。

## 5. 建议 commit 与 PR 标题

```text
feat(agent3): add dual-model visual authenticity review demo
```

```text
Agent3: integrate remote AIDE and IML-ViT workers with evidence fusion
```

## 6. 个人完成任务声明（可直接放入 PR）

```markdown
### Agent3 贡献说明

本人负责并完成了 Agent3 第一版 Demo 的设计、实现与联调，具体包括：

- 定义 Agent3 图片审核 API、结构化 Schema、任务状态和错误/降级契约；
- 实现 AIDE 整图 AIGC 检测与 IML-ViT 局部篡改定位的双模型架构；
- 封装两个独立 Ubuntu GPU Worker，并完成 Windows 主控的远程 HTTP 接入；
- 实现概率 Mask 后处理、区域框、叠加图、固定融合规则和 L6 辅助证据输出；
- 完成真实模型权重加载、GPU 推理、端口映射和端到端冒烟测试；
- 编写 Agent3 API、Schema、融合、后处理和单模型降级测试，以及部署/迁移文档。

本提交不声称本人训练或原创 AIDE、IML-ViT 模型。两者均来自对应的开源上游项目；本人的工作范围是 Agent3 方案设计、工程集成、服务封装、部署联调、测试与文档。当前 Demo 已完成工程链路验收，系统校准和大规模效果验收将在后续 Issue 中继续完成。
```

提交前将“本人”替换为姓名或 GitHub 用户名（如团队规范要求），并在 PR 中链接后续校准 Issue。
