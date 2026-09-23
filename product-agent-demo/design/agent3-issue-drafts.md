# Agent3 GitHub Issue 草案

以下内容可逐条复制为 GitHub Issue。已完成项用于记录工程贡献和验收证据；未完成项用于阶段 E 系统校准与效果验收。不要把“模型成功返回结果”等同于“模型准确率达标”。

## 已完成 Issue

### Issue 1：完成 Agent3 V1 API、Schema 与异步审核流程

**建议标签：** `agent3`, `backend`, `completed`

**状态：** 已完成

**内容：**

- 实现 `POST /api/v1/agent3/reviews`、结果查询、Artifact 下载和健康检查。
- 冻结 `VisualReviewResult`、模型结果、区域、证据和错误 Schema。
- 实现上传格式/尺寸校验、任务状态、运行目录和结构化结果存储。
- 所有模型结论限定为 L6 辅助证据，保留不确定性说明。

**验收证据：** Agent3 API、Schema 和相关测试通过。

---

### Issue 2：完成 IML-ViT 远程 GPU Worker 与局部篡改定位

**建议标签：** `agent3`, `model-integration`, `completed`

**状态：** 已完成

**内容：**

- 封装 IML-ViT Ubuntu Worker 和 Windows 异步客户端。
- 实现 16-bit PNG Base64 概率 Mask 契约。
- 实现 Mask 后处理、连通区域、bbox、二值图和 overlay。
- 在 RTX 4090D 上严格加载 351MB 权重，权重键完全匹配。

**验收证据：** 真实图、局部编辑图和整图 AI 图均完成端到端推理并生成 Artifact。

---

### Issue 3：完成 AIDE 远程 GPU Worker 与整图 AIGC 概率接入

**建议标签：** `agent3`, `model-integration`, `completed`

**状态：** 已完成

**内容：**

- 使用独立 Python 3.10 / PyTorch 2.0.1 CUDA 环境部署 AIDE。
- 严格复用官方 DCT 五路输入预处理和 `0_real / 1_fake` 标签定义。
- 加载官方 `GenImage_train.pth`，返回整图 real/fake 概率。
- 实现 AIDE Worker、客户端、健康检查、超时和错误映射。

**验收证据：** 已知整图 AI 样本得到 `ai_probability=0.940639`，请求无错误。

---

### Issue 4：完成双模型融合、降级与端到端 Demo

**建议标签：** `agent3`, `integration`, `completed`

**状态：** 已完成

**内容：**

- 并行调用 AIDE 与 IML-ViT。
- 实现 `fusion-rules-v1`、风险等级、置信度和结构化 Evidence。
- 实现任一模型失败时的 `degraded` 结果及双模型失败处理。
- 完成 Windows 主控、Ubuntu 双 Worker、端口映射和真实模型联调。

**验收证据：** 已知 AI 图任务完成，AIDE 与 IML-ViT 均为 `ok`，融合 `risk=high`、`errors=[]`；Agent3 当前测试 13/13 通过。

## 后续阶段 E Issue

### Issue 5：建立 Agent3 校准集、测试集与数据清单

**建议标签：** `agent3`, `dataset`, `calibration`, `priority-high`

**目标：** 建立互斥的 calibration/test 数据，禁止用最终测试集反复调阈值。

**任务：**

- 收集真实相机图、完整 AI 生成图、传统编辑图、局部 AI 编辑图四类样本。
- 每类覆盖人物、商品、包装、室内外和不同生成器/编辑器。
- 为局部编辑样本提供像素 Mask 或可靠 bbox；记录来源、许可和编辑流程。
- 建立匿名样本 ID、SHA-256、标签、尺寸、压缩历史和生成工具元数据。
- 明确训练/校准/测试隔离规则，禁止把隐私图片提交到 Git。

**验收标准：**

- [ ] 数据清单可复现且无重复 SHA-256。
- [ ] 每类样本数量和来源分布达到小组约定下限。
- [ ] 标注抽检一致率达到小组约定标准。
- [ ] calibration 与最终 test 完全隔离。

---

### Issue 6：校准 AIDE、IML-ViT 阈值与融合标签语义

**建议标签：** `agent3`, `calibration`, `fusion`, `priority-high`

**背景：** 当前已知整图 AI 图因 IML-ViT 产生局部区域，被 `fusion-rules-v1` 命名为 `suspected_ai_assisted_edit`，说明工程链路正常但标签语义仍需校准。

**任务：**

- 在 calibration 集上搜索 `AIDE_HIGH/LOW`、`TAMPER_HIGH`、`LOCAL_AREA_MIN`、`GLOBAL_AREA_MIN`。
- 比较规则优先级，评估整图 AI 图不应被局部响应错误降为“AI 辅助编辑”。
- 保持输出为审慎表述，不把统计判断改写为确定事实。
- 将最终阈值、规则版本、数据版本和选择依据写入报告。

**验收标准：**

- [ ] 只使用 calibration 集选阈值。
- [ ] 冻结新的融合版本并补齐边界/冲突单元测试。
- [ ] 在独立 test 集报告五分类混淆矩阵。
- [ ] 规则变更不破坏降级行为和 API Schema。

---

### Issue 7：完成模型级与融合级效果验收

**建议标签：** `agent3`, `evaluation`, `priority-high`

**任务：**

- AIDE：报告 Precision、Recall、F1、AUROC、真实图假阳性率和校准曲线。
- IML-ViT：报告图片级 F1、Pixel F1、IoU、区域召回率和空 Mask 误报率。
- 融合：报告 source_type 混淆矩阵、高风险召回率和真实图误报率。
- 给出 95% 置信区间，并按图片来源/生成器/场景分层报告。

**验收标准：**

- [ ] 指标脚本、输入清单和输出报告可复现。
- [ ] 最终 test 只运行冻结版本。
- [ ] 明确失败案例和模型能力边界。
- [ ] 是否达到上线阈值由小组书面确认。

---

### Issue 8：完成压缩、截图、缩放与格式退化测试

**建议标签：** `agent3`, `robustness`, `evaluation`

**任务：**

- 对测试图施加 JPEG 多档压缩、平台二次压缩、截图、缩放、裁剪和 PNG/JPEG/WebP 转换。
- 比较原图与退化图的概率漂移、区域漂移和标签变化。
- 确认 EXIF 缺失、透明通道和超长宽比不会导致异常。

**验收标准：**

- [ ] 输出分操作、分强度的性能退化曲线。
- [ ] 定义不可接受的概率/区域漂移阈值。
- [ ] 将高风险退化条件加入用户可见的不确定性说明。

---

### Issue 9：完成并发、性能、稳定性与安全验收

**建议标签：** `agent3`, `performance`, `security`

**任务：**

- 测量冷启动、热启动、P50/P95/P99 延迟、显存峰值和吞吐量。
- 验证队列上限、超时、单模型崩溃、Worker 重启和网络中断。
- 为小组服务器配置进程守护、健康检查、私网访问、TLS/鉴权或防火墙策略。
- 验证上传大小限制、日志脱敏和 Artifact 生命周期。

**验收标准：**

- [ ] 连续运行和并发压测无内存/显存持续增长。
- [ ] Worker 故障时主控返回预期的 degraded/failed 结构。
- [ ] 无鉴权的 8101/8102 不暴露到公网。
- [ ] 形成可执行的启动、回滚和故障恢复手册。

---

### Issue 10：完善 Demo 展示与结果解释

**建议标签：** `agent3`, `frontend`, `documentation`

**任务：**

- 展示 AIDE 整图概率、IML-ViT 区域、overlay 和融合结论。
- 明确区分“局部篡改区域”和“AI 生成区域”。
- 展示模型版本、耗时、降级状态和不确定性。
- 为演示准备来源明确的真实图、整图 AI 图和局部编辑图。

**验收标准：**

- [ ] 用户能看到两模型各自贡献，而不是只看到单一标签。
- [ ] 页面不使用“确定 AI 生成”等过度表述。
- [ ] 单模型失败时 UI 明确标记降级，不隐藏错误。
