# 本地 LLM Agent Prompt/KV 缓存复用性能评估 - PPT 大纲

来源：`../main.tex`

目标受众：课程汇报。

表达目标：用 10 页幻灯片把实验问题、机制、设计、结果和应用建议讲清楚，突出“稳定前缀比单纯打开缓存更关键”这一主线。

## Slide 1: 标题页

- 布局角色：cover
- 页面意图：建立主题和核心结论，说明这是关于本地 Agent warm turn 延迟的性能评估。
- Key points:
  - 本地 LLM Agent 会反复提交长上下文，重复 prefill 推高 TTFT
  - 评估对象：llama.cpp prompt/KV 缓存、coding agent 与 RAG agent
  - 核心判断：稳定前缀组织方式决定缓存能否真正生效
- Visual idea: 用“长 prompt 前缀 + 动态后缀 + KV cache 命中”的抽象流程图作为主视觉。
- Required images: 无

## Slide 2: 问题背景：warm turn 为什么变慢

- 布局角色：context / problem
- 页面意图：解释多轮 Agent 中重复上下文和动态字段混杂导致 TTFT 上升。
- Key points:
  - 系统指令、工具定义、仓库摘要、检索文档会跨轮重复
  - 当前问题、时间戳、request id、最新错误日志通常每轮变化
  - 若稳定内容不能复用，长上下文每轮都要重新 prefill
  - 用户感知的等待主要体现在首 token 前的 TTFT
- Visual idea: 左侧展示多轮 prompt 的重复块，右侧展示 TTFT 随上下文变长升高。
- Required images: 无

## Slide 3: 机制假设：缓存收益来自前缀复用

- 布局角色：concept explanation
- 页面意图：用 TTFT 分解公式解释为什么完全一致的长前缀是缓存命中的前提。
- Key points:
  - TTFT 主要由新 token prefill、第一个 decode step 和固定开销组成
  - 若前缀 KV 可复用，只需处理未命中的短后缀
  - 有效新增长度可理解为 `L_new = L_prompt - L_reuse`
  - 缓存开关只是必要条件，prefix match 才决定收益
- Visual idea: 使用公式卡片 + token 条形图展示 `L_reuse` 和 `L_new`。
- Required images: 无

## Slide 4: 实验设计：负载、环境与策略

- 布局角色：experiment setup
- 页面意图：集中交代硬件、模型、请求设置、负载和 S0-S3 策略。
- Key points:
  - 硬件：Apple M5、32 GB unified memory，单并发运行
  - 模型：Qwen2.5-Coder-3B-Instruct Q4_K_M，llama-server + Metal
  - 负载：coding agent 与 RAG agent，覆盖 2K、4K、8K、16K
  - 策略：S0 关闭缓存，S1 动态字段前置，S2 稳定前缀，S3 扫描 `--cache-reuse`
- Visual idea: 用四列策略对比表和小型实验 pipeline。
- Required images: 无

## Slide 5: 主结果：稳定前缀让 TTFT 降到百毫秒级

- 布局角色：data evidence
- 页面意图：展示 warm TTFT 与加速比，强调 S2 和 S0/S1 的数量级差异。
- Key points:
  - S0/S1 在长 prompt 下 TTFT 达到秒级到几十秒级
  - S2 在 2K-16K 中保持约 0.10-0.33 秒量级
  - Coding 最高获得 112.2x TTFT 加速，RAG 最高获得 103.7x
  - 加速比随上下文增长而扩大，因为重复 prefill 被跳过
- Visual idea: 两张结果图并排作为主证据，下方放一句结论 callout。
- Required images:
  - Main evidence figure; strict input asset; preserve all chart data, axes, labels, legends, colors, and values.

    ![Warm TTFT by length](assets/figures/report_ttft_by_length.png)

  - Main evidence figure; strict input asset; preserve all chart data, axes, labels, legends, colors, and values.

    ![TTFT speedup by length](assets/figures/report_speedup_by_length.png)

## Slide 6: 数字解读：收益不是偶然命中

- 布局角色：data interpretation
- 页面意图：把主实验表格中的关键数字转成易读指标，说明 S2 的稳定性和端到端影响。
- Key points:
  - 16K Coding：S0 37.443s，S2 0.334s，speedup 112.2x
  - 16K RAG：S0 25.341s，S2 0.244s，speedup 103.7x
  - S2 的 p95 接近中位数，说明 warm turn 中持续稳定命中
  - TTFT 降低后，长回答场景的瓶颈会转向 decode 吞吐
- Visual idea: 用 2 组大数字 KPI + “prefill 瓶颈转移到 decode”的简化阶段条。
- Required images: 无

## Slide 7: 动态字段位置：前置会破坏缓存命中

- 布局角色：comparison / data evidence
- 页面意图：说明同样打开缓存时，动态字段位置是决定性变量。
- Key points:
  - `front_volatile` 会把 8K warm TTFT 拉回 9-12 秒量级
  - `middle_volatile` 只能复用前半部分，收益有限，约 1.6x
  - `end_volatile` 和 `stable_prefix` 接近理想状态，约 0.15-0.18 秒
  - 动态信息可以存在，但不能打断可复用前缀
- Visual idea: 左侧放布局敏感性图，右侧用 prompt 条形图对比前置/后置动态块。
- Required images:
  - Main evidence figure; strict input asset; preserve all chart data, axes, labels, legends, colors, and values.

    ![Layout sensitivity at 8K](assets/figures/report_layout_sensitivity_8k.png)

## Slide 8: `--cache-reuse`：边际调参不是主矛盾

- 布局角色：parameter sweep
- 页面意图：展示在稳定前缀已建立时，reuse 参数只带来小幅变化。
- Key points:
  - 8K 中非零 reuse 反而普遍慢约 10%-11%
  - 16K 中最佳 reuse 只快约 2%-3%
  - 单会话、单并发、稳定前缀下，主要收益已经由 prompt 组织方式获得
  - 调优顺序应先保证稳定前缀，再考虑 reuse、slot 和上下文上限
- Visual idea: 使用扫描图作为中心证据，配一个“先结构、后参数”的优先级标尺。
- Required images:
  - Main evidence figure; strict input asset; preserve all chart data, axes, labels, legends, colors, and values.

    ![Cache reuse sweep](assets/figures/report_cache_reuse_sweep.png)

## Slide 9: 资源开销：缓存要和命中收益匹配

- 布局角色：resource tradeoff / data evidence
- 页面意图：解释 RSS、缓存状态和 TTFT 收益之间的关系。
- Key points:
  - S2 峰值 RSS 从约 2.28 GiB 增至约 3.36 GiB，资源开销可控
  - S1 内存占用更高，但 TTFT 没有改善，说明缓存状态未转化为收益
  - RSS 包含权重、Metal 缓冲区、运行时分配和缓存管理状态
  - 关键不是“占用更多缓存”，而是缓存命中是否稳定
- Visual idea: 内存图 + “开销是否换来收益”的双轴解释卡片。
- Required images:
  - Main evidence figure; strict input asset; preserve all chart data, axes, labels, legends, colors, and values.

    ![Memory by strategy](assets/figures/report_memory_by_strategy.png)

## Slide 10: 工程结论：把 prompt 组织当作性能接口

- 布局角色：summary / recommendation
- 页面意图：给出可落地的 Agent prompt 组装建议。
- Key points:
  - 把系统指令、工具 schema、仓库摘要、固定文档和长期记忆放在 prompt 前部
  - 保持稳定前缀跨轮字节级稳定，避免随机顺序 JSON、时间戳和计数器混入前缀
  - 把当前时间、request id、turn id、最新错误、工具返回和用户请求集中后置
  - 优化顺序：稳定前缀协议优先，然后再扫描 cache-reuse、并发 slot、上下文长度和模型量化
- Visual idea: 一张“推荐 prompt 模板”示意图，以绿色标注稳定前缀、以橙色标注动态后缀。
- Required images: 无

## Required Source Image Mapping

- Slide 5: `assets/figures/report_ttft_by_length.png` and `assets/figures/report_speedup_by_length.png`
- Slide 7: `assets/figures/report_layout_sensitivity_8k.png`
- Slide 8: `assets/figures/report_cache_reuse_sweep.png`
- Slide 9: `assets/figures/report_memory_by_strategy.png`

请确认这些图表与页码映射是否正确。确认后再进入视觉风格选择、图片后端确认和样张生成阶段。
