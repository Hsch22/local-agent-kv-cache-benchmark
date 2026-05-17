# 本地 Agent KV Cache Benchmark 可执行计划

更新时间：2026-05-13  
项目目录：`/Users/sicheng/Desktop/local-agent-kv-cache-benchmark`

## 0. 当前执行状态

本计划已按本机情况落地为可执行 benchmark 项目。当前采用的正式实验约束如下：

1. 主实验和后续补充实验都固定单并发 `--parallel 1`。
2. 后续新增实验统一使用 `repeat=1`，避免长时间运行和散热状态变化带来的额外干扰。
3. 2K 补充实验仍按 S0 / S1 / S2 主矩阵执行，但不再重复 3 到 5 次。
4. cache-reuse sweep 按原计划覆盖 `coding`、`rag` 和 `8K`、`16K`，每组 `repeat=1`。

已完成：

| 实验 | 范围 | 状态 |
| --- | --- | --- |
| 环境准备 | `.venv`、依赖、`llama.cpp`、模型、workload、脚本 | 完成 |
| Smoke test | coding 轻量验证 | 完成，结果已按要求删除 |
| Coding 主实验 | 4K / 8K / 16K，S0 / S1 / S2 | 完成 |
| RAG 主实验 | 4K / 8K / 16K，S0 / S1 / S2 | 完成 |
| Layout sensitivity | 8K，coding + rag，4 种 layout | 完成 |
| 2K 主实验补充 | coding + rag，S0 / S1 / S2 | 完成 |
| Cache-reuse sweep | coding + rag，8K / 16K，reuse 0 / 64 / 128 / 256 / 512 | 完成 |

当前主交付范围已完成：

| 实验 | 范围 | repeat |
| --- | --- | --- |
| 上下文长度和策略实验 | coding + rag，2K / 4K / 8K / 16K，S0 / S1 / S2 | 1；其中早期 coding 4K / 8K 为 3 |
| Layout sensitivity | coding + rag，8K，4 种 layout | 1 |
| Cache-reuse sweep | coding + rag，8K / 16K，5 个 reuse 值 | 1 |

扩展实验 32K、7B、多会话交替请求和 MLX-LM 对照仍作为选做项，不进入当前主交付范围。

## 1. 实验目标

本实验要回答一个具体问题：

> 在本地 LLM Agent 服务中，当多轮请求共享大量 system prompt、工具定义、代码上下文或文档上下文时，Prompt / KV Cache 复用策略会怎样影响首 token 延迟、总延迟、吞吐和内存占用？

最终输出应能回答：

1. 长上下文 Agent 请求里，重复 prefill 的开销有多大。
2. prompt cache 对 TTFT，也就是 first token latency，有多少改善。
3. 哪些 prompt 组织方式会破坏缓存复用。
4. `--cache-reuse` 在本地服务里是否有实际收益。
5. 对本地 coding agent / RAG agent，应该如何组织 prompt 才更适合缓存复用。

本计划以本机可稳定完成为优先级，不把 32K 长上下文和 7B 模型作为第一阶段硬性目标。

## 2. 本机环境基线

已确认的本机情况：

| 项目 | 本机情况 |
| --- | --- |
| 机器 | MacBook Pro, Mac17,2 |
| 芯片 | Apple M5 |
| CPU | 10 核，4 performance + 6 efficiency |
| GPU | Apple M5 10 核 GPU |
| Metal | Supported |
| 内存 | 32 GB unified memory |
| 系统 | macOS 26.4.1, build 25E253 |
| 架构 | arm64 |
| 可用磁盘 | 约 1.4 TiB |
| Homebrew | `/opt/homebrew/bin/brew`，已安装 |
| uv | `uv 0.11.8`，已安装 |
| Conda | `/opt/miniconda3/bin/conda`，已安装 |
| C/C++ 编译器 | Apple clang 21.0.0，已安装 |
| 项目 Python 环境 | `.venv`，已用 uv 创建并安装 benchmark 依赖 |
| `.venv` Python | Python 3.11.14 |
| llama.cpp | 已安装，`/opt/homebrew/bin/llama-server`，version 9110 |
| 本地 GGUF 模型 | 已下载到 `models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` |

项目环境已创建：

```bash
cd /Users/sicheng/Desktop/local-agent-kv-cache-benchmark
source .venv/bin/activate
python --version
```

预期输出：

```text
Python 3.11.14
```

## 3. 本机适合的实验范围

32 GB unified memory 足够做 3B 量化模型的 2K、4K、8K、16K 上下文实验，也可以尝试 32K，但 32K 不放入主实验矩阵，避免内存压力、换页和散热导致结果不稳定。

推荐实验范围：

| 阶段 | 模型 | 上下文长度 | 目标 |
| --- | --- | --- | --- |
| Smoke test | 3B Q4 GGUF | 2K、4K | 验证工具链和指标采集 |
| 主实验 | 3B Q4 或 Q5 GGUF | 2K、4K、8K、16K | 形成核心结论 |
| 选做扩展 | 3B Q4 GGUF | 32K | 验证长上下文极限 |
| 选做扩展 | 7B Q4 GGUF | 2K、4K、8K | 比较模型规模影响 |

本机优先使用 Apple Metal，因此主实验工具选择 `llama.cpp server`。MLX-LM 可以作为后续扩展，不进入第一版主计划。

## 4. 需要安装和准备的组件

### 4.1 Python 依赖

项目当前只有文档，没有 requirements 文件。建议先把 benchmark 脚本需要的依赖安装进项目 `.venv`：

```bash
cd /Users/sicheng/Desktop/local-agent-kv-cache-benchmark
source .venv/bin/activate
uv pip install openai psutil pandas matplotlib seaborn tqdm requests
```

依赖用途：

| 依赖 | 用途 |
| --- | --- |
| `openai` | 调用 llama.cpp OpenAI-compatible API |
| `psutil` | 采集服务进程内存、CPU 信息 |
| `pandas` | 汇总 CSV 结果 |
| `matplotlib`、`seaborn` | 画图 |
| `tqdm` | 进度条 |
| `requests` | 健康检查和 metrics 拉取 |

### 4.2 llama.cpp

本机已通过 Homebrew 安装 `llama.cpp`。重装或在新机器复现时使用：

```bash
brew install llama.cpp
```

安装后验证：

```bash
command -v llama-server
llama-server --help | rg 'cache-prompt|cache-reuse|metrics|parallel|ctx|gpu'
```

如果 Homebrew 安装后没有 `llama-server`，再改用源码构建：

```bash
mkdir -p /Users/sicheng/Desktop/tools
cd /Users/sicheng/Desktop/tools
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 10
```

然后使用：

```bash
/Users/sicheng/Desktop/tools/llama.cpp/build/bin/llama-server --help
```

### 4.3 模型文件

本机已下载主实验模型。模型信息如下：

| 项目 | 值 |
| --- | --- |
| Repo | `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` |
| File | `qwen2.5-coder-3b-instruct-q4_k_m.gguf` |
| Local path | `/Users/sicheng/Desktop/local-agent-kv-cache-benchmark/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` |
| SHA256 | `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7` |

主实验模型选择：

| 优先级 | 模型 | 量化 | 用途 |
| --- | --- | --- | --- |
| 1 | Qwen2.5-Coder-3B-Instruct GGUF | Q4_K_M | 主实验默认 |
| 2 | Qwen2.5-Coder-3B-Instruct GGUF | Q5_K_M | 如果速度和内存都稳定 |
| 3 | Llama-3.2-3B-Instruct GGUF | Q4_K_M | 通用 RAG 对照 |
| 4 | Qwen2.5-Coder-7B-Instruct GGUF | Q4_K_M | 选做扩展 |

建议模型放在：

```text
models/
  qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

后续所有脚本通过环境变量读取模型路径：

```bash
export MODEL_PATH=/Users/sicheng/Desktop/local-agent-kv-cache-benchmark/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

## 5. 推荐目录结构

当前项目只有 `docs/plan.md`。建议后续补齐：

```text
local-agent-kv-cache-benchmark/
  .venv/
  docs/
    plan.md
  models/
    qwen2.5-coder-3b-instruct-q4_k_m.gguf
  workloads/
    coding_2k.jsonl
    coding_4k.jsonl
    coding_8k.jsonl
    coding_16k.jsonl
    rag_2k.jsonl
    rag_4k.jsonl
    rag_8k.jsonl
    rag_16k.jsonl
  scripts/
    make_workloads.py
    run_benchmark.py
    run_suite.py
    plot_results.py
  results/
    raw/
    summary/
    figures/
  logs/
```

## 6. 被比较的缓存策略

主实验比较 4 组策略：

| 编号 | 策略 | llama-server 参数 | 说明 |
| --- | --- | --- | --- |
| S0 | No Cache | `--no-cache-prompt` | baseline，每轮重复 prefill |
| S1 | Default Cache | `--cache-prompt` | 只开缓存，不优化 prompt |
| S2 | Stable Prefix | `--cache-prompt` | 稳定内容放前面，动态内容放最后 |
| S3 | Cache Reuse Sweep | `--cache-prompt --cache-reuse N` | 在 S2 上测试不同 reuse 值 |

`S3` 的 `N` 取值：

```text
0, 64, 128, 256, 512
```

如果实际 `llama-server --help` 中参数名有变化，以本机 help 输出为准，并在实验记录里写明实际命令。

## 7. llama-server 启动命令

所有实验先固定为单并发，避免多 slot 调度影响缓存判断：

```bash
export MODEL_PATH=/Users/sicheng/Desktop/local-agent-kv-cache-benchmark/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

### S0: no cache

```bash
llama-server \
  -m "$MODEL_PATH" \
  -c 16384 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 1 \
  --no-cache-prompt \
  --metrics
```

### S1 / S2: prompt cache

```bash
llama-server \
  -m "$MODEL_PATH" \
  -c 16384 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 1 \
  --cache-prompt \
  --metrics
```

### S3: cache reuse sweep

```bash
llama-server \
  -m "$MODEL_PATH" \
  -c 16384 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 1 \
  --cache-prompt \
  --cache-reuse 128 \
  --metrics
```

上下文长度实验中，`-c` 必须不小于目标 prompt length + max new tokens + 少量余量：

| 目标 prompt 长度 | 推荐 `-c` |
| --- | --- |
| 2K | 4096 |
| 4K | 8192 |
| 8K | 12288 或 16384 |
| 16K | 20480 或 24576 |
| 32K 选做 | 32768 或 40960 |

为减少变量，主实验可统一使用 `-c 16384` 覆盖 2K 到 16K；如果 16K prompt 加 64 输出超过上下文，再切到 `-c 20480`。

## 8. Workload 设计

本项目当前没有真实代码库，因此第一版 workload 使用可控合成数据，保证不同策略之间唯一变化是 prompt 布局和共享前缀比例。

### 8.1 Coding Agent workload

模拟本地 coding agent 请求：

```text
[System Prompt]
你是一个本地代码助手，回答要简洁、准确。

[Tool Definitions]
read_file(path)
grep_search(query)
edit_file(path, patch)
run_tests(command)

[Repository Summary]
合成项目结构、模块说明、依赖说明。

[Relevant Files]
合成 main.py、utils.py、tests/test_main.py 内容。

[Conversation Summary]
前几轮完成了哪些分析和修改。

[Dynamic Metadata]
current_time、request_id、turn_id、错误日志。

[Current User Request]
本轮用户问题。
```

20 轮请求建议：

| 轮次 | 请求类型 |
| --- | --- |
| 1 | 解释模块职责 |
| 2 | 找潜在 bug |
| 3 | 设计修复方案 |
| 4 | 修改函数 |
| 5 | 添加单元测试 |
| 6 | 根据失败测试继续修复 |
| 7 | 总结改动 |
| 8-20 | 重复代码审查、重构、测试、解释类任务 |

### 8.2 RAG workload

模拟长文档问答：

```text
[System Prompt]
你是严谨的文档问答助手，只能基于给定文档回答。

[Document Context]
固定长文档，按目标长度扩展到 2K、4K、8K、16K。

[Question]
每轮只改变问题。
```

20 个问题覆盖：

| 类型 | 示例 |
| --- | --- |
| 摘要 | 总结文档核心结论 |
| 定位 | 找出某个指标的定义 |
| 对比 | 比较两个策略的差异 |
| 推理 | 根据文档判断某种配置是否合理 |
| 引用 | 要求回答时说明依据来自哪个段落 |

### 8.3 Prompt layout

重点比较 4 种布局：

| 布局 | 说明 | 预期 |
| --- | --- | --- |
| `front_volatile` | 时间戳、request id 放最前面 | 最容易破坏缓存 |
| `middle_volatile` | 动态错误日志放在中间 | 部分破坏缓存 |
| `end_volatile` | 动态内容放最后 | 缓存收益明显 |
| `stable_prefix` | system、tools、repo/doc 全部稳定且前置 | 最推荐 |

同一组内容只改变动态字段位置，避免把内容差异误认为缓存差异。

## 9. 实验矩阵

### 9.1 Smoke test

目标：确认环境、模型、server、client、CSV 输出都可用。

| 维度 | 设置 |
| --- | --- |
| 模型 | Qwen2.5-Coder-3B Q4 GGUF |
| workload | coding |
| prompt length | 2K、4K |
| 策略 | S0、S2 |
| layout | `stable_prefix` |
| 重复 | 每组 2 次 |
| max new tokens | 32 |

通过标准：

1. 每个请求都能完成。
2. CSV 有 TTFT、total latency、completion token 数。
3. server 没有 OOM 或上下文溢出。
4. S2 的第 2 轮以后 TTFT 应该低于 S0；如果没有，先检查 prompt 是否真的共享前缀。

### 9.2 主实验一：上下文长度和策略

目标：形成核心结果图。

| 维度 | 设置 |
| --- | --- |
| workload | coding、rag |
| prompt length | 2K、4K、8K、16K |
| 策略 | S0、S1、S2 |
| layout | S1 用 `front_volatile`，S2 用 `stable_prefix` |
| 重复 | 每组 1 次，前期已完成的 coding 4K / 8K 使用过 repeat=3 |
| max new tokens | 64 |
| temperature | 0 |
| top_p | 1 |

规模估算：

```text
当前执行版为：

```text
2 workloads * 4 lengths * 3 strategies * 1 repeat = 24 sequences
```

该矩阵已完成。2K 为后续补充项，已按 `repeat=1` 跑完。
```

每个 sequence 包含 20 轮请求。统计时保留 turn 1 作为 cold cache 指标，turn 2-20 作为 warm cache 指标。

### 9.3 主实验二：动态内容位置

目标：验证 prompt layout 对缓存复用的影响。

| 维度 | 设置 |
| --- | --- |
| workload | coding、rag |
| prompt length | 8K |
| 策略 | `--cache-prompt` |
| layout | `front_volatile`、`middle_volatile`、`end_volatile`、`stable_prefix` |
| 重复 | 每组 1 次 |
| max new tokens | 64 |

关键对比：

```text
front_volatile vs stable_prefix
```

如果内容基本相同但 TTFT 差异明显，就能支持结论：prompt cache 的收益高度依赖稳定前缀长度。

### 9.4 主实验三：cache-reuse sweep

目标：判断本机、本模型、本 workload 下 `--cache-reuse` 是否值得启用。

| 维度 | 设置 |
| --- | --- |
| workload | coding、rag |
| prompt length | 8K、16K |
| layout | `stable_prefix` |
| cache-reuse | 0、64、128、256、512 |
| 重复 | 每组 1 次 |
| max new tokens | 64 |

规模估算：

```text
当前执行版为：

```text
2 workloads * 2 lengths * 5 reuse values * 1 repeat = 20 sequences
```

该矩阵已完成。`coding 8K` 最先完成；`rag 8K`、`coding 16K`、`rag 16K` 已在后续补齐。
```

如果不同 `cache-reuse` 值差异很小，就把结论写成“在单并发、稳定前缀、多轮顺序请求下收益不明显”；如果差异明显，再分析最佳值是否随上下文长度变化。

## 10. 指标采集

核心指标：

| 指标 | 含义 | 采集方法 |
| --- | --- | --- |
| `ttft_s` | 请求发出到第一个非空 token 的时间 | Python streaming client |
| `total_latency_s` | 请求总耗时 | Python client |
| `decode_latency_s` | 第一个 token 后到结束 | `total - ttft` |
| `completion_tokens` | 输出 token 数 | API usage，缺失时用近似计数 |
| `prompt_tokens` | prompt token 数 | API usage，缺失时记录目标长度 |
| `tokens_per_s` | decode 吞吐 | `completion_tokens / decode_latency_s` |
| `server_rss_mb` | server RSS 内存 | `psutil.Process(pid).memory_info().rss` |
| `server_cpu_pct` | server CPU | `psutil` |
| `cache_state` | cold / warm | turn 1 为 cold，turn 2-20 为 warm |

CSV 字段建议：

```text
run_id, timestamp, machine, model_path, model_name,
strategy, cache_prompt, cache_reuse, workload, layout,
prompt_len_target, turn_id, repeat_id, cache_state,
max_tokens, temperature, top_p,
ttft_s, total_latency_s, decode_latency_s,
prompt_tokens, completion_tokens, tokens_per_s,
server_pid, server_rss_mb, server_cpu_pct,
error
```

## 11. 脚本设计

### 11.1 `scripts/make_workloads.py`

职责：

1. 生成 coding 和 RAG 两类 workload。
2. 控制目标 prompt 长度：2K、4K、8K、16K。
3. 控制 layout：`front_volatile`、`middle_volatile`、`end_volatile`、`stable_prefix`。
4. 输出 JSONL，每行一个请求。

输出示例：

```json
{
  "workload": "coding",
  "prompt_len_target": 8192,
  "layout": "stable_prefix",
  "turn_id": 7,
  "messages": [
    {"role": "system", "content": "stable system + tools + repo context ..."},
    {"role": "user", "content": "dynamic metadata + current request ..."}
  ]
}
```

长度控制先用近似 token 估算即可：

```text
英文和代码按 4 characters/token 粗估；
中文按 1.5 到 2 characters/token 粗估；
最终记录 prompt_len_target，真实 token 数以后用 tokenizer 或 API usage 校正。
```

### 11.2 `scripts/run_benchmark.py`

职责：

1. 读取一个 workload JSONL。
2. 以 OpenAI-compatible API 调用本地 `llama-server`。
3. 使用 streaming 测 TTFT。
4. 每轮请求后记录延迟、token、内存。
5. 输出单个 CSV。

核心计时逻辑：

```python
start = time.perf_counter()
first_token_time = None

stream = client.chat.completions.create(
    model="local",
    messages=messages,
    temperature=0,
    top_p=1,
    max_tokens=64,
    stream=True,
)

for event in stream:
    text = extract_delta_text(event)
    if text and first_token_time is None:
        first_token_time = time.perf_counter()
    collect(text)

end = time.perf_counter()
ttft_s = first_token_time - start if first_token_time else None
total_latency_s = end - start
```

### 11.3 `scripts/run_suite.py`

职责：

1. 根据实验矩阵启动不同的 `llama-server`。
2. 等待 server health ready。
3. 运行对应 workload。
4. 结束 server。
5. 写入 `results/raw/`。

每个 strategy / context size 单独重启 server，避免缓存状态跨配置污染。

推荐流程：

```text
start server
wait /health or /v1/models
run 20-turn sequence
record metrics
stop server
sleep 5 seconds
next config
```

### 11.4 `scripts/plot_results.py`

输出图：

| 文件 | 内容 |
| --- | --- |
| `figures/ttft_by_length.png` | prompt length vs warm TTFT |
| `figures/speedup_by_length.png` | prompt length vs TTFT speedup |
| `figures/layout_sensitivity.png` | layout vs warm TTFT |
| `figures/cache_reuse_sweep.png` | cache-reuse value vs TTFT |
| `figures/memory_by_strategy.png` | strategy vs peak RSS |

默认统计：

```text
按 strategy, workload, prompt_len_target, layout 分组；
turn 1 单独作为 cold；
turn 2-20 计算 median、p95、mean；
主要报告 median 和 p95。
```

## 12. 执行顺序

### 第 0 阶段：环境准备

已完成：

```bash
uv venv .venv --python /opt/miniconda3/envs/tunnel-agent/bin/python
uv pip install --python .venv/bin/python openai psutil pandas matplotlib seaborn tqdm requests huggingface-hub hf-xet
brew install llama.cpp
mkdir -p models workloads scripts results/raw results/summary results/figures logs
.venv/bin/python scripts/download_model.py --local-dir models
```

当前模型路径：

```bash
export MODEL_PATH=/Users/sicheng/Desktop/local-agent-kv-cache-benchmark/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

已生成 workload：

```text
workloads/*.jsonl
2 workloads * 4 lengths * 4 layouts = 32 files
```

### 第 1 阶段：Smoke test

1. 生成 2K、4K coding workload。
2. 用 S0 跑 2 次。
3. 用 S2 跑 2 次。
4. 检查 CSV 是否完整。
5. 检查 S2 warm TTFT 是否低于 S0。

通过后再进入主实验。

### 第 2 阶段：主实验

执行顺序：

1. 上下文长度和策略实验。
2. 动态内容位置实验。
3. cache-reuse sweep。
4. 生成 summary CSV。
5. 画图。

当前执行进度：

| 步骤 | 状态 |
| --- | --- |
| 4K / 8K / 16K 上下文长度和策略实验 | 已完成 |
| 8K layout sensitivity | 已完成 |
| coding 8K cache-reuse sweep | 已完成 |
| 2K 上下文长度和策略实验 | 已完成 |
| rag 8K、coding 16K、rag 16K cache-reuse sweep | 已完成 |

为了减少散热和后台任务影响，建议：

1. 接电源。
2. 关闭占用 GPU/CPU 的应用。
3. 每组配置之间 sleep 5 到 10 秒。
4. 每次实验记录开始时间和结束时间。
5. 如果 `server_rss_mb` 超过 26000 或系统出现明显换页，停止 32K 或 7B 实验。

### 第 3 阶段：扩展实验

只在主实验结果稳定后做：

1. 32K context。
2. 7B Q4 模型。
3. 多会话交替请求。
4. MLX-LM prompt cache 对照。

## 13. 结果分析方法

### 13.1 TTFT 加速比

```text
ttft_speedup = median_ttft_no_cache / median_ttft_strategy
```

示例表：

| strategy | median warm TTFT | speedup |
| --- | ---: | ---: |
| S0 no cache | 12.0s | 1.0x |
| S1 default cache | 7.5s | 1.6x |
| S2 stable prefix | 2.1s | 5.7x |

### 13.2 总延迟加速比

```text
latency_speedup = median_latency_no_cache / median_latency_strategy
```

如果输出固定为 64 tokens，总延迟差异主要来自 prefill；如果输出很长，decode 会稀释 cache 的总延迟收益。

### 13.3 布局敏感性

核心对比：

```text
front_volatile vs stable_prefix
```

这两个配置内容接近，只是动态字段位置不同。如果 TTFT 差距明显，结论可以写成：

> Prompt cache 不是简单的开关优化；稳定前缀长度本身决定了可复用 KV cache 的规模。

### 13.4 上下文长度曲线

预期趋势：

| prompt length | 预期现象 |
| --- | --- |
| 2K | cache 收益存在但不大 |
| 4K | TTFT 差距开始明显 |
| 8K | stable prefix 明显优于 no cache |
| 16K | no cache 的 prefill 成本突出 |
| 32K | 可能出现内存、换页或散热干扰 |

## 14. 风险和处理

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| Homebrew 安装的 llama.cpp 没有 `llama-server` | `command -v llama-server` 失败 | 源码构建 |
| 参数名变化 | `--cache-reuse` 或 `--cache-prompt` 不存在 | 以 `llama-server --help` 为准，记录实际参数 |
| GGUF 模型下载慢或失败 | 没有模型文件 | 先下载更小的 1.5B/3B Q4 模型做 smoke test |
| API usage 缺 prompt token | CSV 里 token 为空 | 先用目标长度，后续用 tokenizer 补 |
| S2 没有加速 | warm TTFT 接近 S0 | 检查每轮 messages 是否在开头有变化内容 |
| 内存压力 | RSS 高、系统卡顿、延迟抖动大 | 降低 context，先停 32K/7B |
| 热降频 | 后半段所有配置变慢 | 每组间隔休息，重复时随机化配置顺序 |
| 多 slot 污染 | 缓存行为难解释 | 主实验固定 `--parallel 1` |

## 15. 最小可交付版本

如果只做一个能写成报告的版本，范围收敛为：

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen2.5-Coder-3B Q4 GGUF |
| 工具 | llama.cpp server + Metal |
| workload | Coding Agent |
| prompt length | 4K、8K、16K |
| 策略 | S0 no cache、S1 default cache、S2 stable prefix |
| 重复 | 每组 1 次；需要更强统计稳定性时再追加 repeat |
| 输出 | 64 tokens |
| 指标 | TTFT、total latency、tokens/s、RSS memory |

最关键图：

```text
不同上下文长度下，S0 / S1 / S2 的 warm TTFT 对比。
```

只要这张图显示 stable prefix 明显降低 TTFT，就足以支撑第一版结论。

## 16. 立即下一步

环境、脚本、workload、llama.cpp、模型文件、smoke test、主实验矩阵、layout sensitivity 和 cache-reuse sweep 都已经完成。2K 主实验执行命令为：

```bash
cd /Users/sicheng/Desktop/local-agent-kv-cache-benchmark
source .venv/bin/activate
export MODEL_PATH=/Users/sicheng/Desktop/local-agent-kv-cache-benchmark/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
python scripts/run_suite.py \
  --mode main \
  --workloads coding rag \
  --lengths 2k \
  --strategies S0 S1 S2 \
  --run-label formal_2k_r1 \
  --repeat 1 \
  --max-tokens 64 \
  --cooldown-s 8 \
  --model-path "$MODEL_PATH"
```

cache-reuse sweep 补充执行命令模板为：

```bash
python scripts/run_suite.py \
  --mode main \
  --workloads rag \
  --lengths 8k \
  --strategies S2 \
  --layouts stable_prefix \
  --run-label formal_reuse8k_r1 \
  --repeat 1 \
  --max-tokens 64 \
  --cache-reuse 0 \
  --cooldown-s 8 \
  --model-path "$MODEL_PATH"
```

`--cache-reuse` 依次替换为 `0`、`64`、`128`、`256`、`512`；16K 的 `coding` 和 `rag` 同理运行，`--lengths 16k`，run label 使用 `formal_reuse16k_r1`。
