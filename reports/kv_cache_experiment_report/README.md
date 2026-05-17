# 本地 LLM Agent KV Cache 复用实验报告

本目录是中文实验报告工程，当前已移除原始技术报告模板依赖，使用普通 `article` 文档类和本地 `config.tex` 配置。

编译命令：

```bash
latexmk -interaction=nonstopmode main.tex
```

本目录包含 `.latexmkrc`，会强制 `latexmk` 使用 XeLaTeX。不要用 `pdflatex` 编译，因为报告依赖 `fontspec` 和 `xeCJK`。

主要文件：

- `main.tex`: 中文报告正文。
- `config.tex`: XeLaTeX 中文字体与模板配置。
- `figures/`: 从 `results/figures/report/` 复制的实验图。
