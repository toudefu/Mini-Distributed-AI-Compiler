# Mini Distributed AI Compiler

一个面向学习与研究的分布式 AI 编译器骨架，Python 优先、CPU-first，未来扩展到 GPU 和 MLIR/C++。目前仅为 v0.0.1 scaffold，核心算法尚未实现。

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,cpu]"
mini-dist --version
python -m pytest
```

基础包不依赖 GPU 或深度学习框架。测试仅验证框架可导入以及未实现入口正确报错。

## Architecture

[架构设计](docs/architecture.md) · [学习路线](docs/roadmap.md)

核心目录位于 `src/mini_dist/`，包含 frontend、IR、ops、distributed、passes、parallel、analysis、hardware、cost_model、scheduler、codegen、runtime、autotune 和 profiling。每个模块包含职责说明与 TODO。

首个目标：在 CPU 上将一个 MatMul 按 K 轴分片，生成各 rank 的局部 MatMul 与 AllReduce，并对比 NumPy reference。

`configs/virtual_cluster.json` 为假设硬件参数，不代表实测性能。真实 GPU 后端尚未实现。
