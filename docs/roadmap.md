# 学习路线

| 阶段 | 实现内容 | 验收 |
|---|---|---|
| M1 | TensorType、Value、Operation、Graph、打印、验证 | 手工构建 MatMul + Add IR |
| M2 | Shape inference、PassManager、NumPy reference | 单设备结果与 NumPy 一致 |
| M3 | DeviceMesh、Shard/Replicate/Partial、CPU 模拟器 | 切分与重组可逆 |
| M4 | Sharding propagation、SPMD partition、collective insertion | K 轴 MatMul + AllReduce 等于 reference |
| M5 | Reshard、DP/TP、MLP | 多种分片策略结果一致 |
| M6 | 拓扑、通信与计算代价、策略搜索 | 可复现地选择低估算代价方案 |
| M7 | DAG 调度、流水线、通信融合 | 验证依赖和模拟 makespan |
| M8 | PyTorch 导入、Gloo 运行时 | CPU 多进程集成测试 |
| M9 | PP/EP、MoE、训练扩展 | AllToAll 与负载均衡正确性 |
| M10 | TVM/Triton/MLIR/GPU 后端 | 真实硬件正确性与 benchmark |

建议逐阶段实现，不必一次实现所有目录。M1-M5 无需 GPU。
