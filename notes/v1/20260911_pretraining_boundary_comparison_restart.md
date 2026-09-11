# 20260911_双卡预训练边界对比实验（16:27:19 UTC+08:00）

创建时间：20260911 16:27:19 UTC+08:00
最后修订时间：20260911 16:27:19 UTC+08:00

## 实验设置

semantic、random、mixed 三组从相同模型种子重新初始化，依次使用物理 GPU 0、1 进行同步数据并行训练。每卡 microbatch 2，梯度累积 2，全局有效 batch 8；关闭梯度检查点，BF16。每组 20,000 步，模型种子 42，数据种子 20260907。学习率峰值 3e-5，600 步 warmup，余弦衰减至 3e-6；长度课程为前 6,000 步，压缩率 2/4/8 等概率。AE/LM 权重均为 1。

沿用 `data/v1/boundary-comparison-2048-20260910`，每组训练集 157,320 条，AE/LM 数量相等，mixed 两来源各占一半。输入和 LM 目标各不超过 2048 tokens。三组共享 semantic/random 两套 dev/test。每 1,000 步保存 checkpoint 并评估两套 dev，各取固定 120 条；每 2,000 步及初始基线进行小规模 AE 生成评估。每组结束后两卡分别评估两套 test，再启动下一组训练。

数据并行按全局 AE/LM 样本数归一化损失，按长度交错分配样本，每次参数更新前合并梯度。主进程写入指标和 checkpoint，保存全局采样状态以及每个 rank 的随机状态，恢复时校验 world_size、配置及数据身份。周期 dev 由主进程执行，另一进程等待。

## SwanLab 与产物

项目为 `latent-working-memory-v1`，group 为 `lwm-boundary-comparison-2048-20260911`。训练 run 名称为 `pretrain-semantic-157k-20260911`、`pretrain-random-157k-20260911`、`pretrain-mixed-157k-20260911`。测试 run 名称采用 `evaluate-训练来源-test-测试来源-157k-20260911`，其中 157k 仍表示对应模型的训练集规模，评估样本数记录在 config 中。job_type 为 train/evaluate，tags 保留 scope:main、method:latent-working-memory、study:boundary-comparison、data:fineweb 及实际来源。

配置：`configs/v1/pretrain_boundary_comparison_dual_a800.json`。训练产物：`artifacts/v1/experiments/boundary-comparison-2048-20260911/`。测试产物：`artifacts/v1/evaluations/boundary-comparison-2048-20260911/`。调度状态：`artifacts/v1/experiment-plans/boundary-comparison-2048-20260911/`。

本次重跑沿用现有数据集。原 20260910 系列本地与服务器训练产物和调度日志按用户要求清理；单卡、双卡性能测试报告和结果按用户要求清理，测试代码保留。云端旧记录的删除状态单独核实，不将新旧 run 混用。
