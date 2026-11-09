# 20261109_TTT checkpoint 评估入口

创建时间：20261109 00:14:58 Asia/Shanghai

最后修订时间：20261109 12:18:53 Asia/Shanghai

## 入口与数据

- 为 LaCT、E²-TTT、In-Place TTT 各提供 `reproductions/<method>/eval.sh`，保留各方法的模型实现与原始评分。
- 显式传入 `--checkpoint` 与 `--output-dir`；本地数据默认采用评估数据准备文档约定的目录，并允许指定实际路径。
- LaCT 评估六项 RULER，默认 4K／8K／16K／32K、128-token 输出；E²-TTT 评估两项 RULER、SQuAD completion 和 SWDE；In-Place TTT 评估十三项 RULER、4K–256K、64-token 输出。
- RULER 使用 `input + answer_prefix`，不添加聊天模板或截断；E²-TTT 的 SQuAD／SWDE 保留原任务的 prompt、生成与计分设置。
- 输出保存到各方法独立的 `artifacts/reproductions/<method>/eval/`，保留运行配置、逐样本预测与汇总，不上传 SwanLab。
- 不新增测试。GPU 入口允许单卡 0 或 1，默认 0；当前按用户授权，显式使用 `CUDA_VISIBLE_DEVICES=0`，代码先经 GitHub 同步。本轮执行计划与结果见 [GPU 验证与评估](20261109_checkpoint_gpu_evaluation.md)。

## 方法设置

- LaCT：沿用原模型的 `use_cache=False` 完整前缀生成，每 token 重算历史 chunk；FP32 参数配合 BF16 autocast。严格按 checkpoint 配置加载并拒绝缺失或不匹配的权重。三个发布目录的名字与实际 head 数相反：Momentum 实际为 2 heads，两个 Muon 实际为 4 heads。
- E²-TTT：复用随附 harness 的模型 wrapper、原 SQuAD／SWDE task 及 RULER 评分。SQuAD／SWDE 沿用 2048 总长度上限，预留 48 输出后 prompt 最多 2000，超出时按原协议左截断；RULER 不截断。显式传入本地 task dict，避免经过仅适用于已注册 YAML task 的名称查找。
- In-Place TTT：顺序调用原 OpenCompass 模型 wrapper、预测后处理和十三项评分器；按所选最大长度设置运行上限，配合逐样本预算检查，避免 256K 被原 128K 上限截断。当前第三方 32K 权重的更长输入属于长度外推，不额外引入 YaRN。

## 运行说明

完整命令与方法环境分别见 [LaCT](lact/README.md#本地-ruler-评估)、[E²-TTT](e2_ttt/README.md#环境与评估) 和 [In-Place TTT](in_place_ttt/README.md#ruler-评估入口)。数据下载与构造见 [评估数据准备](20261108_ttt_data_preparation.md)。

入口记录 checkpoint、实际数据路径和生成设置，保存逐样本预测及完整结果。三个服务器独立评估环境已安装，依赖检查见[评估环境](20261109_environment_setup.md)。GPU checkpoint 加载、真实样本生成与原生计分的验证结果见 [GPU 执行记录](20261109_checkpoint_gpu_evaluation.md)；完整 benchmark 指标仍需完成全量评估后报告。
