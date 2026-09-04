# 20260904 C-DIC MSC 训练代码验证（20260904 22:09:00 CST）

创建时间：20260904 22:00:30 CST（UTC+08:00）

最后修订时间：20260904 22:09:00 CST（UTC+08:00）

状态：训练代码与两轮真实 GPU autograd smoke test 通过；官方 MSC pilot 尚未运行

## 验证范围

- MSC v0.1 `session_4/train.txt` 解析；
- teacher-forced response loss；
- gold-response incremental compression；
- one-hop ra-TBPTT gradient path；
- frozen generator 与可训练 LoRA/compression-token 参数边界；
- episode-level optimizer step；
- checkpoint/resume schema；
- JSON config、metrics 和 memory trace。

## 官方数据检查

归档：`msc_v0.1.tar.gz`

SHA256：`e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60`

`session_4/train.txt` 解析结果：

- episodes：1001；
- source utterances：50,374；
- mean source utterances：50.3237；
- paired turns：25,126；
- odd-length session 的无配对尾 utterance：122；
- 空 utterance：1，规范化为 `__SILENCE__`。

episode 数与论文一致；utterance 均值低于论文报告的 53.3，暂不通过隐式补样本或修改原始数据追齐。

## 真实 GPU autograd smoke test

环境：NVIDIA A800-SXM4-80GB、PyTorch 2.0.1+cu118、Llama-2-7B-Chat、ICAE v1 checkpoint。

输入：1 个合成 episode、2 turns；retrieval threshold 设为 `-1.0`，保证第二轮沿同一 thread 执行 replace 和 one-hop gradient。

最终结果：

- mean loss：2.687118；
- gradient norm：0.808781；
- LoRA：128/128 trainable tensors 获得 gradient；
- compression tokens：1/1 tensor 获得 gradient；
- final memory states：1；
- optimizer step：1；
- 单 episode runtime：2.2277 秒，不含模型加载；
- peak allocated GPU memory：15,322,347,520 bytes，约 14.27 GiB；
- 进程正常退出。

真实 checkpoint 写入与恢复测试通过：checkpoint 包含 trainable model state、AdamW state、CPU/CUDA RNG state 和训练位置；单步 checkpoint 约 775 MiB。训练配置默认只保留最近两个 step checkpoints，避免长训练无限占用磁盘。

## 修复记录

首次测试发现启用 gradient checkpointing 后 LoRA gradient 为 0。原因是 ICAE 的 checkpoint branch 未把 `enable_lora` 传给 `LlamaDecoderLayer`，导致 compressor forward 退化为 base weights。已在 `reproductions/icae/src/icae/base/modeling_llama_icae.py` 补充该参数，并再次验证全部 128 个 LoRA tensors 获得 gradient。

query routing 使用临时 eval mode 和 `inference_mode()`，避免无梯度 query encoding 进入 gradient-checkpoint branch；gold-response compression 保持 train mode。

## 下一步

使用 `reproductions/cdic/configs/msc_pilot_a800.json` 运行官方 MSC 两 episode pilot，检查真实数据上的 loss、gradient coverage、peak GPU memory、slot growth、trace 和 checkpoint resume。
