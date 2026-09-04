# C-DIC 上游来源与复现边界（20260904 22:09:00 CST）

创建时间：20260904 16:19:08 CST（UTC+08:00）

最后修订时间：20260904 22:09:00 CST（UTC+08:00）

## 论文信息

- 标题：*Context-Driven Incremental Compression for Multi-Turn Dialogue Generation*
- 作者：Yeongseo Jung、Jaehyeok Kim、Eunseo Jung、Jiachuan Wang、Yongqi Zhang、Ka Chun Cheung、Simon See、Lei Chen
- 发表状态：已被 ICML 2026 接收
- arXiv：`2606.12411v1`
- 发布时间：`2026-06-10T17:59:54Z`

主要来源：

- https://arxiv.org/abs/2606.12411v1
- https://arxiv.org/pdf/2606.12411v1

## MSC 数据来源

- 数据版本：ParlAI `msc_v0.1`；
- 官方归档：`https://parl.ai/downloads/msc/msc_v0.1.tar.gz`；
- SHA256：`e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60`；
- 主训练输入：`msc/msc_dialogue/session_4/train.txt`，共 1001 records。

当前 loader 直接读取官方 JSONL，不依赖 ParlAI runtime。原始数据中的空 utterance、无配对尾项和解析统计会写入 training artifact。

## ICAE 兼容修复

ICAE v1 的 gradient-checkpoint branch 原先未把 `enable_lora` 传入 decoder layer。启用 gradient checkpointing 时会导致 compressor LoRA 不参与 forward，表现为训练可运行但全部 LoRA gradients 缺失。本项目在 `reproductions/icae/src/icae/base/modeling_llama_icae.py` 中补充参数转发；两轮 A800 autograd smoke test 已确认 128/128 LoRA tensors 获得 gradient。

## 代码开放状态

截至 20260904 16:19 CST，论文和 arXiv source 均未提供实现仓库，Jaehyeok Kim 的个人页面仍标记为 `Code (Coming soon!)`。因此，本目录属于 paper-based reimplementation（基于论文的重新实现），不是作者代码迁移。

若官方代码发布，应保留当前实现，并在采用上游行为前逐项比较：

- state layout；
- retrieval 与 recency；
- write-back；
- gradient path；
- 数据序列化；
- 指标实现。

## 已实现的论文约束

- Memory 由可修订的 compressed thread states 组成；
- 检索得分为 pooled cosine similarity 乘以 `exp(-alpha * recency)`；
- 所有达到 threshold 的 states 一同参与 generation；
- 无 state 达到 threshold 时，仍使用 top-1 fallback 维持前向连续性；
- 得分低于 threshold 时插入新 state；达到 threshold 时仅替换 argmax state；
- 训练 credit 只沿 argmax write-back state 回传一跳；
- 其他 retrieved states 与 off-topic fallback 作为 stop-gradient context；
- Inference 使用 generated response，训练使用 gold response；
- Generator 冻结，只训练 compressor 与 learnable compression tokens。

上述约束对应论文 Eq. 3–8 和 Algorithm 1；未明确的实现选择记录在 `ASSUMPTIONS.md`。
