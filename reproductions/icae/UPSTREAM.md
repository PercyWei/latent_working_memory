# ICAE v1 上游来源与迁移记录（20260904 10:13:57 CST）

创建时间：20260904 10:13:57 CST（UTC+08:00）

最后修订时间：20260904 19:55:44 CST（UTC+08:00）

迁移日期：20260903

## 上游信息

- 仓库：https://github.com/getao/icae
- Commit：`469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd`
- Commit 时间：`2024-05-11T16:53:58-07:00`
- 上游许可证：CC0-1.0，副本保存在 `LICENSE`；
- 选用实现：`code/icae_v1`。C-DIC 使用公开的 Llama-2-7B-Chat ICAE checkpoint，其压缩 token 数量为 128。

`code/icae_v1/peft` 中的定制 PEFT 快照将自身版本标记为 `0.4.0.dev0`，其 Apache-2.0 许可证保存在 `vendor/peft/LICENSE`。

## 迁移路径

| 上游路径 | 本地路径 |
|---|---|
| `code/icae_v1/base` | `src/icae/base` |
| `code/icae_v1/utils` | `src/icae/utils` |
| `code/icae_v1/llama_icae_modeling.py` | `src/icae/llama_icae_modeling.py` |
| `code/icae_v1/llama_icae_learning.py` | `src/icae/llama_icae_learning.py` |
| `code/icae_v1/ft_inference.py` | `examples/ft_inference_upstream.py` |
| `code/icae_v1/peft/src/peft` | `vendor/peft/src/peft` |

上游目录 `icae_v1` 在本地安装为 `icae` package，使 `from icae.utils import stable_trainer` 等绝对导入能够保持原有含义，无需修改。

## 本地兼容性改动

1. 仅在启用可选的 `better_transformer` 参数时导入 `optimum.bettertransformer.BetterTransformer`。上游代码无条件导入该模块，使默认路径依赖一个未固定版本的可选 package；本地修改不改变默认行为。
2. 在迁移的 ICAE 实现之外增加了 package 配置、环境检查和测试。
3. 原始推理示例及其中的占位路径保持不变，且未注册为 CLI。
4. 服务器锁文件使用 PyTorch 2.0.1+cu118。该版本比当前 cu130 build 更接近定制 PEFT 所对应的 PyTorch 版本；支持 CUDA 13.0 的 NVIDIA 驱动可以运行 wheel 内包含的旧版 CUDA runtime。
5. 清理了复制文本和 Python 文件中的行尾空格，不影响运行行为。
6. 公开的 ICAE v1 checkpoint 使用标量 `0.0` 作为冻结 Llama 参数的占位符。本地 loader 会先从初始化后的基础模型恢复这些条目，再严格加载完整 state dict。该流程由上游作者在 [issue #1](https://github.com/getao/icae/issues/1) 中确认。
7. 本地推理入口使用 ICAE 的 `model.eos_id`（`1`）作为停止 token。它不同于 Llama tokenizer 的 EOS ID（`2`）；服务器 smoke test 表明，使用 tokenizer EOS 会使模型在生成正确答案后继续生成。

## 已知的上游 v1 缺口

- `ft_inference.py` 中存在拼写错误 `memopry_mask`；
- 该文件导入了 `instruct_ft_tokenize_function`，但公开的 `llama_icae_learning.py` 快照中没有此函数；
- 文件路径与输出位置使用占位值，而不是 CLI 参数。

保留的上游示例只记录这些问题，不作静默修复。本项目单独实现所需推理行为，并已使用公开 checkpoint 完成验证。

除此之外，迁移过程未修改模型架构、压缩 token 逻辑、loss、trainer 或定制 PEFT 实现。
