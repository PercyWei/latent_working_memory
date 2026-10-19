# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RULER 官方子串评分与输出清理。

来源：NVIDIA/RULER，commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a：
scripts/eval/synthetic/constants.py 与 scripts/eval/evaluate.py。
修改：去除未使用的 task_config；逐题评分不 round，汇总阶段保持官方两位百分比精度。
"""

import re


def postprocess_pred(predict_str):
    predict_str = predict_str.strip()
    return re.sub(r"[\x00-\x1f]", "\n", predict_str).strip()


def string_match_part(preds, refs):
    return (
        sum(max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) for pred, ref in zip(preds, refs))
        / len(preds)
        * 100
    )


def string_match_all(preds, refs):
    return (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
            for pred, ref in zip(preds, refs)
        )
        / len(preds)
        * 100
    )
