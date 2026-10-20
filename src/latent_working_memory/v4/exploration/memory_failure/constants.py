"""四条件的实验身份、语义与固定展示顺序。"""

CONDITIONS = ("full-context", "no-read", "fixed-initial", "ttt")

LABELS = {
    "full-context": "完整上下文",
    "no-read": "同裁剪·关闭读取",
    "fixed-initial": "同裁剪·固定 M₀",
    "ttt": "正常 TTT",
}

COLORS = {
    "full-context": "#52764a",
    "no-read": "#777f8b",
    "fixed-initial": "#d08b32",
    "ttt": "#31688e",
}

DESCRIPTIONS = {
    "full-context": "原生完整 KV；关闭新增记忆分支；与公共 full-context 基线相同的 greedy 解码",
    "no-read": "从正文到生成全程关闭 Q/O 读取修正；保留正常局部更新及 KV 裁剪",
    "fixed-initial": "全程固定 checkpoint 学到的初始 slots；按原时序裁剪，首次边界后启用读取",
    "ttt": "checkpoint 原始配置；逐块局部 TTT、KV 裁剪与 Q/O 读取修正",
}
