"""只导入被选择的方法，其依赖由该方法的独立环境提供。"""

from importlib import import_module


METHODS = {
    "full-context": "latent_working_memory.full_context.locomo",
    "icae": "reproductions.icae.evaluation.locomo_adapter",
    "v3.1.1": "latent_working_memory.v3_1_1.evaluation.locomo_adapter",
}


def method_module(name):
    return import_module(METHODS[name])
