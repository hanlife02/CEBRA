#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""CEBRA是一个用于估计高维记录一致性嵌入的库，使用辅助变量。
它包含在PyTorch中实现的自监督学习算法，并支持生物学和神经科学中常见的各种数据集。
"""

is_sklearn_available = False
try:
    # TODO(stes): 可以在这里添加更多人们关心的常见集成（例如PyTorch lightning）
    from cebra.integrations.sklearn.cebra import CEBRA
    from cebra.integrations.sklearn.decoder import KNNDecoder
    from cebra.integrations.sklearn.decoder import L1LinearRegressor

    is_sklearn_available = True
except ImportError:
    # 暂时静默失败
    pass

is_matplotlib_available = False
try:
    from cebra.integrations.matplotlib import *

    is_matplotlib_available = True
except ImportError:
    # 暂时静默失败
    pass

is_plotly_available = False
try:
    from cebra.integrations.plotly import *

    is_plotly_available = True
except ImportError:
    # 暂时静默失败
    pass

from cebra.data.load import load as load_data

is_load_deeplabcut_available = False
try:
    from cebra.integrations.deeplabcut import load_deeplabcut
    is_load_deeplabcut_available = True
except (ImportError, NameError):
    pass

import cebra.integrations.sklearn as sklearn

__version__ = "0.6.0a1"
__all__ = ["CEBRA"]
__allow_lazy_imports = False
__lazy_imports = {}


def allow_lazy_imports():
    """启用cebra所有子模块和包的延迟导入。

    如果调用，对``cebra.<module_name>``的引用将在代码中首次调用时自动延迟导入，
    并且不会引发警告。
    """
    __allow_lazy_imports = True


def __getattr__(key):
    """cebra子模块和包的延迟导入。

    一旦导入:py:mod:`cebra`，就可以进行延迟导入

    """
    if key == "CEBRA":
        from cebra.integrations.sklearn.cebra import CEBRA

        return CEBRA
    elif key == "KNNDecoder":
        from cebra.integrations.sklearn.decoder import KNNDecoder  # noqa: F811

        return KNNDecoder
    elif key == "L1LinearRegressor":
        from cebra.integrations.sklearn.decoder import L1LinearRegressor  # noqa: F811

        return L1LinearRegressor
    elif not key.startswith("_"):
        import importlib
        import warnings

        if key not in __lazy_imports:
            # NOTE(celia): 测试字符串示例时需要此条件
            # 以便函数不会尝试将测试包
            # （pytest插件、SetUpModule和TearDownModule）导入为cebra.{key}。
            # 我们只需确保安装了pytest。
            if any(name in key.lower()
                   for name in ["pytest", "setup", "module"]):
                import pytest

                return importlib.import_module(pytest)
            __lazy_imports[key] = importlib.import_module(f"{__name__}.{key}")
            if not __allow_lazy_imports:
                warnings.warn(
                    f"Your code triggered a lazy import of {__name__}.{key}. "
                    f"While this will (likely) work, it is recommended to "
                    f"add an explicit import statement to you code instead. "
                    f"To disable this warning, you can run "
                    f"``cebra.allow_lazy_imports()``.")
        return __lazy_imports[key]
    raise AttributeError(f"module 'cebra' has no attribute '{key}'. "
                         f"Did you import cebra.{key}?")
