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
"""数据加载器使用分布和索引来为训练提供样本。

此包包含用于在CEBRA的各种使用模式（例如单会话和多会话数据集）中定义和加载数据集的
所有辅助函数和类。它不特定于特定数据集（有关实际数据集实现，请参见:py:mod:`cebra.datasets`）。
但是，所有数据集的基类都在此处定义，以及与数据集交互的辅助函数。

CEBRA开箱即用地支持不同的数据集类型：

- :py:class:`cebra.data.single_session.SingleSessionDataset` 是单会话数据集的抽象基类。单会话数据集
  在样本（例如神经数据）和所有上下文变量（例如行为、刺激等）中具有相同的特征维度。
- :py:class:`cebra.data.multi_session.MultiSessionDataset` 是多会话数据集的抽象基类。
  多会话数据集包含多个单会话数据集。关键是，辅助变量维度的维数需要在会话之间匹配，
  这允许多个会话的对齐。信号变量的维数可以在会话之间任意变化。

请注意，数据集的实际实现（例如用于基准测试）在:py:mod:`cebra.datasets`包中完成。

"""

# NOTE(stes): 有意的导入顺序以避免循环导入
#             这些导入不会被isort重新排序（参见.isort.cfg）
from cebra.data.base import *
from cebra.data.datatypes import *
from cebra.data.single_session import *
from cebra.data.multi_session import *
from cebra.data.multiobjective import *
from cebra.data.datasets import *
from cebra.data.helper import *
