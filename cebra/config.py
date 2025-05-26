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
import argparse
import json
from typing import Literal, Optional

import literate_dataclasses as dataclasses

import cebra.data
import cebra.datasets


@dataclasses.dataclass
class Config:
    data: str = dataclasses.field(
        init=False,
        doc="""运行CEBRA的数据集。
    标准数据集在cebra.datasets中可用。
    您可以通过继承cebra.data.Dataset并使用
    ``@cebra.datasets.register``装饰器注册数据集来创建自己的数据集。
    """,
    )

    variant: str = dataclasses.field(
        default="single-session",
        doc="""要运行的CEBRA变体。
    """,
    )

    logdir: str = dataclasses.field(
        default="/logs/single-rat-hippocampus-behavior/",
        doc="""模型日志目录。
    这应该是一个新的空目录，或者是包含已训练CEBRA模型的
    现有目录。
    """,
    )
    distance: str = dataclasses.field(
        default="cosine", doc="""计算损失时使用的距离类型""")

    loss_distance: str = dataclasses.field(
        default="cosine",
        doc=
        """'distance'参数的旧版本。计算损失时使用的距离类型""",
    )

    temperature_mode: Literal["auto", "constant"] = dataclasses.field(
        default="constant",
        doc=
        """InfoNCE损失的温度。如果为'auto'，温度是可学习的。如果设置为'constant'，则固定为给定值""",
    )

    temperature: float = dataclasses.field(
        default=1.0, doc="""InfoNCE损失的温度。""")

    min_temperature: Optional[float] = dataclasses.field(
        default=None, doc="""可学习温度的最小值""")

    time_offset: int = dataclasses.field(
        default=10,
        doc=
        """ 正样本对之间的距离（时间上）。此参数的解释取决于所选的条件分布，
        但通常较高的时间偏移会增加学习任务的难度，并且（在一定范围内）提高表示的质量。
        时间偏移通常应大于模型指定的感受野。""",
    )

    delta: float = dataclasses.field(
        default=0.1,
        doc=
        """ Standard deviation of gaussian distribution if it is chossed to use 'delta' distribution.
        The positive sample will be chosen by closest sample to a reference which is sampled from the defined gaussian
        distribution.""",
    )

    conditional: str = dataclasses.field(
        default="time_delta",
        doc=
        """条件分布的类型。有效的标准方法有"time_delta"、"time"、"delta"，
        更多方法可以添加到``cebra.data``注册表中。""",
    )

    num_steps: int = dataclasses.field(
        default=1000,
        doc="""总训练步骤数。
    CEBRA的训练持续时间与数据集大小无关。看到的总训练
    示例将达到``num-steps x batch-size``，
    与数据集大小无关。
    """,
    )

    learning_rate: float = dataclasses.field(
        default=3e-4, doc="""Adam优化器的学习率。""")
    model: str = dataclasses.field(
        default="offset10-model",
        doc=
        """模型架构。可用选项有'offset10-model'、'offset5-model'和'offset1-model'。""",
    )

    models: list = dataclasses.field(
        default_factory=lambda: [],
        doc=
        """多会话训练的模型架构。如果未设置，将对所有会话使用model参数""",
    )

    batch_size: int = dataclasses.field(
        default=512, doc="""每个训练步骤的总批次大小。""")

    num_hidden_units: int = dataclasses.field(default=32,
                                              doc="""隐藏单元数量。""")

    num_output: int = dataclasses.field(default=8,
                                        doc="""输出嵌入的维度""")

    device: str = dataclasses.field(
        default="cpu", doc="""训练设备。选项：cpu/cuda""")

    tqdm: bool = dataclasses.field(
        default=False, doc="""Activate tqdm for logging during the training""")

    save_frequency: int = dataclasses.field(
        default=None, doc="""Interval of saving intermediate model""")
    valid_frequency: int = dataclasses.field(
        default=100, doc="""Interval of validation in training""")

    train_ratio: float = dataclasses.field(
        default=0.8,
        doc="""Ratio of train dataset. The remaining will be used for
        valid and test split.""",
    )
    valid_ratio: float = dataclasses.field(
        default=0.1,
        doc="""Ratio of validation set after the train data split.
        The remaining will be test split""",
    )

    @classmethod
    def _add_arguments(cls, parser, **override_kwargs):
        _metavars = {int: "N", float: "val"}

        def _json(self):
            return json.dumps(self.__dict__)

        for field in dataclasses.fields(cls):
            if field.type == list:
                kwargs = dict(
                    metavar=field.default_factory(),
                    default=field.default_factory(),
                    help=f"{str(field.metadata['doc'])}",
                )
                kwargs.update(override_kwargs.get(field.name, {}))
                parser.add_argument("--" + field.name.replace("_", "-"),
                                    **kwargs,
                                    nargs="+")
            else:
                kwargs = dict(
                    type=field.type,
                    metavar=field.default,
                    default=field.default,
                    help=f"{str(field.metadata['doc'])}",
                )
                kwargs.update(override_kwargs.get(field.name, {}))
                parser.add_argument("--" + field.name.replace("_", "-"),
                                    **kwargs)

        return parser

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """向参数解析器添加参数。"""
        cls._add_arguments(
            parser,
            data={"choices": cebra.datasets.get_options()},
            device={"choices": ["cpu", "cuda"]},
        )
        return parser

    def asdict(self):
        return self.__dict__

    def as_namespace(self):
        return argparse.Namespace(**self.asdict())


def add_arguments(parser):
    """向argparser添加CEBRA命令行参数。"""
    return Config.add_arguments(parser)
