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
"""数据集和加载器的基类。"""

import abc

import literate_dataclasses as dataclasses
import torch

import cebra.data.assets as cebra_data_assets
import cebra.distributions
import cebra.io
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex
from cebra.data.datatypes import Offset

__all__ = ["Dataset", "Loader"]


class Dataset(abc.ABC, cebra.io.HasDevice):
    """实现数据集的抽象基类。

    类属性提供有关索引此数据集时数据形状的信息。

    Attributes:
        input_dimension: 此数据集中信号的输入维度。
            应用于此数据集的模型应匹配此维度。
        offset: 偏移量确定通过``__getitem__``和:py:meth:`expand_index`方法获得的数据形状。
    """

    def __init__(self,
                 device="cpu",
                 download=False,
                 data_url=None,
                 data_checksum=None,
                 location=None,
                 file_name=None):

        self.offset: Offset = cebra.data.Offset(0, 1)
        super().__init__(device)

        self.download = download
        self.data_url = data_url
        self.data_checksum = data_checksum
        self.location = location
        self.file_name = file_name

        if self.download:
            if self.data_url is None:
                raise ValueError(
                    "缺少数据URL。请提供下载数据的URL。"
                )

            if self.data_checksum is None:
                raise ValueError(
                    "缺少数据校验和。请提供校验和以验证数据完整性。"
                )

            cebra_data_assets.download_file_with_progress_bar(
                url=self.data_url,
                expected_checksum=self.data_checksum,
                location=self.location,
                file_name=self.file_name)

    @property
    @abc.abstractmethod
    def input_dimension(self) -> int:
        raise NotImplementedError

    @property
    def continuous_index(self) -> torch.Tensor:
        """连续索引（如果可用）。

        连续索引与相似性度量一起用于绘制正样本和/或负样本。

        Returns:
            形状为``(N,d)``的张量，表示数据集中所有``N``个样本的索引。
        """
        return None

    @property
    def discrete_index(self) -> torch.Tensor:
        """离散索引（如果可用）。

        离散索引可用于使嵌入对变量不变，或限制正样本共享相同的索引变量。
        要实现更复杂的索引操作（例如建模索引之间的相似性），
        最好将离散索引转换为连续索引。

        Returns:
            形状为``(N,)``的张量，表示数据集中所有``N``个样本的索引。
        """
        return None

    def expand_index(self, index: torch.Tensor) -> torch.Tensor:
        """扩展索引以包含偏移量。

        Args:
            index: 一个类型为long的一维张量，包含要从数据集中选择的索引。

        Returns:
            一个形状为``(len(index), len(self.offset))``的扩展索引，其中
            元素将是``expanded_index[i,j] = index[i] + j - self.offset.left``，
            对于所有``j``在``range(0, len(self.offset))``中。

        注意:
            需要设置:py:attr:`offset`。
        """

        # TODO(stes) 通过预分配这些张量/使用非阻塞复制操作可能有提高速度的空间。
        offset = torch.arange(-self.offset.left,
                              self.offset.right,
                              device=index.device)

        index = torch.clamp(index, self.offset.left,
                            len(self) - self.offset.right)

        return index[:, None] + offset[None, :]

    def expand_index_in_trial(self, index, trial_ids, trial_borders):
        """当神经/行为在离散试验中时，例如）猴子伸手数据集
        切片应该在试验内定义。
        trial_ids的大小为self.index的长度，表示索引所属的试验id。
        trial_borders的大小为self.idnex的长度，表示每个试验的边界。

        待办:
            - 重写
        """

        # TODO(stes) 通过预分配这些张量/使用非阻塞复制操作可能有提高速度的空间。
        offset = torch.arange(-self.offset.left,
                              self.offset.right,
                              device=index.device)
        index = torch.tensor(
            [
                torch.clamp(
                    i,
                    trial_borders[trial_ids[i]] + self.offset.left,
                    trial_borders[trial_ids[i] + 1] - self.offset.right,
                ) for i in index
            ],
            device=self.device,
        )
        return index[:, None] + offset[None, :]

    @abc.abstractmethod
    def __getitem__(self, index: torch.Tensor) -> torch.Tensor:
        """返回给定时间索引处的样本。

        Args:
            index: 类型为:py:attr:`torch.long`的索引张量。

        Returns:
            来自数据集的样本，匹配形状
            ``(len(index), self.input_dimension, len(self.offset))``
        """

        raise NotImplementedError

    @abc.abstractmethod
    def load_batch(self, index: BatchIndex) -> Batch:
        """返回指定索引位置的数据。

        TODO: 调整签名以支持Batches和List[Batch]
        """
        raise NotImplementedError()

    def configure_for(self, model: "cebra.models.Model"):
        """Configure the dataset offset for the provided model.

        Call this function before indexing the dataset. This sets the
        :py:attr:`offset` attribute of the dataset.

        Args:
            model: The model to configure the dataset for.
        """
        self.offset = model.get_offset()


@dataclasses.dataclass
class Loader(abc.ABC, cebra.io.HasDevice):
    """Base dataloader class.

    Args:
        See dataclass fields.

    Yields:
        Batches of the specified size from the given dataset object.

    Note:
        The ``__iter__`` method is non-deterministic, unless explicit seeding is implemented
        in derived classes. It is recommended to avoid global seeding in numpy
        and torch, and instead locally instantiate a ``Generator`` object for
        drawing samples.
    """

    dataset: Dataset = dataclasses.field(
        default=None,
        doc="""A dataset instance specifying a ``__getitem__`` function.""",
    )

    time_offset: int = dataclasses.field(default=10)

    num_steps: int = dataclasses.field(
        default=None,
        doc=
        """The total number of batches when iterating over the dataloader.""",
    )

    batch_size: int = dataclasses.field(default=None,
                                        doc="""The total batch size.""")

    def __post_init__(self):
        if self.num_steps is None or self.num_steps <= 0:
            raise ValueError(
                f"num_steps cannot be less than or equal to zero or None. Got {self.num_steps}"
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(
                f"Batch size has to be None, or a non-negative value. Got {self.batch_size}."
            )

    def __len__(self):
        """The number of batches returned when calling as an iterator."""
        return self.num_steps

    def __iter__(self) -> Batch:
        for _ in range(len(self)):
            index = self.get_indices(num_samples=self.batch_size)
            yield self.dataset.load_batch(index)

    @abc.abstractmethod
    def get_indices(self, num_samples: int):
        """Sample and return the specified number of indices.

        The elements of the returned `BatchIndex` will be used to index the
        `dataset` of this data loader.

        Args:
            num_samples: The size of each of the reference, positive and
                negative samples.

        Returns:
            batch indices for the reference, positive and negative sample.
        """
        raise NotImplementedError()
