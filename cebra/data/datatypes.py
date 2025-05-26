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
import collections

__all__ = ["Batch", "BatchIndex", "Offset"]

# Batch = collections.namedtuple(
#    'batch', ['reference', 'positive', 'negative', 'index', 'index_reversed'],
#    defaults=(None, None))


class Batch:
    """A batch of reference, positive, negative samples and an optional index.

    Attributes:
        reference: The reference samples, typically sampled from the prior
            distribution
        positive: The positive samples, typically sampled from the positive
            conditional distribution depending on the reference samples
        negative: The negative samples, typically sampled from the negative
            conditional distribution depending (but often independent) from
            the reference samples
        index: TODO(stes), see docs for multisession training distributions
        index_reversed: TODO(stes), see docs for multisession training distributions
    """

    __slots__ = ["reference", "positive", "negative", "index", "index_reversed"]

    def __init__(self,
                 reference,
                 positive,
                 negative,
                 index=None,
                 index_reversed=None):
        self.reference = reference
        self.positive = positive
        self.negative = negative
        self.index = index
        self.index_reversed = index_reversed

    def to(self, device):
        """Move all batch elements to the GPU."""
        self.reference = self.reference.to(device)
        self.positive = self.positive.to(device)
        self.negative = self.negative.to(device)
        # TODO(stes): Unclear if the indices should also be best represented by
        # torch.Tensors vs. np.ndarrays---this should probably be updated once
        # the GPU implementation of the multi-session sampler is fully ready.
        # if self.index is not None:
        #    self.index = self.index.to(device)
        # if self.index_reversed is not None:
        #    self.index_reversed = self.index_reversed.to(device)


BatchIndex = collections.namedtuple(
    "BatchIndex",
    ["reference", "positive", "negative", "index", "index_reversed"],
    defaults=(None, None),
)


class Offset:
    """索引左侧和右侧的样本数量。

    在索引数据集时，某些操作需要跨时间维度的多个相邻样本的输入。``Offset``表示相对于索引的
    简单的左右偏移对。它提供了在时间维度上采样时要考虑的当前索引周围的样本范围。

    提供的偏移量是正的:py:class:`int`，因此``left``偏移量对应于要考虑的索引之前的样本数量，
    而``right``偏移量是严格正数，对应于索引本身以及要考虑的索引之后的样本数量。

    Note:
        按照惯例，右边界应该始终是**严格正数**，因为它包括当前索引本身。
        因此，例如，要仅考虑当前元素，您必须在:py:class:`Offset`初始化时提供(0,1)。

    """

    __slots__ = ["left", "right"]

    def __init__(self, *offset):
        if len(offset) == 1:
            (offset,) = offset
            self.left = offset
            self.right = offset
        elif len(offset) == 2:
            self.left, self.right = offset
        else:
            raise ValueError(
                f"Invalid number of elements to bound the Offset, expect 1 or 2 elements, got {len(offset)}."
            )
        self._check_offset_positive()

    def _check_offset_positive(self):
        for offset in [self.right, self.left]:
            if offset < 0:
                raise ValueError(
                    f"Invalid Offset bounds, expect value superior or equal to 0, got {offset}."
                )

        if self.right == 0:
            raise ValueError(
                f"Invalid right bound. By convention, the right bound includes the current index. It should be at least set to 1, "
                f"got {self.right}")

    @property
    def _right(self):
        return None if self.right == 0 else -self.right

    @property
    def left_slice(self):
        """Slice from array start to left border."""
        return slice(0, self.left)

    @property
    def right_slice(self):
        """Slice from right border to array end."""
        return slice(self._right, None)

    @property
    def valid_slice(self):
        """Slice between the two borders."""
        return slice(self.left, self._right)

    def __len__(self):
        return self.left + self.right

    def mask_array(self, array, value):
        array[self.left_slice] = value
        array[self.right_slice] = value
        return array

    def __repr__(self):
        return f"Offset(left = {self.left}, right = {self.right}, length = {len(self)})"
