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
import numpy as np
import pytest
import scipy.stats
import sklearn.model_selection

import cebra.data.helper as cebra_data_helper
import cebra.datasets


def _initialize_orthogonal_alignment_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_150_4d = np.random.uniform(0, 1, (150, 4))
    embedding_1_4d = np.random.uniform(0, 1, (1, 4))
    embedding_100_1d = np.random.uniform(0, 1, (100, 1))
    embedding_100_none = np.random.uniform(0, 1, (100,))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_150_1d = np.random.uniform(0, 1, (150, 1))
    labels_100_3d = np.random.uniform(0, 1, (100, 3))
    labels_150_3d = np.random.uniform(0, 1, (150, 3))
    labels_100_none = np.random.uniform(0, 1, (100,))
    labels_1_1d = np.random.uniform(0, 1, (1, 1))

    orthogonal_alignment_data = []

    # 不同的样本大小
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_100_1d, labels_150_1d))
    # 不同的样本大小且标签具有多个特征
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_100_3d, labels_150_3d))
    # 要对齐的嵌入只有单个样本
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_1_4d, labels_100_1d, labels_1_1d))
    # 只有一个特征维度的嵌入
    orthogonal_alignment_data.append(
        (embedding_100_1d, embedding_100_1d, labels_100_1d, labels_100_1d))
    # 只有一个特征维度的嵌入，形状为None
    orthogonal_alignment_data.append(
        (embedding_100_none, embedding_100_none, labels_100_1d, labels_100_1d))
    # 只有一个特征维度的标签，形状为None
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_100_4d, labels_100_none, labels_100_none))
    # 相同样本大小，无标签
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_100_4d, None, None))

    return orthogonal_alignment_data


def _initialize_invalid_orthogonal_alignment_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_150_4d = np.random.uniform(0, 1, (150, 4))
    embedding_110_6d = np.random.uniform(0, 1, (110, 6))
    embedding_1_4d = np.random.uniform(0, 1, (1, 4))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_150_1d = np.random.uniform(0, 1, (150, 1))
    labels_110_1d = np.random.uniform(0, 1, (110, 1))
    labels_1_1d = np.random.uniform(0, 1, (1, 1))

    orthogonal_alignment_data = []

    # 具有不同特征数量的嵌入
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_110_6d, labels_100_1d, labels_110_1d,
         "Invalid.*data"))
    # 参考嵌入和标签的样本大小不同
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_150_1d, labels_150_1d,
         "Mismatched.*data.*labels"))
    # 要对齐的嵌入和标签的样本大小不同
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_100_1d, labels_100_1d,
         "Mismatched.*data.*labels"))
    # 参考嵌入只包含一个样本
    orthogonal_alignment_data.append(
        (embedding_1_4d, embedding_100_4d, labels_1_1d, labels_100_1d,
         "Invalid.*reference.*data"))
    # 缺少参考标签
    orthogonal_alignment_data.append((embedding_100_4d, embedding_150_4d, None,
                                      labels_100_1d, "Missing.*labels"))
    # 缺少对齐标签
    orthogonal_alignment_data.append((embedding_100_4d, embedding_150_4d,
                                      labels_100_1d, None, "Missing.*labels"))
    # 对于在时间上未对齐的嵌入（样本数量不同）缺少标签
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, None, None, "Missing.*labels"))

    return orthogonal_alignment_data


def _does_shape_match(data, aligned_data):
    if len(data.shape) == 2:
        is_matching = aligned_data.shape == data.shape
    else:
        is_matching = aligned_data.shape == (data.shape[0], 1)
    return is_matching


@pytest.mark.parametrize("ref_data,data,ref_labels,labels",
                         _initialize_orthogonal_alignment_data())
def test_orthogonal_alignment_shapes(ref_data, data, ref_labels, labels):
    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()

    alignment_model.fit(ref_data, data, ref_labels, labels)
    aligned_embedding = alignment_model.transform(data)
    assert _does_shape_match(data, aligned_embedding)

    aligned_embedding = alignment_model.fit_transform(ref_data, data,
                                                      ref_labels, labels)
    assert _does_shape_match(data, aligned_embedding)

    # Test with non-default parameters
    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=10)

    aligned_embedding = alignment_model.fit_transform(ref_data, data,
                                                      ref_labels, labels)
    assert _does_shape_match(data, aligned_embedding), (data.shape,
                                                        aligned_embedding.shape)


@pytest.mark.parametrize("ref_data,data,ref_labels,labels,match",
                         _initialize_invalid_orthogonal_alignment_data())
def test_invalid_orthogonal_alignment(ref_data, data, ref_labels, labels,
                                      match):
    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()

    with pytest.raises(ValueError, match=match):
        alignment_model.fit(ref_data, data, ref_labels, labels)


def test_orthogonal_alignment_without_labels():
    random_seed = 2160
    np.random.seed(random_seed)
    embedding_100_4d = np.random.uniform(0, 1, (1000, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (1000, 4))

    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()

    alignment_model.fit(embedding_100_4d, embedding_100_4d_2, None, None)
    aligned_embedding = alignment_model.transform(embedding_100_4d_2)

    alignment_model.fit(embedding_100_4d, embedding_100_4d_2)
    aligned_embedding_without_labels = alignment_model.transform(
        embedding_100_4d_2)

    assert np.allclose(aligned_embedding, aligned_embedding_without_labels)


@pytest.mark.parametrize("seed", [483, 425, 166, 672, 123])
def test_orthogonal_alignment(seed):
    np.random.seed(seed)
    embedding_100_4d = np.random.uniform(0, 1, (1000, 4))
    orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=4, random_state=seed)
    labels_100_1d = np.random.uniform(0, 1, (1000, 1))

    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()
    aligned_embedding = alignment_model.fit_transform(ref_data=embedding_100_4d,
                                                      data=np.dot(
                                                          embedding_100_4d,
                                                          orthogonal_matrix),
                                                      ref_label=labels_100_1d,
                                                      label=labels_100_1d)
    assert np.allclose(aligned_embedding, embedding_100_4d, atol=0.03)

    # 不使用标签的情况
    aligned_embedding = alignment_model.fit_transform(ref_data=embedding_100_4d,
                                                      data=np.dot(
                                                          embedding_100_4d,
                                                          orthogonal_matrix))
    assert np.allclose(aligned_embedding, embedding_100_4d, atol=0.03)


def _initialize_embedding_ensembling_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_3 = np.random.uniform(0, 1, (100, 4))
    embedding_110_none = np.random.uniform(0, 1, (110,))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_100_1d_2 = np.random.uniform(0, 1, (100, 1))
    labels_100_1d_3 = np.random.uniform(0, 1, (100, 1))

    embedding_ensembling_data = []

    # 相同形状的嵌入，无标签
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3], None, 0))
    # 具有单个特征维度的嵌入（形状为None）
    embedding_ensembling_data.append(([embedding_110_none,
                                       embedding_110_none], None, 0))
    # 嵌入和用于对齐的相应标签（可选）
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3],
         [labels_100_1d, labels_100_1d_2, labels_100_1d_3], 0))
    # 并行处理，使用2个作业
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3], None, 2))
    # 并行处理，使用所有可用作业
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3], None, -1))

    return embedding_ensembling_data


def _initialize_invalid_embedding_ensembling_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_3 = np.random.uniform(0, 1, (100, 4))
    embedding_150_4d = np.random.uniform(0, 1, (150, 4))
    embedding_100_6d = np.random.uniform(0, 1, (100, 6))
    embedding_1_4d = np.random.uniform(0, 1, (1, 4))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_100_1d_2 = np.random.uniform(0, 1, (100, 1))
    labels_150_1d = np.random.uniform(0, 1, (150, 1))

    embedding_ensembling_data = []

    # 作业数设置为None
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_100_4d_3], None, None, "Invalid.*n_jobs"))
    # 具有不同样本大小的嵌入
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_150_4d], None, 0, "Inconsistent.*embeddings"))
    # 具有不同特征数量的嵌入
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_100_6d], None, 0, "Inconsistent.*embeddings"))
    # 嵌入和标签，其中标签与其对应嵌入的样本大小不同
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_100_4d_3], [labels_100_1d, labels_100_1d_2,
                                labels_150_1d], 0, "Mismatched.*data.*labels"))
    # 只有单个样本的嵌入（至少需要{k_top}个）
    embedding_ensembling_data.append(
        ([embedding_1_4d, embedding_1_4d], None, 0, "Invalid.*data"))

    return embedding_ensembling_data


@pytest.mark.parametrize("embeddings,labels,n_jobs",
                         _initialize_embedding_ensembling_data())
def test_embedding_ensembling_shapes(embeddings, labels, n_jobs):

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=embeddings, labels=labels, n_jobs=n_jobs)
    assert _does_shape_match(embeddings[0], joint_embedding)


def test_embeddings_ensembling_without_labels():
    random_seed = 2160
    np.random.seed(random_seed)

    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (100, 4))

    # 相同样本大小，无标签
    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2], labels=[None, None])
    joint_embedding_without_labels = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2])
    assert np.allclose(joint_embedding, joint_embedding_without_labels)


@pytest.mark.parametrize("embeddings,labels,n_jobs,match",
                         _initialize_invalid_embedding_ensembling_data())
def test_invalid_embedding_ensembling(embeddings, labels, n_jobs, match):
    with pytest.raises(ValueError, match=match):
        _ = cebra_data_helper.ensemble_embeddings(
            embeddings=embeddings,
            labels=labels,
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize("seed", [483, 426, 166, 674, 123])
def test_embedding_ensembling(seed):
    np.random.seed(seed)
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=4, random_state=seed)
    orthogonal_matrix_2 = scipy.stats.ortho_group.rvs(dim=4,
                                                      random_state=seed + 1)

    embedding_100_4d_2 = np.dot(embedding_100_4d, orthogonal_matrix)
    embedding_100_4d_3 = np.dot(embedding_100_4d, orthogonal_matrix_2)

    labels = [labels_100_1d, labels_100_1d, labels_100_1d]

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3],
        labels=labels)
    assert np.allclose(joint_embedding, embedding_100_4d, atol=0.05)

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3])
    assert np.allclose(joint_embedding, embedding_100_4d, atol=0.05)


@pytest.mark.benchmark
@pytest.mark.requires_dataset
def test_ensembling_performances(n_models=3):
    dataset = cebra.datasets.init("rat-hippocampus-single-achilles")

    scores, embeddings = [], []
    for i in range(n_models):
        cebra_model = cebra.CEBRA(model_architecture="offset10-model",
                                  max_iterations=100,
                                  batch_size=512,
                                  verbose=False)
        cebra_model.fit(dataset.neural)
        embedding = cebra_model.transform(dataset.neural)

        (train_embedding, valid_embedding, train_label,
         valid_label) = sklearn.model_selection.train_test_split(
             embedding, dataset.continuous_index[:, 0], test_size=0.3)

        decoder = cebra.KNNDecoder()
        decoder.fit(train_embedding, train_label)
        score = decoder.score(valid_embedding, valid_label)

        scores.append(score)
        embeddings.append(embedding)

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=embeddings)

    (train_embedding, valid_embedding, train_label,
     valid_label) = sklearn.model_selection.train_test_split(
         joint_embedding, dataset.continuous_index[:, 0], test_size=0.3)

    decoder = cebra.KNNDecoder()
    decoder.fit(train_embedding, train_label)
    score = decoder.score(valid_embedding, valid_label)

    assert all(score_i < score for score_i in scores)
