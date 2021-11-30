# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
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


import tensorflow as tf

def softmax_cross_entropy(labels, depth, logits):
    # 移除维度为 1 的轴
    labels = tf.squeeze(labels)
    # 独热编码
    one_hot_labels = tf.one_hot(labels, depth=depth, dtype=tf.float32)
    # 交叉熵
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                           logits=logits)
    return loss

def mean_square_error(labels, logits):
    # 均方损失
    return tf.losses.mean_squared_error(labels, logits)

def multi_label_sigmoid_cross_entropy(labels, depth, logits):
    # 多标签的数据格式与单标签类似，只是labelName一列改成用英文逗号分隔的标签
    one_hots = tf.one_hot(labels, depth)
    # 计算在轴 1 上的最大值
    multi_hots = tf.reduce_max(one_hots, axis=1)
    multi_hots = tf.cast(multi_hots, logits.dtype)
    # 多标签
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=multi_hots, logits=logits)