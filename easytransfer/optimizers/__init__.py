# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from tensorflow.python.framework import ops
from .adam_weight_decay_optimizer import AdamWeightDecayOptimizer
from .lamb_weight_decay_optimizer import LambWeightDecayOptimizer
import numpy as np


def get_train_op(
    learning_rate: float,
    weight_decay_ratio: float,
    loss: float,
    num_towers: int = 1,
    warmup_ratio: float = 0.1,
    lr_decay: str = "polynomial",
    optimizer_name: str = None,
    tvars=None,
    train_steps: int = None,
    clip_norm: bool = True,
    clip_norm_value: float = 1.0,
    num_freezed_layers: int = 0,
):
    """
    获取训练操作
    learning_rate: 学习率
    weight_decay_ratio: 权重衰减系数
    lss: 损失值
    num_towers: 分布式训练的tower数量
    warmup_ratio: 学习率开始warm up的比例
    lr_decay: 学习率衰减策略
    optimizer_name: 优化器名称
    tvars: 需要训练的变量
    train_steps: 训练总步数
    clip_norm: 是否使用clip_norm
    clip_norm_value: clip_norm的值
    num_freezed_layers: 冻结的层数
    """
    # warm up 的步数. 总的训练步长 * 预热百分比
    warmup_steps = int(train_steps * warmup_ratio)
    # 全局步骤, 也就是当前步长
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # 支持这种学习策略
    if lr_decay == "polynomial":
        # 原来都是这个需要的参数
        # global_step, train_steps
        learning_rate = tf.compat.v1.train.polynomial_decay(
            learning_rate, global_step, train_steps, end_learning_rate=0.0, power=1.0, cycle=False
        )
    else:
        learning_rate = learning_rate

    # 不等于 0 时
    if warmup_steps != 0:
        tf.compat.v1.logging.info("*******Warmup {} steps***********".format(warmup_steps))
        # 转换类型
        global_steps_int = tf.cast(global_step, tf.int32)
        # 定义常量
        warmup_steps_int = tf.constant(warmup_steps, dtype=tf.int32)

        # 转换类型
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        # 已经完成的百分比
        warmup_percent_done = global_steps_float / warmup_steps_float
        # 预热阶段的学习率是 学习率 * 预热百分比
        warmup_learning_rate = learning_rate * warmup_percent_done

        # 这也能变成浮点数, 1 或者 0. 是否完全预热 warmup. 当小于预热步长时, 就是在预热阶段
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        # 更新学习率. 如果在预热阶段, 使用 warmup_learning_rate. 否则, 使用 learning_rate
        learning_rate = (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate
    else:
        tf.compat.v1.logging.info("*******Don't warm up, then lr will polynomial_decay only************")

    # 当优化器名字是 adam 时
    if optimizer_name == "adam":
        # 如果有权重衰减系数, 则使用 AdamWeightDecayOptimizer
        if weight_decay_ratio > 0:
            tf.compat.v1.logging.info("*******Using adamW optimizer************")
            optimizer = AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                weight_decay_rate=weight_decay_ratio,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                # 有几个层不使用权重衰减系数
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            )
        else:
            # 没有就使用普通的 AdamOptimizer
            tf.compat.v1.logging.info("*******Using adam optimizer************")
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-6
            )

    # 当优化器名字是 lamb 时, 使用 LambWeightDecayOptimizer
    elif optimizer_name == "lamb":
        tf.compat.v1.logging.info("*******Using lamb optimizer************")
        optimizer = LambWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_ratio,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    # 另外两个优化器, 不需要使用权重衰减系数, 只需要学习率
    elif optimizer_name == "adagrad":
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate)
    else:
        raise ValueError("Set train op optimizer adam or lamb")

    # 如果没有传入 tvars, 就初始化成所有可学习的变量
    if tvars is None:
        tvars = tf.compat.v1.trainable_variables()

    # 计算梯度
    grads = tf.gradients(ys=loss, xs=tvars)

    # np.prod 计算元素的乘积, 然后用 np.sum 求和
    tf.compat.v1.logging.info(
        "*******Num of trainable variables {}************".format(
            np.sum([np.prod(v.get_shape().as_list()) for v in tvars])
        )
    )

    # 如果需要裁剪标准化
    if clip_norm:
        tf.compat.v1.logging.info("*******Clip Gradients************")
        tf.compat.v1.logging.info("*******Clip Norm Value {}*********".format(clip_norm_value))
        # 新的梯度
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm_value)
    else:
        tf.compat.v1.logging.info("*******Don't Clip Gradients************")

    tf.compat.v1.logging.info("*********Num towers is {} *********".format(num_towers))
    # 对每一个梯度
    for i in range(len(grads)):
        # 如果不是 None
        if grads[i] is not None:
            # 如果是 index 索引
            if isinstance(grads[i], ops.IndexedSlices):
                # 就需要将这个位置的值转换成索引
                grads[i] = ops.convert_to_tensor(grads[i])
            # 值变成 塔数分之一
            grads[i] *= 1.0 / num_towers

    # 如果冻结的层数大于 0
    if num_freezed_layers > 0:
        tf.compat.v1.logging.info("*******Num Freezed Layers is {} ************".format(num_freezed_layers))
        for i in range(len(grads)):
            # 对于每一个梯度
            freeze = False
            # 冻结的层数, 从 0 开始到 num_freezed_layers - 1
            for l in range(num_freezed_layers):
                # 入彀 layer_l 在 tvars[i] 的名字中, 就要冻结
                if "layer_{}/".format(l) in tvars[i].name:
                    freeze = True
            if freeze:
                # 如果冻结, 这个梯度就要设置为 0
                grads[i] *= 0
                tf.compat.v1.logging.info("Freezing var name is {}".format(tvars[i].name))

    # 在优化器上使用梯度
    train_op = optimizer.apply_gradients(
        # 并行梯度和变量
        zip(grads, tvars),
        global_step=global_step,
    )
    tf.compat.v1.summary.scalar("learning_rate", learning_rate)

    return train_op
