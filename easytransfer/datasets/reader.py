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

import sys
sys.path.append("../..")
import tensorflow as tf
from easytransfer.engines.distribution import Process
from collections import OrderedDict

class Reader(Process):
    """
    读取器
    """
    def __init__(self,
                 batch_size,
                 is_training,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTReader',
                 **kwargs):

        # 先初始化父类
        Process.__init__(self, job_name,
                         thread_num,
                         input_queue,
                         output_queue,
                         batch_size=batch_size)

        # 训练样本数和测试样本数
        self.num_train_examples = 0
        self.num_eval_examples = 0

        self.is_training = is_training
        self.num_parallel_batches = kwargs.pop("num_parallel_batches", 1)
        self.shuffle_buffer_size = kwargs.pop("shuffle_buffer_size", None)
        self.prefetch_buffer_size = kwargs.pop("prefetch_buffer_size", 1)
        self.input_schema = kwargs.pop("input_schema", None)
        # for all mode, generate tf.Tensor placeholders
        # all mode need a input_schema, column_name:type:length
        # 对于所有的模式, 都会生成一个 tf.Tensor 占位符
        self.input_tensors = OrderedDict()
        self.input_tensor_names = []  # 包含所有的列名
        # self.input_schema 可能是 None, 这会导致出错
        """
        input_schema 的结构是用逗号`,`分隔的, 每个表示一个列, 每个列用冒号`:`分隔
        column_name:type:length
        """
        for schema in self.input_schema.split(","):
            # 第一个是列名
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)
            # 第二个是类型名
            type = schema.split(":")[1]
            # 第三个序列长度
            seq_len = int(schema.split(":")[2])
            # 对类型名直接进行枚举, 转换成对应的 tf.Tensor 类型, 以及指定默认值
            if type == "int":
                tensor_type = tf.int64
                default_value = 0
            elif type == "float":
                tensor_type = tf.float32
                default_value = 0.0
            elif type == "str":
                tensor_type = tf.string
                default_value = ''
            elif type == "base64":
                tensor_type = "base64"
                default_value = "base64"
            else:
                raise ValueError("unsupported feature type")

            # 变成一个固定长度的输入特征
            self.input_tensors[name] = tf.io.FixedLenFeature([seq_len], tensor_type, default_value)
        # 分布式策略
        distribution_strategy = kwargs.pop("distribution_strategy", None)
        # 小批量数
        num_micro_batches = kwargs.pop("num_micro_batches", 1)
        # 一个特定的策略, 会更新 batch_size
        if distribution_strategy == "ExascaleStrategy":
            self.batch_size = batch_size * num_micro_batches
        else:
            self.batch_size = batch_size

        # 日志中记录下配置参数
        tf.logging.info("num_parallel_batches {}".format(self.num_parallel_batches))
        tf.logging.info("shuffle_buffer_size {}".format(self.shuffle_buffer_size))
        tf.logging.info("prefetch_buffer_size {}".format(self.prefetch_buffer_size))
        tf.logging.info("batch_size {}".format(self.batch_size))
        tf.logging.info("distribution_strategy {}".format(distribution_strategy))
        tf.logging.info("num_micro_batches {}".format(num_micro_batches))
        tf.logging.info("input_schema {}".format(self.input_schema))

    def _get_data_pipeline(self, dataset: tf.data.Dataset, _decode_fn):
        """
        获取数据流水线
        """
        # 如果是训练模式
        if self.is_training:
            # 当没有指定时, 直接用整个训练集的数量作为 shuffle_buffer_size
            if self.shuffle_buffer_size is None:
                tf.logging.info("Random shuffle on the whole {} training examples".format(self.num_train_examples))
                self.shuffle_buffer_size = self.num_train_examples
            # 如果没有参数, 就是无限重复数据集
            dataset = dataset.repeat()
            # 打乱数据
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        else:
            dataset = dataset.repeat(1)

        return self._map_batch_prefetch(dataset, _decode_fn)

    def _map_batch_prefetch(self, dataset, decode_fn):
        # 就是在数据流水线中应用了 decode_fn 函数
        dataset = dataset.apply(
            # map_and_batch 已被弃用, 等效于 map 和 batch 的组合
            tf.data.experimental.map_and_batch(
                lambda *record: decode_fn(*record),
                batch_size=self.batch_size,
                num_parallel_batches=self.num_parallel_batches,
                drop_remainder=False))
        # 预取数据, 提高吞吐量. 文档上建议流水线以这个结束
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        return dataset

    def process(self, input_data):
        raise NotImplementedError("must be implemented in descendants")








