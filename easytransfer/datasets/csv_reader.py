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
from .reader import Reader

class CSVReader(Reader):
    """ Read csv format

    Args:

        input_glob : input file fp
        batch_size : input batch size
        is_training : True or False
        thread_num: thread number

    """

    def __init__(self,
                 input_glob,
                 batch_size,
                 is_training,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTCSVReader',
                 **kwargs):

        super(CSVReader, self).__init__(batch_size,
                                        is_training,
                                        thread_num,
                                        input_queue,
                                        output_queue,
                                        job_name,
                                        **kwargs)

        self.input_glob = input_glob
        # 都是读取下最大行数
        if is_training:
            with tf.gfile.Open(input_glob, 'r') as f:
                for record in f:
                    self.num_train_examples += 1
            tf.logging.info("{}, total number of training examples {}".format(input_glob, self.num_train_examples))
        else:
            with tf.gfile.Open(input_glob, 'r') as f:
                for record in f:
                    self.num_eval_examples += 1
            tf.logging.info("{}, total number of eval examples {}".format(input_glob, self.num_eval_examples))

        # 然后再次打开文件
        self.csv_reader = tf.gfile.Open(input_glob)

    def get_input_fn(self):
        """
        获取输入函数, 这里是读取 csv 文件, 然后经过数据流水线处理
        """
        def input_fn():
            dataset = tf.data.TextLineDataset(self.input_glob)
            return self._get_data_pipeline(dataset, self._decode_csv)

        return input_fn

    def _decode_csv(self, record):
        """
        解析 csv 文件
        """
        record_defaults = [] # 默认值
        tensor_names = [] # 列名
        shapes = [] # 形状
        # 遍历每个输入列, 全部提取到上面三个变量中
        for name, feature in self.input_tensors.items():
            default_value = feature.default_value
            shape = feature.shape
            # 根据第一维的长度来判断
            if shape[0] > 1:
                # 只保留 base64, 其他都将默认值设置为空字符串
                if default_value == 'base64':
                    default_value = 'base64'
                else:
                    default_value = ''
            else:
                # 这个 else 就是多余的, 前面已经设置过了
                default_value = feature.default_value
            record_defaults.append([default_value])
            tensor_names.append(name)
            shapes.append(feature.shape)

        # 列数
        num_tensors = len(tensor_names)

        # 解析 csv 成 tensor, 分隔符是 \t
        items = tf.decode_csv(record, field_delim='\t', record_defaults=record_defaults, use_quote_delim=False)
        outputs = dict()
        total_shape = 0
        for shape in shapes:
            # 形状加起来有什么意义吗?
            total_shape += sum(shape)

        for idx, (name, feature) in enumerate(self.input_tensors.items()):
            # finetune feature_text
            # 如果总的形状数不等于列数
            if total_shape != num_tensors:
                # 取出一个 tensor
                input_tensor = items[idx]
                # 如果 shape 和大于 1
                if sum(feature.shape) > 1:
                    # 取出相应的默认值
                    default_value = record_defaults[idx]
                    # 默认值是个数组, 且长度为 1, 所以取出第一个元素
                    if default_value[0] == '':
                        # 将字符串类型的数字, 转换成对应的数字类型
                        # 但是这个 feature.dtype 可能是 tf.string, 会导致报错的. TODO: 这是怎么在哪里避免的?
                        # 难不成是因为 shape 大于 1 的这种情况下, 不允许 str 类型吗?
                        output = tf.string_to_number(
                            # string_split 将字符串用逗号分隔
                            # expand_dims 添加一个维度, axis 指定位置, 在最前面
                            tf.string_split(tf.expand_dims(input_tensor, axis=0), delimiter=",").values,
                            feature.dtype)
                        # 重新塑造形状, 就是保留第一个维度, 后面那个维度自行推导出来
                        output = tf.reshape(output, [feature.shape[0], ])
                    elif default_value[0] == 'base64':
                        # 如果是 base64, 先解析下
                        decode_b64_data = tf.io.decode_base64(tf.expand_dims(input_tensor, axis=0))
                        # 用 decode_raw 重新解读为 tf.float32 类型
                        output = tf.reshape(tf.io.decode_raw(decode_b64_data, out_type=tf.float32),
                                            [feature.shape[0], ])
                else:
                    # 就是在最前面直接加上一个维度
                    output = tf.reshape(input_tensor, [1, ])
            elif total_shape == num_tensors:
                # preprocess raw_text
                output = items[idx]
            # 一个 output 就是一列的数据
            outputs[name] = output
        return outputs

    def process(self, input_data):
        # WARN: 这里都没用到 input_data 这个参数
        for line in self.csv_reader:
            line = line.strip()
            segments = line.split("\t")
            output_dict = {}
            # 构建一个 output_dict, key 是列名, value 是从每行中解析出来的值
            for idx, name in enumerate(self.input_tensor_names):
                output_dict[name] = segments[idx]
            self.put(output_dict)
        raise IndexError("Read tabel done")

    def close(self):
        self.csv_reader.close()

class BundleCSVReader(CSVReader):
    """ Read group of csv formats

    Args:

        input_glob : input file fp
        batch_size : input batch size
        worker_hosts: worker hosts
        task_index: task index
        is_training : True or False

    """
    def __init__(self, input_glob, batch_size, worker_hosts, task_index, is_training=False, **kwargs):
        super(BundleCSVReader, self).__init__(input_glob, batch_size, is_training, **kwargs)

        self.input_fps = []
        # input_glob 是 glob 语法, 允许多文件, 即允许 * 符号
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                # 空的和全数字的都不要
                if line == '' or line.isdigit():
                    continue
                self.input_fps.append(line)
        self.worker_hosts = worker_hosts
        self.task_index = task_index

    def get_input_fn(self):
        def input_fn():
            if self.is_training:
                # 先变成一个常量 tf.constant, 然后转换成 Dataset
                d = tf.data.Dataset.from_tensor_slices(tf.constant(self.input_fps))
                # 分片, 用于分布式训练
                d = d.shard(len(self.worker_hosts.split(',')), self.task_index)
                d = d.repeat()
                d = d.shuffle(buffer_size=len(self.input_fps))

                # 最多 4 次, 并行数量
                cycle_length = min(4, len(self.input_fps))
                d = d.apply(
                    # 并行交错
                    tf.data.experimental.parallel_interleave(
                        tf.data.TextLineDataset,
                        sloppy=True,  # sloppy 允许是乱序输出
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=self.shuffle_buffer_size)
            else:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(self.input_fps))
                d = d.shard(len(self.worker_hosts.split(',')), self.task_index)
                # 只重复一次
                d = d.repeat(1)
                cycle_length = min(4, len(self.input_fps))
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        tf.data.TextLineDataset,
                        sloppy=True,
                        cycle_length=cycle_length))
                # d = tf.data.TextLineDataset(self.input_fps)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                #d = d.repeat()

            d = self._map_batch_prefetch(d, self._decode_csv)
            return d

        return input_fn
