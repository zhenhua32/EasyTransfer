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


from collections import OrderedDict
import tensorflow as tf
from .preprocessor import Preprocessor, PreprocessorConfig, truncate_seq_pair
from .tokenization import convert_to_unicode


class ClassificationRegressionPreprocessorConfig(PreprocessorConfig):
    def __init__(self, **kwargs):
        super(ClassificationRegressionPreprocessorConfig, self).__init__(**kwargs)
        """
        定义了需要这些配置
        """
        self.input_schema = kwargs.get("input_schema")
        self.output_schema = kwargs.get("output_schema", None)
        self.sequence_length = kwargs.get("sequence_length")
        self.first_sequence = kwargs.get("first_sequence")
        self.second_sequence = kwargs.get("second_sequence")
        self.label_name = kwargs.get("label_name")
        self.label_enumerate_values = kwargs.get("label_enumerate_values")


class ClassificationRegressionPreprocessor(Preprocessor):
    """ Preprocessor for classification/regression task
    对于分类和回归任务都适用的预处理器
    """
    config_class = ClassificationRegressionPreprocessorConfig

    def __init__(self, config, **kwargs):
        Preprocessor.__init__(self, config, **kwargs)
        self.config = config

        # 定义输入 tensor 的名字, 从 input_schema 中解析
        self.input_tensor_names = []
        for schema in config.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)

        # 定义 label 到 idx 的映射, key 是 label, val 是 idx
        self.label_idx_map = OrderedDict()
        if self.config.label_enumerate_values is not None:
            # label_enumerate_values 是用逗号分隔的
            for (i, label) in enumerate(self.config.label_enumerate_values.split(",")):
                self.label_idx_map[convert_to_unicode(label)] = i

        # 处理多标签分类
        if hasattr(self.config, "multi_label") and self.config.multi_label is True:
            self.multi_label = True
            # 定义最大标签数, 默认值是 5
            self.max_num_labels = self.config.max_num_labels if hasattr(self.config, "max_num_labels") else 5
        else:
            self.multi_label = False
            self.max_num_labels = None

    def set_feature_schema(self):
        """
        设置特征的模式
        """
        # 如果是预测或者预处理, 需要设置输出格式 output_schema
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        # 输出的 tensor 名字
        self.output_tensor_names = ["input_ids", "input_mask", "segment_ids", "label_id"]
        # 多标签处理
        if self.multi_label:
            # 序列长度的列表是 sequence_length 重复三次加上最大标签数, 应该是对应 output_tensor_names 的长度都是 4
            self.seq_lens = [self.config.sequence_length] * 3 + [self.max_num_labels]
            # 特征值的类型的列表
            self.feature_value_types = [tf.int64] * 3 + [tf.int64]
        else:
            # 单标签分类, label_id 的长度就是 1
            self.seq_lens = [self.config.sequence_length] * 3 + [1]
            # 如果标签总数大于等于2, label_id 的类型是int64
            if len(self.label_idx_map) >= 2:
                self.feature_value_types = [tf.int64] * 4
            else:
                # 为什么需要 label_id 的类型是 float32, 是因为有小数吗? 也就是是指回归吗?
                self.feature_value_types = [tf.int64] * 3 + [tf.float32]

    def convert_example_to_features(self, items):
        """ Convert single example to classifcation/regression features
        将字符串类型的样本转换成分类/回归需要的特征
        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`tuple`): (input_ids, input_mask, segment_ids, label_id)
        """
        # 获取第一个序列的文本, first_sequence 是索引位置, input_tensor_names 是列名的列表
        text_a = items[self.input_tensor_names.index(self.config.first_sequence)]
        # tokenizer 是在 PreprocessorConfig 中设置的
        tokens_a = self.config.tokenizer.tokenize(convert_to_unicode(text_a))
        # 如果第二个序列存在
        if self.config.second_sequence in self.input_tensor_names:
            text_b = items[self.input_tensor_names.index(self.config.second_sequence)]
            tokens_b = self.config.tokenizer.tokenize(convert_to_unicode(text_b))
            # 裁剪句子对, 直到长度满足要求
            truncate_seq_pair(tokens_a, tokens_b, self.config.sequence_length - 3)
        else:
            # 同样的, 也要裁剪第一个序列的长度
            if len(tokens_a) > self.config.sequence_length - 2:
                tokens_a = tokens_a[0:(self.config.sequence_length - 2)]
            tokens_b = None

        tokens = []
        segment_ids = []
        # 添加 tokens 的开头
        tokens.append("[CLS]")
        segment_ids.append(0)
        # 然后一个个放入 token
        # 对于第一个序列, segment_ids 全是 0
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        # 添加 tokens 的结尾
        tokens.append("[SEP]")
        segment_ids.append(0)
        # 第一个序列到此结束了

        # 如果有第二个序列
        if tokens_b:
            # 那么 segment_ids 全是 1
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            # 同样的, 添加结尾
            tokens.append("[SEP]")
            segment_ids.append(1)

        # 将 tokens 转换成 id 形式
        input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # 真实 tokens 的 mask 都是 1
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # 如果长度不足, 就用 0 填充全部的 input_ids, input_mask, segment_ids
        while len(input_ids) < self.config.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # 验证数据长度
        assert len(input_ids) == self.config.sequence_length
        assert len(input_mask) == self.config.sequence_length
        assert len(segment_ids) == self.config.sequence_length

        # 如果是有标签的
        if self.config.label_name is not None:
            # 找出标签对应的值
            label_value = items[self.input_tensor_names.index(self.config.label_name)]
            if isinstance(label_value, str) or isinstance(label_value, bytes):
                label = convert_to_unicode(label_value)
            else:
                label = str(label_value)

            # 处理多标签分类
            if self.multi_label:
                # label 是用逗号分隔的, 然后转换成对应的 id 数组
                label_ids = [self.label_idx_map[convert_to_unicode(x)] for x in label.split(",") if x]
                # 只取最大数量的前
                label_ids = label_ids[:self.max_num_labels]
                # 长度不够的, 就需要用 -1 填充了. 总之, 数组的长度等于 self.max_num_labels
                label_ids = label_ids + [-1 for _ in range(self.max_num_labels - len(label_ids))]
                label_ids = [str(t) for t in label_ids]
                # 又变成空格分隔的字符串了
                label_id = ' '.join(label_ids)
            # 如果是类别数量大于等于2
            elif len(self.label_idx_map) >= 2:
                # 找出对应的 id, 转换成字符串
                label_id = str(self.label_idx_map[convert_to_unicode(label)])
            else:
                label_id = label

        else:
            # 没有标签就是 0
            label_id = '0'

        # 前面三项都是变成空格分隔的字符串, 最后的 label_id 其实也是, 在前面已经转换过了
        return ' '.join([str(t) for t in input_ids]), \
               ' '.join([str(t) for t in input_mask]), \
               ' '.join([str(t) for t in segment_ids]), label_id


class PairedClassificationRegressionPreprocessor(ClassificationRegressionPreprocessor):
    """ Preprocessor for paired classification/regression task
    paired 是成对的意思
    居然是直接从上面的 ClassificationRegressionPreprocessor 继承过来的
    """
    config_class = ClassificationRegressionPreprocessorConfig

    def __init__(self, config, **kwargs):
        super(PairedClassificationRegressionPreprocessor, self).__init__(config, **kwargs)

    def set_feature_schema(self):
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        #self.output_tensor_names = ["input_ids", "input_mask", "segment_ids", "label_id"]
        # 因为是成对的, 输出就多了三个 b
        self.output_tensor_names = ["input_ids_a", "input_mask_a", "segment_ids_a",
                                    "input_ids_b", "input_mask_b", "segment_ids_b",
                                    "label_id"]
        # 变成 6 + 1 了
        self.seq_lens = [self.config.sequence_length] * 6 + [1]
        if len(self.label_idx_map) >= 2:
            self.feature_value_types = [tf.int64] * 6 + [tf.int64]
        else:
            # 同样的, 这里是回归
            self.feature_value_types = [tf.int64] * 6 + [tf.float32]

    def convert_example_to_features(self, items):
        """ Convert single example to classifcation/regression features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`tuple`): (input_ids_a, input_mask_a, segment_ids_a,
                                 input_ids_b, input_mask_b, segment_ids_b,
                                 label_id)
        """
        # 这个需要两个序列文本都存在
        assert self.config.first_sequence in self.input_tensor_names \
               and self.config.second_sequence in self.input_tensor_names
        text_a = items[self.input_tensor_names.index(self.config.first_sequence)]
        tokens_a = self.config.tokenizer.tokenize(convert_to_unicode(text_a))

        text_b = items[self.input_tensor_names.index(self.config.second_sequence)]
        tokens_b = self.config.tokenizer.tokenize(convert_to_unicode(text_b))

        # Account for [CLS] and [SEP] with "- 2"
        # 全部都是 序列长度 - 2, 2 就是指 [CLS] 和 [SEP]
        if len(tokens_a) > self.config.sequence_length - 2:
            tokens_a = tokens_a[0:(self.config.sequence_length - 2)]

        if len(tokens_b) > self.config.sequence_length - 2:
            tokens_b = tokens_b[0:(self.config.sequence_length - 2)]

        # 先填充第一个序列文本
        tokens = []
        segment_ids_a = []
        tokens.append("[CLS]")
        segment_ids_a.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids_a.append(0)
        tokens.append("[SEP]")
        segment_ids_a.append(0)

        # 然后把 tokens 转换成 id 格式, 当作第一个序列的
        input_ids_a = self.config.tokenizer.convert_tokens_to_ids(tokens)

        # 开始填充第二个序列文本, 清空了 tokens
        # 然后是这里的 segment_ids_b 全是 1
        tokens = []
        segment_ids_b = []
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids_b.append(1)
            tokens.append("[SEP]")
            segment_ids_b.append(1)

        input_ids_b = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_mask 都是 1
        input_mask_a = [1] * len(input_ids_a)
        input_mask_b = [1] * len(input_ids_b)

        # Zero-pad up to the sequence length.
        # 长度不够的都用 0 填充
        while len(input_ids_a) < self.config.sequence_length:
            input_ids_a.append(0)
            input_mask_a.append(0)
            segment_ids_a.append(0)

        # Zero-pad up to the sequence length.
        # 对于第二个序列, 也是同样的填充方式
        while len(input_ids_b) < self.config.sequence_length:
            input_ids_b.append(0)
            input_mask_b.append(0)
            segment_ids_b.append(0)

        # 验证
        assert len(input_ids_a) == self.config.sequence_length
        assert len(input_mask_a) == self.config.sequence_length
        assert len(segment_ids_a) == self.config.sequence_length
        assert len(input_ids_b) == self.config.sequence_length
        assert len(input_mask_b) == self.config.sequence_length
        assert len(segment_ids_b) == self.config.sequence_length

        # support single/multi classification and regression
        # 有标签的那种
        if self.config.label_name is not None:
            # 标签对应的值
            label_value = items[self.input_tensor_names.index(self.config.label_name)]
            # 转成字符串
            if isinstance(label_value, str) or isinstance(label_value, bytes):
                label = convert_to_unicode(label_value)
            else:
                label = str(label_value)

            # 标签数大于等于 2, 是分类问题
            if len(self.label_idx_map) >= 2:
                # 找出标签对应的 id
                label_id = str(self.label_idx_map[convert_to_unicode(label)])
            else:
                # 否则就是回归问题
                label_id = label
        else:
            label_id = '0'
        return ' '.join([str(t) for t in input_ids_a]), \
               ' '.join([str(t) for t in input_mask_a]), \
               ' '.join([str(t) for t in segment_ids_a]), \
               ' '.join([str(t) for t in input_ids_b]), \
               ' '.join([str(t) for t in input_mask_b]), \
               ' '.join([str(t) for t in segment_ids_b]), label_id
