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
import random
from collections import OrderedDict
import collections
from .tokenization import convert_to_unicode
from .preprocessor import Preprocessor, PreprocessorConfig, truncate_seq_pair

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_chinese_subwords(segment):
    """
    创建中文的子词
    """
    # 片段
    new_segment = []
    # 一个个词取
    for token in segment:
        # 如果是中文词汇
        if u'\u4e00' > token[0] or token[0] > u'\u9fa5':
            new_segment.append(token)
        else:
            if len(token) > 1:
                # 添加第一个字母
                new_segment.append(token[0])
                # 对于后续的每个字母
                for ele in token[1:]:
                    # 添加 ## 加上这个字母
                    new_segment.append("##"+ele)
            else:
                # 单个字母直接添加
                new_segment.append(token)
    return new_segment

def create_int_feature(values):
    """
    创建整数特征
    """
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    """
    创建浮点数特征
    """
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, do_whole_word_mask, rng):
    """Creates the predictions for the masked LM objective.
    创建对掩码的 LM 对象的预测
    全称是 Masked Language Models, 掩码语言模型
    """
    # candidate 候选索引的集合
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        # 跳过特殊字符
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.

        # Whole Word Masking 意味着如果我们屏蔽一个原始单词的所有 wordpieces.
        # 当一个单词被切成 wordpieces 时, 第一个 token 没有任何掩码, 且任何其他子序列的 token 都会以 ## 前缀开始.
        # 所以无论何时我们看到 ## token, 我们将它添加到先前的 word indexes 集合中.
        # 
        # 注意 Whole Word Masking 不会改变任何的训练代码. 我们仍然独立的预测每一个 WordPiece, 在整个词汇表上进行 softmaxed.

        # 如果进行 do_whole_word_mask, 且 cand_indexes 的长度大于等于 1, 且 token 以 ## 开头
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            # 将它添加到 cand_indexes 的最后一个项上
            cand_indexes[-1].append(i)
        else:
            # 既然一个单词的第一个 wordpieces 肯定不是 ## 开头的, 所以 cand_indexes 一开始就是这个单词的 wordpieces
            # 添加一个数组, 只包含一个元素 i
            cand_indexes.append([i])

    # 随机索引数组
    rng.shuffle(cand_indexes)

    # 要输出的 tokens
    output_tokens = list(tokens)

    # 需要预测的数量
    # max_predictions_per_seq 是每个序列的最大预测数
    # masked_lm_prob 掩码 lm 的概率
    # tokens 的数量 * 掩码 lm 的概率, 四舍五入取最大值
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # 掩码的 lms
    masked_lms = []
    # 覆盖的索引
    covered_indexes = set()
    # 对于每一个索引集合
    for index_set in cand_indexes:
        # 如果大于等于需要预测的数量, 就跳过
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        # 如果添加整个 whole-word mask 会超出最大数量, 就直接跳过了
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        # 是否又任何索引被覆盖
        is_any_index_covered = False
        # 遍历索引集合中的每个索引
        for index in index_set:
            # 如果这个索引已经被覆盖了, 就跳出循环
            if index in covered_indexes:
                is_any_index_covered = True
                break
        # 有任一一个索引被覆盖了, 就跳过
        if is_any_index_covered:
            continue
        # 将每个索引添加到被覆盖的索引集合中
        for index in index_set:
            covered_indexes.add(index)

            # 掩码的 token
            masked_token = None
            # 80% of the time, replace with [MASK]
            # 有 80% 的概率被替换成 [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 有 10% 的概率保持原样
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    # 有 10% 的概率, 替换成词汇表中的任意单词
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            # 将输出 tokens 的 index 位置上的替换成 masked_token
            output_tokens[index] = masked_token

            # 添加索引和token 组
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    # 要求数量满足要求
    assert len(masked_lms) <= num_to_predict
    # 按索引顺序排序
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    # 掩码的位置, 即索引
    masked_lm_positions = []
    # 掩码的标签, 即token
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    # 最后返回一个元组, 包含三个项, 输出的 tokens, 掩码的位置, 掩码的标签
    return (output_tokens, masked_lm_positions, masked_lm_labels)


class TrainingInstance(object):
    """A single training instance (sentence pair).
    训练实例
    """

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


class PretrainPreprocessorConfig(PreprocessorConfig):
    """
    预训练的配置
    """

    def __init__(self, **kwargs):
        super(PretrainPreprocessorConfig, self).__init__(**kwargs)

        self.input_schema = kwargs.get("input_schema")
        self.sequence_length = kwargs.get("sequence_length")
        self.first_sequence = kwargs.get("first_sequence")
        self.second_sequence = kwargs.get("second_sequence")
        self.label_name = kwargs.get("label_name")
        self.label_enumerate_values = kwargs.get("label_enumerate_values")
        self.max_predictions_per_seq = kwargs.get("max_predictions_per_seq")
        # 掩码概率默认值是 0.15
        self.masked_lm_prob = kwargs.get("masked_lm_prob", 0.15)
        self.do_whole_word_mask = kwargs.get("do_whole_word_mask", True)


class PretrainPreprocessor(Preprocessor):
    """
    预训练的预处理器
    """

    config_class = PretrainPreprocessorConfig

    def __init__(self, config, **kwargs):
        Preprocessor.__init__(self, config, **kwargs)
        self.config = config

        # 从 input_schema 中解析输入列的名字
        self.input_tensor_names = []
        for schema in config.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)

        # 词汇表
        self.vocab_words = list(self.config.tokenizer.vocab.keys())

        # 随机数生成器
        self.rng = random.Random(12345)

        # label => idx 的映射
        self.label_idx_map = OrderedDict()
        if self.config.label_enumerate_values is not None:
            # 逗号分隔的
            for (i, label) in enumerate(self.config.label_enumerate_values.split(",")):
                self.label_idx_map[convert_to_unicode(label)] = i

        # 特征类型
        self.feature_type = kwargs.get('feature_type', "pretrain_lm")


    def set_feature_schema(self):
        """
        设置特征模式
        """
        # 预测或者预处理, 需要设置 output_schema
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        # 输出的 tensor 名字
        self.output_tensor_names = ["input_ids", "input_mask", "segment_ids",
                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]

        # 如果是预训练语言模型
        if self.feature_type == "pretrain_lm":
            # 输出的类型都是 tf.string
            self.Tout = [tf.string] * 6
            # 长度分别是 3 个序列长度, 3 个 每个序列的最大预测数
            self.seq_lens = [self.config.sequence_length] * 3 + [self.config.max_predictions_per_seq] * 3
            # 特征类型是 5 个 int64, 一个 float32
            self.feature_value_types = [tf.int64] * 5 + [tf.float32]
        elif self.feature_type == "pretrain_multimodel":
            # 如果是预训练多模型
            # 需要 7 个输出
            self.Tout = [tf.string] * 7
            # 最后一个的序列长度是 4
            self.seq_lens = [self.config.sequence_length] * 3 + [self.config.max_predictions_per_seq] * 3 + [4]
            # 最后一个的类型是 int64
            self.feature_value_types = [tf.int64] * 5 + [tf.float32] + [tf.int64]

    def convert_example_to_features(self, items):
        """
        将样本转换成特征
        """
        # 第一个序列文本
        text_a = items[self.input_tensor_names.index(self.config.first_sequence)]
        # 转换成 tokens
        tokens_a = self.config.tokenizer.tokenize(convert_to_unicode(text_a))
        if self.config.second_sequence in self.input_tensor_names:
            # 如果第二个序列文本存在
            text_b = items[self.input_tensor_names.index(self.config.second_sequence)]
            tokens_b = self.config.tokenizer.tokenize(convert_to_unicode(text_b))
            truncate_seq_pair(tokens_a, tokens_b, self.config.sequence_length - 3)
        else:
            # 否则只裁剪第一个序列文本
            if len(tokens_a) > self.config.sequence_length - 2:
                tokens_a = tokens_a[0:(self.config.sequence_length - 2)]
            tokens_b = None

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        # 添加第一个文本
        for token in tokens_a:
            tokens.append(token)
            # 第一个文本的 segment_ids 全是 0
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # 有第二个文本, 就添加
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                # 第二个文本的 segment_ids 全是 1
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        # 转换成 id
        input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # 输入 mask 都是 1, 长度是 input_ids 的长度
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # 长度不足, 全部是用 0 填充
        while len(input_ids) < self.config.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # 验证长度满足要求
        assert len(input_ids) == self.config.sequence_length
        assert len(input_mask) == self.config.sequence_length
        assert len(segment_ids) == self.config.sequence_length

        # 创建掩码语言模型的预测
        # 分别是
        # tokens 单词列表
        # masked_lm_positions 掩码的位置
        # masked_lm_labels 掩码的标签
        tokens, masked_lm_positions, masked_lm_labels = \
            create_masked_lm_predictions(tokens, self.config.masked_lm_prob,
                                     self.config.max_predictions_per_seq, self.vocab_words, self.config.do_whole_word_mask, self.rng)

        # 又转换了一次?
        input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config.sequence_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == self.config.sequence_length
        assert len(input_mask) == self.config.sequence_length
        assert len(segment_ids) == self.config.sequence_length

        # 位置变成数组, labels 变成 id 格式
        masked_lm_positions = list(masked_lm_positions)
        masked_lm_ids = self.config.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        # 权重初始化成 1.0
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        # 如果数量超出了, 只取前面的部分
        if len(masked_lm_positions) >= self.config.max_predictions_per_seq:
            masked_lm_positions = masked_lm_positions[0:self.config.max_predictions_per_seq]
            masked_lm_ids = masked_lm_ids[0:self.config.max_predictions_per_seq]
            masked_lm_weights = masked_lm_weights[0:self.config.max_predictions_per_seq]

        # 数量不足, 就用 0 补充
        while len(masked_lm_positions) < self.config.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)
        # 总之, 数量要刚好和 max_predictions_per_seq 相等

        # 根据 特征类型 返回不同的结果
        if self.feature_type == "pretrain_lm":
            return ' '.join([str(t) for t in input_ids]), \
                   ' '.join([str(t) for t in input_mask]), \
                   ' '.join([str(t) for t in segment_ids]), \
                   ' '.join([str(t) for t in masked_lm_positions]), \
                   ' '.join([str(t) for t in masked_lm_ids]), \
                   ' '.join([str(t) for t in masked_lm_weights])

        elif self.feature_type == "pretrain_multimodel":
            # 前面 6 个是一样的, 最后一个是 长度为 4 的数组, 数组里的元素是从 0-9 中随机取出的, 然后变成空格分隔的字符串
            return ' '.join([str(t) for t in input_ids]), \
                   ' '.join([str(t) for t in input_mask]), \
                   ' '.join([str(t) for t in segment_ids]), \
                   ' '.join([str(t) for t in masked_lm_positions]), \
                   ' '.join([str(t) for t in masked_lm_ids]), \
                   ' '.join([str(t) for t in masked_lm_weights]), \
                   ' '.join(sorted(random.sample([str(x) for x in range(10)], 4)))