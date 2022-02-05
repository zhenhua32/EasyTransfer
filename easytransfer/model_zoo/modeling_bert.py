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

from easytransfer import layers
from .modeling_utils import PretrainedConfig, PreTrainedModel, MyPreTrainedModel

# 名字和模型路径的映射
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "google-bert-tiny-en": "bert/google-bert-tiny-en/model.ckpt",
    "google-bert-small-en": "bert/google-bert-small-en/model.ckpt",
    "google-bert-base-zh": "bert/google-bert-base-zh/model.ckpt",
    "google-bert-base-en": "bert/google-bert-base-en/model.ckpt",
    "google-bert-large-en": "bert/google-bert-large-en/model.ckpt",
    "pai-bert-tiny-zh-L2-H768-A12": "bert/pai-bert-tiny-zh-L2-H768-A12/model.ckpt",
    "pai-bert-tiny-zh": "bert/pai-bert-tiny-zh/model.ckpt",
    "pai-bert-small-zh": "bert/pai-bert-small-zh/model.ckpt",
    "pai-bert-base-zh": "bert/pai-bert-base-zh/model.ckpt",
    "pai-bert-large-zh": "bert/pai-bert-large-zh/model.ckpt",
}

# 名字和配置文件的映射
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google-bert-tiny-en": "bert/google-bert-tiny-en/config.json",
    "google-bert-small-en": "bert/google-bert-small-en/config.json",
    "google-bert-base-zh": "bert/google-bert-base-zh/config.json",
    "google-bert-base-en": "bert/google-bert-base-en/config.json",
    "google-bert-large-en": "bert/google-bert-large-en/config.json",
    "pai-bert-tiny-zh-L2-H768-A12": "bert/pai-bert-tiny-zh-L2-H768-A12/config.json",
    "pai-bert-tiny-zh": "bert/pai-bert-tiny-zh/config.json",
    "pai-bert-small-zh": "bert/pai-bert-small-zh/config.json",
    "pai-bert-base-zh": "bert/pai-bert-base-zh/config.json",
    "pai-bert-large-zh": "bert/pai-bert-large-zh/config.json",
}


class BertConfig(PretrainedConfig):
    """Configuration for `Bert`.

    Args:

      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      词汇表大小
      hidden_size: Size of the encoder layers and the pooler layer.
      编码层和池化层的大小
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      在 Transformer 编码器中隐藏层的数量
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      在 Transformer 编码器中每个注意力层中注意头的数量
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      在 Transformer 编码器中中间层的大小
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      丢弃注意力的概率
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      模型可能会用到的最大的序列长度
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      token_type_ids 的词汇表长度
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      truncated_normal_initializer 的标准差, 用于初始化所有的权重矩阵
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        max_position_embeddings,
        type_vocab_size,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        **kwargs
    ):
        super(BertConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range


class BertBackbone(layers.Layer):
    """
    bert 的骨架, 也是一个层
    """

    def __init__(self, config: BertConfig, **kwargs):
        """
        config 是 BertConfig 的实例
        """
        # 嵌入层
        self.embeddings = layers.BertEmbeddings(config, name="embeddings")
        # 编码器
        if not kwargs.pop("enable_whale", False):
            # 不开启 whale
            self.encoder = layers.Encoder(config, name="encoder")
        else:
            # 开启 whale
            self.encoder = layers.Encoder_whale(config, name="encoder")

        # 池化器
        self.pooler = layers.Dense(
            units=config.hidden_size,
            activation="tanh",
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="pooler/dense",
        )

        super().__init__(**kwargs)

    def call(self, inputs, input_mask=None, segment_ids=None, training=False):
        """
        前向传播, 返回 编码器和池化层输出的元素
        call 是给 Layer.__call__ 调用的, 可以直接将实例当作函数用
        """
        # 如果是元组或列表, 最多有三个输入
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            # 取出可能的 input_mask 和 segment_ids
            input_mask = inputs[1] if len(inputs) > 1 else input_mask
            segment_ids = inputs[2] if len(inputs) > 2 else segment_ids
        elif isinstance(inputs, tf.Tensor):
            # 增加一种可能性, 如果 inputs 的维度是 3, 且第一个维度表示的是把输入堆叠起来
            # inputs.shape 为 (None, batch_size, seq_length)
            inputs_shape = inputs.shape
            print("input_shape", inputs_shape)
            if len(inputs_shape) == 3:
                inputs_shape_0 = inputs_shape[0]
                input_ids = inputs[0]
                input_mask = inputs[1] if inputs_shape_0 is None or inputs_shape_0 > 1 else input_mask
                segment_ids = inputs[2] if inputs_shape_0 is None or inputs_shape_0 > 1 else segment_ids
            else:
                input_ids = inputs
        else:
            # 不然输入就只有 input_ids
            input_ids = inputs

        # 获取形状的数组
        input_shape = layers.get_shape_list(input_ids)
        print("input_shape", input_shape)
        # 第一个是批次, 第二个是序列长度
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # 如果不存在 input_mask, 就全部初始化成 1
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        # 如果不存在 segment_ids, 就全部初始化成 0
        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        # 嵌入层的输出
        embedding_output = self.embeddings([input_ids, segment_ids], training=training)
        # TODO: 继续, 到这里的输出已经是一样的了, 而且检查点里相关的变量也载入了
        # 注意力的 mask
        attention_mask = layers.get_attn_mask_bert(input_ids, input_mask)
        # 编码器的输出
        encoder_outputs = self.encoder([embedding_output, attention_mask], training=training)
        # 池化层的输出, 参数是 编码器输出的第 0 个的最后一个, 然后第一维度取全部, 第二个维度取第一个
        pooled_output = self.pooler(encoder_outputs[0][-1][:, 0])
        # 最后的输出, 是编码器输出的第 0 个的最后一个 和 池化层输出
        outputs = (encoder_outputs[0][-1], pooled_output)
        return outputs


class BertPreTrainedModel(PreTrainedModel):
    """
    bert 的预训练模型
    """

    # 配置类
    config_class = BertConfig
    # 模型名字和模型 ckpt 路径的映射
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    # 模型名字和 config.json 路径的映射
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(BertPreTrainedModel, self).__init__(config, **kwargs)
        # bert 骨架
        self.bert = BertBackbone(config, name="bert", enable_whale=kwargs.get("enable_whale", False))
        # MLMHead, 预测
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        # NSPHead, 序列关系
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")

    def call(self, inputs, masked_lm_positions=None, **kwargs):
        """
        Args:

            inputs : [input_ids, input_mask, segment_ids]
            masked_lm_positions: masked_lm_positions

        Returns:

            sequence_output, pooled_output

        Examples::


            google-bert-tiny-zh

            google-bert-tiny-en

            google-bert-small-zh

            google-bert-small-en

            google-bert-base-zh

            google-bert-base-en

            google-bert-large-zh

            google-bert-large-en

            pai-bert-tiny-zh

            pai-bert-tiny-en

            pai-bert-small-zh

            pai-bert-small-en

            pai-bert-base-zh

            pai-bert-base-en

            pai-bert-large-zh

            pai-bert-large-en

            model = model_zoo.get_pretrained_model('google-bert-base-zh')
            outputs = model([input_ids, input_mask, segment_ids], mode=mode)

        """
        # 是否是训练模式
        training = kwargs["mode"] == tf.estimator.ModeKeys.TRAIN

        # 是否需要输出特征
        if kwargs.get("output_features", True) == True:
            # bert 的输出
            outputs = self.bert(inputs, training=training)
            # 第一个是序列输出
            sequence_output = outputs[0]
            # 第二个是池化输出
            pooled_output = outputs[1]
            # 这个就是更明确点的, 变成了两个返回值, 原本的 outputs 是个元组
            return sequence_output, pooled_output
        else:
            # 前三行同上
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            # 获取输入的形状
            input_shape = layers.get_shape_list(sequence_output)
            # 第一个维度是批次数量
            batch_size = input_shape[0]
            # 第二个维度是序列长度
            seq_length = input_shape[1]
            # 如果 masked_lm_positions 不存在, 初始化成 1
            if masked_lm_positions is None:
                masked_lm_positions = tf.ones(shape=[batch_size, seq_length], dtype=tf.int64)

            # mlm 的输出, 需要使用 sequence_output 和 masked_lm_positions
            mlm_logits = self.mlm(sequence_output, masked_lm_positions)
            # nsp 的输出, 需要使用 pooled_output
            nsp_logits = self.nsp(pooled_output)

            # 然后返回三个值
            return mlm_logits, nsp_logits, pooled_output


class MyBertPreTrainedModel(MyPreTrainedModel):
    """
    bert 的预训练模型
    """

    # 配置类
    config_class = BertConfig
    # 模型名字和模型 ckpt 路径的映射
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    # 模型名字和 config.json 路径的映射
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        """
        初始化很基础, 就是设置了几个属性, 但是要确保这些都是来自 keras.layer.Layer 的
        """
        super().__init__(config, **kwargs)
        # bert 骨架
        self.bert = BertBackbone(config, name="bert", enable_whale=kwargs.get("enable_whale", False))
        # MLMHead, 预测
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        # NSPHead, 序列关系
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")

    def call(self, inputs, masked_lm_positions=None, **kwargs):
        """
        前向传播是关键
        Args:

            inputs : [input_ids, input_mask, segment_ids]
            masked_lm_positions: masked_lm_positions

        Returns:

            sequence_output, pooled_output

        Examples::


            google-bert-tiny-zh

            google-bert-tiny-en

            google-bert-small-zh

            google-bert-small-en

            google-bert-base-zh

            google-bert-base-en

            google-bert-large-zh

            google-bert-large-en

            pai-bert-tiny-zh

            pai-bert-tiny-en

            pai-bert-small-zh

            pai-bert-small-en

            pai-bert-base-zh

            pai-bert-base-en

            pai-bert-large-zh

            pai-bert-large-en

            model = model_zoo.get_pretrained_model('google-bert-base-zh')
            outputs = model([input_ids, input_mask, segment_ids], mode=mode)

        """
        # 是否是训练模式
        training = bool(kwargs["training"])

        # 是否需要输出特征
        if kwargs.get("output_features", True) is True:
            # bert 的输出
            outputs = self.bert(inputs, training=training)
            # 第一个是序列输出
            sequence_output = outputs[0]
            # 第二个是池化输出
            pooled_output = outputs[1]
            # 这个就是更明确点的, 变成了两个返回值, 原本的 outputs 是个元组
            return sequence_output, pooled_output
        else:
            # 前三行同上
            # 先把输入数据喂给 bert
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            # 获取输入的形状
            input_shape = layers.get_shape_list(sequence_output)
            # 第一个维度是批次数量
            batch_size = input_shape[0]
            # 第二个维度是序列长度
            seq_length = input_shape[1]
            # 如果 masked_lm_positions 不存在, 初始化成 1
            if masked_lm_positions is None:
                masked_lm_positions = tf.ones(shape=[batch_size, seq_length], dtype=tf.int64)

            # mlm 的输出, 需要使用 sequence_output 和 masked_lm_positions
            mlm_logits = self.mlm(sequence_output, masked_lm_positions)
            # nsp 的输出, 需要使用 pooled_output
            nsp_logits = self.nsp(pooled_output)

            # 然后返回三个值
            return mlm_logits, nsp_logits, pooled_output
