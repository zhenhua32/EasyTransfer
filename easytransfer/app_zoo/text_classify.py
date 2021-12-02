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
from easytransfer import preprocessors, model_zoo
from easytransfer.app_zoo.base import ApplicationModel
from easytransfer.evaluators import classification_eval_metrics, multi_label_eval_metrics, regression_eval_metrics
import easytransfer.layers as layers
from easytransfer.losses import mean_square_error, multi_label_sigmoid_cross_entropy, softmax_cross_entropy
from easytransfer.preprocessors.deeptext_preprocessor import DeepTextPreprocessor


class BaseTextClassify(ApplicationModel):
    """
    基础的文本分类模型
    """
    def __init__(self, **kwargs):
        """ Basic Text Classification Model """
        super(BaseTextClassify, self).__init__(**kwargs)

    @staticmethod
    def default_model_params():
        """ The default value of the Text Classification Model """
        raise NotImplementedError

    def build_logits(self, features, mode=None):
        """ Building graph of the Text Classification Model
        """
        raise NotImplementedError

    def build_loss(self, logits, labels):
        """ Building loss for training the Text Classification Model
        logits: 是预测结果
        labels: 是正确标签
        """
        # 多标签分类
        if hasattr(self.config, "multi_label") and self.config.multi_label:
            return multi_label_sigmoid_cross_entropy(labels, self.config.num_labels, logits)
        elif self.config.num_labels == 1:
            # 类别数只有一个, 比较奇怪, 难道判断是否是正确的类别?
            return mean_square_error(labels, logits)
        else:
            # 多类别
            return softmax_cross_entropy(labels, self.config.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        """ Building evaluation metrics while evaluating

        Args:
            logits (`Tensor`): shape of [None, num_labels]
            labels (`Tensor`): shape of [None]
        Returns:
            ret_dict (`dict`): A dict with (`py_accuracy`, `py_micro_f1`, `py_macro_f1`) tf.metrics op
        """
        # 评估指标, 代码结构类似上面的 build_loss
        if hasattr(self.config, "multi_label") and self.config.multi_label:
            return multi_label_eval_metrics(logits, labels, self.config.num_labels)
        elif self.config.num_labels == 1:
            return regression_eval_metrics(logits, labels)
        else:
            return classification_eval_metrics(logits, labels, self.config.num_labels)

    def build_predictions(self, predict_output):
        """ Building  prediction dict of the Text Classification Model

        Args:
            predict_output (`tuple`): (logits, _) 是从 build_logits 返回的
        Returns:
            ret_dict (`dict`): A dict with (`predictions`, `probabilities`, `logits`)
        """
        # 预测只需要分为多标签 和 单标签
        if hasattr(self.config, "multi_label") and self.config.multi_label:
            return self._build_multi_label_predictions(predict_output)
        else:
            return self._build_single_label_predictions(predict_output)


    def _build_single_label_predictions(self, predict_output):
        logits, _ = predict_output
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        probs = tf.nn.softmax(logits, axis=1)
        ret_dict = {
            "predictions": predictions,
            "probabilities": probs,
            "logits": logits,
        }
        return ret_dict

    def _build_multi_label_predictions(self, predict_output):
        logits, _ = predict_output
        probs = tf.sigmoid(logits)
        predictions = tf.cast(probs > 0.5, tf.int32)
        ret_dict = {
            "predictions": predictions,
            "probabilities": probs,
            "logits": logits,
        }
        return ret_dict


class BertTextClassify(BaseTextClassify):
    """ BERT Text Classification Model

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "pretrain_model_name_or_path": "pai-bert-base-zh",
                "multi_label": False,
                "num_labels": 2,
                "max_num_labels": 5,
                "dropout_rate": 0.1
            }
    """
    def __init__(self, **kwargs):
        super(BertTextClassify, self).__init__(**kwargs)

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids:int:64,input_mask:int:64,segment_ids:int:64,label_id:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids:int:64,input_mask:int:64,segment_ids:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "pretrain_model_name_or_path": "pai-bert-base-zh",
            "multi_label": False,
            "num_labels": 2,
            "max_num_labels": 5,
            "dropout_rate": 0.1
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building graph of BERT Text Classifier

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        """
        前向传播的过程
        """
        # 是否是多标签分类
        multi_label_flag = self.config.multi_label if hasattr(self.config, "multi_label") else False
        # 预处理器
        preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                      multi_label=multi_label_flag,
                                                      user_defined_config=self.config)
        # 将输入 features 进行预处理后的结果
        input_ids, input_mask, segment_ids, labels = preprocessor(features)

        # bert 模型, 然后直接调用前向传播
        bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
        _, pool_output = bert_backbone([input_ids, input_mask, segment_ids], mode=mode)

        # 添加一个 dropout 层
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        pool_output = tf.layers.dropout(
            pool_output, rate=self.config.dropout_rate, training=is_training)
        # 添加一个 dense 层, 输出的长度是标签数
        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='app/ez_dense')(pool_output)

        # 如果可以, 就从检查点加载
        self.check_and_init_from_checkpoint(mode)
        return logits, labels


class TextCNNClassify(BaseTextClassify):
    """ TextCNN Text Classification Model """
    def __init__(self, **kwargs):
        super(TextCNNClassify, self).__init__(**kwargs)
        self.pre_build_vocab = self.config.mode.startswith("train")

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64,label_id:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "max_vocab_size": 30000,
            "embedding_size": 300,
            "num_filters": "100,100,100",
            "filter_sizes": "3,4,5",
            "dropout_rate": 0.5,
            "pretrain_word_embedding_name_or_path": "",
            "fix_embedding": False
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building DAM text match graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        # 预处理器
        text_preprocessor = DeepTextPreprocessor(self.config, mode=mode)
        text_indices, text_masks, _, _, label_ids = text_preprocessor(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 词嵌入层
        word_embeddings = self._add_word_embeddings(vocab_size=text_preprocessor.vocab.size,
                                                    embed_size=self.config.embedding_size,
                                                    pretrained_word_embeddings=text_preprocessor.pretrained_word_embeddings,
                                                    trainable=not self.config.fix_embedding)
        text_embeds = tf.nn.embedding_lookup(word_embeddings, text_indices)

        # textcnn 编码器
        output_features = layers.TextCNNEncoder(num_filters=self.config.num_filters,
                                                filter_sizes=self.config.filter_sizes,
                                                embed_size=self.config.embedding_size,
                                                max_seq_len=self.config.sequence_length,
                                                )([text_embeds, text_masks], training=is_training)

        output_features = tf.layers.dropout(
            output_features, rate=self.config.dropout_rate, training=is_training, name='output_features')

        # 最后的 dense 层
        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='output_layer')(output_features)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids

    def _add_word_embeddings(self, vocab_size, embed_size, pretrained_word_embeddings=None, trainable=False):
        with tf.name_scope("input_representations"):
            # 使用预训练的词嵌入
            if pretrained_word_embeddings is not None:
                tf.logging.info("Initialize word embedding from pretrained")
                # 常量初始化器
                word_embedding_initializer = tf.constant_initializer(pretrained_word_embeddings)
            else:
                word_embedding_initializer = layers.get_initializer(0.02)
            # 得到或创建一个新的 tf 变量
            word_embeddings = tf.get_variable("word_embeddings",
                                              [vocab_size, embed_size],
                                              dtype=tf.float32, initializer=word_embedding_initializer,
                                              trainable=trainable)
        return word_embeddings