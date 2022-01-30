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

import functools
import json
import six

import numpy as np
import os
import tensorflow as tf

import easytransfer
from easytransfer.engines.distribution import Process
from easytransfer.engines.model import FLAGS
from .tokenization import FullTokenizer
from easytransfer.model_zoo import get_config_path

# 句子 的 vocab 路径
sentencepiece_model_name_vocab_path_map = {
    'google-albert-base-zh': "albert/google-albert-base-zh/vocab.txt",
    'google-albert-large-zh': "albert/google-albert-large-zh/vocab.txt",
    'google-albert-xlarge-zh': "albert/google-albert-xlarge-zh/vocab.txt",
    'google-albert-xxlarge-zh': "albert/google-albert-xxlarge-zh/vocab.txt",
    'google-albert-base-en': "albert/google-albert-base-en/30k-clean.model",
    'google-albert-large-en': "albert/google-albert-large-en/30k-clean.model",
    'google-albert-xlarge-en': "albert/google-albert-xlarge-en/30k-clean.model",
    'google-albert-xxlarge-en': "albert/google-albert-xxlarge-en/30k-clean.model",
    'pai-albert-base-zh': "albert/pai-albert-base-zh/vocab.txt",
    'pai-albert-large-zh': "albert/pai-albert-large-zh/vocab.txt",
    'pai-albert-xlarge-zh': "albert/pai-albert-xlarge-zh/vocab.txt",
    'pai-albert-xxlarge-zh': "albert/pai-albert-xxlarge-zh/vocab.txt",
    'pai-albert-base-en': "albert/pai-albert-base-en/30k-clean.model",
    'pai-albert-large-en': "albert/pai-albert-large-en/30k-clean.model",
    'pai-albert-xlarge-en': "albert/pai-albert-xlarge-en/30k-clean.model",
    'pai-albert-xxlarge-en': "albert/pai-albert-xxlarge-en/30k-clean.model",
}

# 单词 的 vocab 路径
wordpiece_model_name_vocab_path_map = {
    'google-bert-tiny-zh': "bert/google-bert-tiny-zh/vocab.txt",
    'google-bert-tiny-en': "bert/google-bert-tiny-en/vocab.txt",
    'google-bert-small-zh': "bert/google-bert-small-zh/vocab.txt",
    'google-bert-small-en': "bert/google-bert-small-en/vocab.txt",
    'google-bert-base-zh': "bert/google-bert-base-zh/vocab.txt",
    'google-bert-base-en': "bert/google-bert-base-en/vocab.txt",
    'google-bert-large-zh': "bert/google-bert-large-zh/vocab.txt",
    'google-bert-large-en': "bert/google-bert-large-en/vocab.txt",
    'pai-bert-tiny-zh-L2-H768-A12': "bert/pai-bert-tiny-zh-L2-H768-A12/vocab.txt",
    'pai-bert-tiny-zh-L2-H128-A2': "bert/pai-bert-tiny-zh-L2-H128-A2/vocab.txt",
    'pai-bert-tiny-en': "bert/pai-bert-tiny-en/vocab.txt",
    'pai-bert-tiny-zh': "bert/pai-bert-tiny-zh/vocab.txt",
    'pai-bert-small-zh': "bert/pai-bert-small-zh/vocab.txt",
    'pai-bert-small-en': "bert/pai-bert-small-en/vocab.txt",
    'pai-bert-base-zh': "bert/pai-bert-base-zh/vocab.txt",
    'pai-bert-base-en': "bert/pai-bert-base-en/vocab.txt",
    'pai-bert-large-zh': "bert/pai-bert-large-zh/vocab.txt",
    'pai-bert-large-en': "bert/pai-bert-large-en/vocab.txt",
    'hit-roberta-base-zh': "roberta/hit-roberta-base-zh/vocab.txt",
    'hit-roberta-large-zh': "roberta/hit-roberta-large-zh/vocab.txt",
    'pai-imagebert-base-zh': "imagebert/pai-imagebert-base-zh/vocab.txt",
    'pai-videobert-base-zh': "imagebert/pai-videobert-base-zh/vocab.txt",
    'brightmart-roberta-small-zh': "roberta/brightmart-roberta-small-zh/vocab.txt",
    'brightmart-roberta-large-zh': "roberta/brightmart-roberta-large-zh/vocab.txt",
    'icbu-imagebert-small-en': "imagebert/icbu-imagebert-small-en/vocab.txt",
    'pai-transformer-base-zh': "transformer/pai-transformer-base-zh/vocab.txt",
    'pai-linformer-base-en': "linformer/pai-linformer-base-en/vocab.txt",
    'pai-xformer-base-en': "xformer/pai-xformer-base-en/vocab.txt",
    'pai-imagebert-base-en': "imagebert/pai-imagebert-base-en/vocab.txt",
    'pai-synthesizer-base-en': "synthesizer/pai-synthesizer-base-en/vocab.txt",
    'pai-sentimentbert-base-zh': "sentimentbert/pai-sentimentbert-base-zh/vocab.txt"
}


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    """
    裁剪句子对
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        # 一个个单词去除, 先从长的开始去除, 每次从尾部删除一个单词
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class PreprocessorConfig(object):
    """
    预处理器的配置
    """
    def __init__(self, **kwargs):

        self.mode = kwargs.get("mode")

        # multitask classification
        self.append_feature_columns = kwargs.get("append_feature_columns")

        # configurate tokenizer
        pretrain_model_name_or_path = kwargs['pretrain_model_name_or_path']

        # 如果路径中没有 / 说明不是一个文件路径, 而是预定义的模型的名称
        if "/" not in pretrain_model_name_or_path:
            # - 分隔的第二个位置是模型类型 model_type
            model_type = pretrain_model_name_or_path.split("-")[1]
            # 检查配置文件是否存在, 如果存在, 就跳过. 否则就要下载模型
            if tf.io.gfile.exists(os.path.join(FLAGS.modelZooBasePath, model_type,
                                                pretrain_model_name_or_path, "config.json")):
                # If exists directory, not download
                pass
            else:
                # python 2 版本? 没想到啊, 一开始居然是支持 python 2.7 的
                if six.PY2:
                    import errno
                    def mkdir_p(path):
                        try:
                            os.makedirs(path)
                        except OSError as exc:  # Python >2.5 (except OSError, exc: for Python <2.5)
                            if exc.errno == errno.EEXIST and os.path.isdir(path):
                                pass
                            else:
                                raise
                    mkdir_p(os.path.join(FLAGS.modelZooBasePath, model_type))
                else:
                    # 创建完整的目录
                    os.makedirs(os.path.join(FLAGS.modelZooBasePath, model_type), exist_ok=True)

                # 目标路径, 是 pretrain_model_name_or_path + .tgz 的文件
                des_path = os.path.join(os.path.join(FLAGS.modelZooBasePath, model_type),
                                        pretrain_model_name_or_path + ".tgz")
                if not os.path.exists(des_path):
                    tf.compat.v1.logging.info("********** Begin to download to {} **********".format(des_path))
                    # 没想到是直接调用命令行工具下载和解压的
                    os.system(
                        'wget -O ' + des_path + ' https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_modelzoo/' + model_type + '/' + pretrain_model_name_or_path + ".tgz")
                    os.system('tar -zxvf ' + des_path + ' --directory ' + FLAGS.modelZooBasePath + "/" + model_type)

        # 如果是包含 train 的模式
        if "train" in self.mode:
            model_dir = kwargs['model_dir']
            assert model_dir is not None, "model_dir should be set in config"
            # 同样的, 组合出模型所在的目录 dir_path
            if "/" not in pretrain_model_name_or_path:
                model_type = pretrain_model_name_or_path.split("-")[1]
                config_path = get_config_path(model_type, pretrain_model_name_or_path)
                config_path = os.path.join(FLAGS.modelZooBasePath, config_path)
                # 配置文件的目录就是 dir_path
                dir_path = os.path.dirname(config_path)
            else:
                dir_path = os.path.dirname(pretrain_model_name_or_path)

            # 目录不存在, 就先创建剥
            if not tf.io.gfile.exists(model_dir):
                tf.io.gfile.makedirs(model_dir)

            # 如果 model_dir 中配置文件 config.json 不存在, 就从 dir_path 中复制配置文件和词汇表
            if not tf.io.gfile.exists(os.path.join(model_dir, "config.json")):
                tf.io.gfile.copy(os.path.join(dir_path, "config.json"),
                              os.path.join(model_dir, "config.json"))
                # 上面的 config.json 是一定有的, 直接复制过去
                # 而词汇表, 因为文件名不统一, 所以是用 Exists 试探下再复制过去
                if tf.io.gfile.exists(os.path.join(dir_path, "vocab.txt")):
                    tf.io.gfile.copy(os.path.join(dir_path, "vocab.txt"),
                                  os.path.join(model_dir, "vocab.txt"))
                if tf.io.gfile.exists(os.path.join(dir_path, "30k-clean.model")):
                    tf.io.gfile.copy(os.path.join(dir_path, "30k-clean.model"),
                                  os.path.join(model_dir, "30k-clean.model"))

        albert_language = "zh"
        # 当使用预定义的模型名称时, 该如何获取词汇表的路径
        if "/" not in pretrain_model_name_or_path:
            model_type = pretrain_model_name_or_path.split("-")[1]
            # albert 用的是句子级别的词汇表 sentencepiece_model_name_vocab_path_map
            if model_type == "albert":
                vocab_path = os.path.join(FLAGS.modelZooBasePath,
                                          sentencepiece_model_name_vocab_path_map[pretrain_model_name_or_path])
                if "30k-clean.model" in vocab_path:
                    albert_language = "en"
                else:
                    albert_language = "zh"
            else:
                # 反之就是单词级别的词汇表 wordpiece_model_name_vocab_path_map
                vocab_path = os.path.join(FLAGS.modelZooBasePath,
                                          wordpiece_model_name_vocab_path_map[pretrain_model_name_or_path])

        else:
            # 否则, 就需要从 config.json 中读取 model_type
            with tf.io.gfile.GFile(os.path.join(os.path.dirname(pretrain_model_name_or_path), "config.json")) as reader:
                text = reader.read()
            json_config = json.loads(text)
            model_type = json_config["model_type"]
            if model_type == "albert":
                # 如果 vocab.txt 存在, 就读取这个文件, 并将 albert_language 设置为 zh
                if tf.io.gfile.exists(os.path.join(os.path.dirname(pretrain_model_name_or_path), "vocab.txt")):
                    albert_language = "zh"
                    vocab_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "vocab.txt")
                elif tf.io.gfile.exists(os.path.join(os.path.dirname(pretrain_model_name_or_path), "30k-clean.model")):
                    # 否则, 就是 en, 同时要读取 30k-clean.model
                    albert_language = "en"
                    vocab_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "30k-clean.model")
            else:
                vocab_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "vocab.txt")

        assert model_type is not None, "you must specify model_type in pretrain_model_name_or_path"

        if model_type == "albert":
            # albert 的 en 语言有点特殊的, 这个类的大部分 if else 都是它产生的
            if albert_language == "en":
                self.tokenizer = FullTokenizer(spm_model_file=vocab_path)
            else:
                self.tokenizer = FullTokenizer(vocab_file=vocab_path)
        else:
            self.tokenizer = FullTokenizer(vocab_file=vocab_path)

        # Additional attributes without default values
        # 将所有的关键字参数都添加到 self 上
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                tf.compat.v1.logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_json_file(cls, **kwargs):
        """
        这个类方法本质上也是初始化实例, 没啥特殊的, 不知道为什么要加这个
        """
        config = cls(**kwargs)
        return config


# 这个 easytransfer.layers.Layer 就是 from tensorflow.python.layers.base import Layer
class Preprocessor(easytransfer.layers.Layer, Process):
    """
    预处理器
    """
    def __init__(self,
                 config,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTPreprocessor',
                 **kwargs):

        kwargs.clear()
        easytransfer.layers.Layer.__init__(self, **kwargs)

        # 预测和预处理的 batch_size 从不同的配置项中获取
        if config.mode.startswith("predict"):
            Process.__init__(
                self, job_name, thread_num, input_queue, output_queue, batch_size=config.predict_batch_size)

        elif config.mode == "preprocess":
            Process.__init__(
                self, job_name, thread_num, input_queue, output_queue, batch_size=config.preprocess_batch_size)

        # 如果有额外添加的列, 将这些名字添加进去
        self.append_tensor_names = []
        if hasattr(config, "append_feature_columns") and config.append_feature_columns is not None:
            # 用逗号分隔的
            for schema in config.append_feature_columns.split(","):
                # 名字是用冒号分隔的第一个
                name = schema.split(":")[0]
                self.append_tensor_names.append(name)

        self.mode = config.mode

    @classmethod
    def get_preprocessor(cls, **kwargs):
        """
        获取预处理器
        """
        # 如果有 user_defined_config, 就用这个当作配置
        # 所有的配置都是为了添加到 kwargs 中
        if kwargs.get("user_defined_config", None) is not None:
            config = kwargs["user_defined_config"]
            for key, val in config.__dict__.items():
                kwargs[key] = val
            if kwargs["mode"] == "export":
                kwargs["input_schema"] = config.input_tensors_schema
        else:
            # 从 FLAGS.config 中获取配置文件的路径
            json_file = FLAGS.config
            with tf.io.gfile.GFile(json_file, mode='r') as reader:
                text = reader.read()

            config_dict = json.loads(text)
            for values in config_dict.values():
                # 跳过值为 str 类型的
                if isinstance(values, str):
                    continue
                # 这是要求值是个字典
                for k, v in values.items():
                    # 跳过 v 是个字典, 且 k 是 label_enumerate_values 的
                    if isinstance(v, dict) and k != "label_enumerate_values":
                        continue
                    else:
                        kwargs[k] = v
            kwargs["mode"] = FLAGS.mode
            # export 也是和上面类似的处理, 是从 input_tensors_schema 中获取的
            if FLAGS.mode == "export":
                kwargs["input_schema"] = config_dict['export_config']['input_tensors_schema']

        # 然后就可以用 kwargs 重新构建 config 了
        # cls.config_class 果然是在子类里定义的, 基本上都是 PreprocessorConfig 的子类
        config = cls.config_class.from_json_file(**kwargs)

        # 调用自身, 初始化实例
        preprocessor = cls(config, **kwargs)
        return preprocessor

    def set_feature_schema(self):
        raise NotImplementedError("must be implemented in descendants")

    def convert_example_to_features(self, items):
        raise NotImplementedError("must be implemented in descendants")

    def _convert(self, convert_example_to_features, *args):
        """
        转换
        convert_example_to_features 是个函数
        """
        # 只有 on_the_fly 系列的模式和 preprocess 模式可以使用这个方法
        # mode check
        if not ("on_the_fly" in self.mode or self.mode == "preprocess"):
            raise ValueError("Please using on_the_fly or preprocess mode")

        batch_features = []  # 一个批次需要的数量
        batch_size = len(args[0])  # 第一个 args 的长度是批次数量, args[0] 是
        # 遍历每一个批次
        for i in range(batch_size):
            # items 是一个样本
            items = []
            # feat 是一个特征列, 这里是遍历所有的特征列, 合并成一个样本
            for feat in args:
                # 如果 feat[i] 是 np.ndarray, 那么 feat[i][0] 就不能是 None
                if isinstance(feat[i], np.ndarray):
                    assert feat[i][0] is not None, "In on the fly mode where object is ndarray, column has null value"
                    items.append(feat[i][0])
                else:
                    assert feat[i] is not None, "In on the fly mode, column has null value"
                    items.append(feat[i])
            # 然后调用 convert_example_to_features 将样本转换格式
            features = convert_example_to_features(items)
            batch_features.append(features)

        # batch_features 里面有 n 个样本, 每个样本有 m 个特征
        # 那么堆叠起来, axi=1 表示的形状就是 (m, n)
        # 每一行是由不同的样本的同一个特征组成
        stacked_features = np.stack(batch_features, axis=1)
        concat_features = []
        # 遍历不同的特征, shape[0] 就是 m
        for i in range(stacked_features.shape[0]):
            # 将不同的样本的同一个特征, 用空格连接起来, 然后类型转换成 batch_features
            concat_features.append(np.asarray(" ".join(stacked_features[i])))
        return concat_features

    # Inputs from Reader's map_batch_prefetch method
    def call(self, inputs):
        """
        调用
        """
        self.set_feature_schema()

        # 按名字从 inputs 中取, 都添加进去
        items = []
        # 这个 input_tensor_names 也是在子类中设置的
        for name in self.input_tensor_names:
            items.append(inputs[name])

        # 对模式有要求, 非 on_the_fly 系列的和 preprocess 的都可以直接返回结果了
        if not ("on_the_fly" in self.mode or self.mode == "preprocess"):
            return items

        # 这是定义了输出的结构, 数组的长度为 len(self.seq_lens), 类型都是 tf.string
        # seq_lens 也不是在这个类, 或者它的父类中定义的, 又是给子类定义用的
        self.Tout = [tf.string] * len(self.seq_lens)

        # partial 组装了一个新函数, 预先定义了 convert_example_to_features 参数
        # py_func 定义了一个 python 函数, 然后将它封装为 tf op 操作
        # 这个函数的输入是 items, 输出结构是 self.Tout
        batch_features = tf.compat.v1.py_func(functools.partial(self._convert,
                                                      self.convert_example_to_features),
                                    items, self.Tout)

        ret = []
        # 循环 self._convert 的输出结果
        for idx, feature in enumerate(batch_features):
            # 序列长度
            seq_len = self.seq_lens[idx]
            # 特征类型
            feature_type = self.feature_value_types[idx]
            # 按类型处理
            if feature_type == tf.int64:
                input_tensor = tf.strings.to_number(
                    # 在最前面还添加了一个轴, 变成了 (1, )
                    tf.compat.v1.string_split(tf.expand_dims(feature, axis=0), delimiter=" ").values,
                    tf.int64)
            elif feature_type == tf.float32:
                input_tensor = tf.strings.to_number(
                    tf.compat.v1.string_split(tf.expand_dims(feature, axis=0), delimiter=" ").values,
                    tf.float32)
            elif feature_type == tf.string:
                input_tensor = feature
            else:
                raise NotImplementedError

            input_tensor = tf.reshape(input_tensor, [-1, seq_len])
            ret.append(input_tensor)

        # 新增的几个
        for name in self.append_tensor_names:
            ret.append(inputs[name])

        return ret

    def process(self, inputs):
        """
        处理
        """
        self.set_feature_schema()

        # 如果类型是字典, 就先变成单元素的数组
        if isinstance(inputs, dict):
            inputs = [inputs]

        # 构建批次需要的样本
        batch_features = []
        for input in inputs:
            items = []
            # 又是取出了需要的特征
            for name in self.input_tensor_names:
                items.append(input[name])
            features = self.convert_example_to_features(items)
            batch_features.append(features)

        # 这个堆叠也是似曾相识. 每一行都是不同样本的同一个特征
        stacked_features = np.stack(batch_features, axis=1)
        concat_features = []
        for i in range(stacked_features.shape[0]):
            # 每一行都是不同样本的同一个特征
            concat_features.append(np.asarray(" ".join(stacked_features[i])))

        # 如果是预测或者预处理
        if self.mode.startswith("predict") or self.mode == "preprocess":
            # 输出模式是按逗号分隔的名字
            for name in self.output_schema.split(","):
                # 如果名字在 input_tensor_names 中, 就添加到 output_tensor_names 中
                if name in self.input_tensor_names:
                    self.output_tensor_names.append(name)

        ret = {}
        # 循环输出tensor的名字
        for idx, name in enumerate(self.output_tensor_names):
            # 当索引小于特征数时
            if idx < len(concat_features):
                # 特征
                feature = concat_features[idx]
                # 序列长度
                seq_len = self.seq_lens[idx]
                # 特征类型
                feature_type = self.feature_value_types[idx]
                # 变成数组
                feature = feature.tolist()
                # 将特征从按空格分隔的字符串转变成数组
                if feature_type == tf.int64:
                    input_tensor = [int(x) for x in feature.split(" ")]
                elif feature_type == tf.float32:
                    input_tensor = [float(x) for x in feature.split(" ")]
                elif feature_type == tf.string:
                    input_tensor = feature
                else:
                    raise NotImplementedError
                input_tensor = np.reshape(input_tensor, [-1, seq_len])
                # 你确定不是在逗我, 这个 name 不就是前面的 idx, name 中的 name 吗?
                name = self.output_tensor_names[idx]
                ret[name] = input_tensor
            else:
                left = []
                # inputs 中的每个元素, 将对应名字的值加进去
                for ele in inputs:
                    left.append(ele[name])
                # 然后变成一个 np.array, 同时还要 reshape 变换形状 (n, 1), n 是 inputs 的数量
                left_tensor = np.asarray(left)
                ret[name] = np.reshape(left_tensor, [-1, 1])

        # 返回一个字典, 字典里的 key 来自于 output_tensor_names
        return ret
