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


import json
import re
import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from easytransfer.engines.model import FLAGS
from easytransfer import layers

class PretrainedConfig(object):
    """
    预训练模型的配置
    """
    def __init__(self, **kwargs):
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                tf.compat.v1.logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def get(cls, json_file, **kwargs):
        """
        从 json 文件中实例化
        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        从字典中实例化
        """
        config = cls(**config_dict)
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config

    @classmethod
    def _dict_from_json_file(cls, json_file):
        """
        从 json 文件中获取字典
        """
        with gfile.GFile(json_file, mode='r') as reader:
            text = reader.read()
        return json.loads(text)


class PreTrainedModel(layers.Layer):
    """
    预训练模型, 也是从 Layer 继承的
    """
    # 配置类
    config_class: PretrainedConfig = None
    # 模型名字和模型 ckpt 路径的映射
    pretrained_model_archive_map = {}
    # 模型名字和 config.json 路径的映射
    pretrained_config_archive_map = {}

    @classmethod
    def dummy_inputs(self, seq_length):
        """ Dummy inputs to build the network.
        假的输入
        Returns:
            tf.Tensor with dummy inputs
        """
        #input_ids = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        # 序列长度个 1, 然后外面再套一层数组
        input_ids = [[1]*seq_length]
        return tf.constant(input_ids)

    def __init__(self, config, **kwargs):
        # 这是什么神奇的魔法, 把参数都清空了?
        kwargs.clear()
        super(PreTrainedModel, self).__init__(**kwargs)
        # config 要求是 PretrainedConfig 的实例
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    @classmethod
    def get(cls, pretrained_model_name_or_path, **kwargs):
        """
        获取预训练模型的实例
        """
        # 获取配置文件的路径
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_path = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
            config_path = os.path.join(FLAGS.modelZooBasePath, config_path)
        else:
            config_path = os.path.join(os.path.dirname(pretrained_model_name_or_path), "config.json")

        # 获取配置的实例
        config: PretrainedConfig = cls.config_class.get(
            config_path,
            **kwargs)
        # 实例化模型
        model = cls(config, **kwargs)

        # 先用假的输入调用一次, 看能否调用成功
        model(model.dummy_inputs(kwargs.get('input_sequence_length', 512)), mode='eval', output_features=False)

        # 模型的文件
        archive_file = None
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            archive_file = os.path.join(FLAGS.modelZooBasePath, archive_file)
        elif "/" in pretrained_model_name_or_path:
            archive_file = pretrained_model_name_or_path

        # 如果这个模型文件存在, 就从这个文件中恢复
        if tf.io.gfile.exists(archive_file+".data-00000-of-00001"):
            model._init_from_pretrained_model(archive_file)
        else:
            tf.compat.v1.logging.info("archive file {} does not exists".format(archive_file))
            tf.compat.v1.logging.info("ckpt {} not in model zoo, random initialization".format(pretrained_model_name_or_path))

        return model

    def _init_from_pretrained_model(self, pretrained_model_path):
        """
        从文件中还原预训练模型
        """
        # 所有可训练的变量
        tvars = tf.compat.v1.trainable_variables()
        # 网络名字到变量的映射
        network_name_to_variable = {}
        for var in tvars:
            # 变量的名字
            name = var.name
            # 用 : 分隔的, 前面部分是任意的, 后面是由数字组成的
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                # 获取到名字, 也就是前面部分, 即 (.*) 部分
                # m.group(0) 是正则完全匹配的部分, 1 以后就是分组匹配的部分, 就是括号里的部分
                name = m.group(1)
            network_name_to_variable[name] = var

        try:
            # 检查点读取器
            reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model_path)
            # 获取变量名到形状的映射
            var_to_shape_map = reader.get_variable_to_shape_map()
        except errors_impl.DataLossError:
            raise ImportError(
                '`load_weights` requires correct tf ckpts.')

        # 已经分配的映射
        assignment_map = {}
        for key in var_to_shape_map:
            # 跳过带有这些名字的变量
            if "Adam" in key or "beta1_power" in key or "beta2_power" in key:
                continue
            if "global_step" in key:
                continue

            var = None
            if "pre_trained_model" in key:
                # 如果有 pre_trained_model 字样, 就先用 / 分隔, 取出第一个, 然后加上 /, 将这部分清除掉
                root_key = key.replace(key.split("/")[0]+"/","")
            else:
                root_key = key

            for network_key in network_name_to_variable.keys():
                # 如果 root_key 在 network_key 中
                if root_key in network_key:
                    # 获取到变量
                    var = network_name_to_variable[network_key]
                    break
            # 如果变量不存在, 就提示, 说这个 key 不在可训练的变量中, 也就是 key 不在 network_name_to_variable 中
            if var is None:
                print("Variable: {} in ckpt not in trainable variable".format(key))
                continue
                #raise ValueError("ckpt var name {} not in trainable variable".format(key))
            # key 和 var 的对应关系
            assignment_map[key] = var
        tf.compat.v1.logging.info("Load weights from {}".format(pretrained_model_path))
        # 从检查点初始化
        tf.compat.v1.train.init_from_checkpoint(pretrained_model_path, assignment_map)


def init_from_checkpoint_without_training_ops(pretrained_model_path):
    """
    这个和 PreTrainedModel._init_from_pretrained_model 是一样的, 想不到吧
    """
    tvars = tf.compat.v1.trainable_variables()
    network_name_to_variable = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        network_name_to_variable[name] = var

    try:
        reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
    except errors_impl.DataLossError:
        raise ImportError(
            '`load_weights` requires correct tf ckpts.')

    assignment_map = {}
    for key in var_to_shape_map:
        if "Adam" in key or "beta1_power" in key or "beta2_power" in key:
            continue
        if "global_step" in key:
            continue

        var = None
        if "pre_trained_model" in key:
            root_key = key.replace(key.split("/")[0]+"/","")
        else:
            root_key = key

        for network_key in network_name_to_variable.keys():
            if root_key in network_key:
                var = network_name_to_variable[network_key]
                break
        if var is None:
            print("Variable: {} in ckpt not in trainable variable".format(key))
            continue
            #raise ValueError("ckpt var name {} not in trainable variable".format(key))

        assignment_map[key] = var
    tf.compat.v1.logging.info("Load weights from {}".format(pretrained_model_path))
    tf.compat.v1.train.init_from_checkpoint(pretrained_model_path, assignment_map)
