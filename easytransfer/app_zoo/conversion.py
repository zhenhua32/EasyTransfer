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
import os
import tensorflow as tf
from easytransfer.app_zoo.base import ApplicationModel
from easytransfer.app_zoo.app_utils import copy_file_to_new_path, log_duration_time
from easytransfer.model_zoo.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from easytransfer.optimizers import AdamWeightDecayOptimizer


class ConversionModel(ApplicationModel):
    """ Base Conversion Model"""
    def __init__(self, **kwargs):
        super(ConversionModel, self).__init__(**kwargs)

    @log_duration_time
    def run(self):
        """
        用来在 google bert 和 ali bert 之间转换
        """
        if self.config.export_type == "convert_bert_to_google":
            self.convert_bert_model_to_google()
        elif self.config.export_type == "convert_google_bert_to_ez_transfer":
            self.convert_google_bert_to_ez_transfer()
        else:
            raise NotImplementedError

    def convert_bert_model_to_google(self):
        """ convert EasyTransfer model to google/bert
        转换成 google bert 形式
        """
        pre_defined_model_dirs = set()
        for val in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.values():
            _dir = os.path.dirname(os.path.join(self.config.model_zoo_base_path, val))
            pre_defined_model_dirs.add(_dir)
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        export_dir_base = self.config.export_dir_base
        # 检查是否有 checkpoint 目录
        flag = True
        if checkpoint_dir in pre_defined_model_dirs:
            flag = False

        if not flag:
            raise RuntimeError("Invalid operation")

        prefix = "bert_pre_trained_model/"
        # 这变量都没用上, 是为了测试能否正常加载吗?
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        with tf.compat.v1.Session() as sess:
            # 列出检查点的变量名和变量形状
            for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
                # Load the variable
                # 跳过这些变量
                if "Adam" in var_name or "beta1_power" in var_name or "beta2_power" in var_name:
                    continue
                if "global_step" in var_name:
                    continue
                # 直接加载变量
                var = tf.train.load_variable(checkpoint_dir, var_name)

                # Set the new name
                # 取个新名字
                new_name = var_name
                new_name = new_name.replace(prefix, "")
                # 直接把变量覆盖掉, 就是替换成功了吗?
                var = tf.Variable(var, name=new_name)

            # Save the variables
            # 在循环完所有的变量后, 直接导出就可以了吗?
            saver = tf.compat.v1.train.Saver()
            # 那这一步就是收集到所有变量了?
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, os.path.join(export_dir_base, "bert_model.ckpt"))

        # 复制词汇表和配置文件
        copy_file_to_new_path(checkpoint_dir, export_dir_base, "vocab.txt")
        copy_file_to_new_path(checkpoint_dir, export_dir_base, "config.json", "bert_config.json")

    def convert_google_bert_to_ez_transfer(self):
        """ convert google/bert model to EasyTransfer model

        转换成 EasyTransfer 形式, 基本就是反向操作了
        """
        pre_defined_model_dirs = set()
        for val in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.values():
            _dir = os.path.dirname(os.path.join(self.config.model_zoo_base_path, val))
            pre_defined_model_dirs.add(_dir)
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        export_dir_base = self.config.export_dir_base
        flag = True
        if checkpoint_dir in pre_defined_model_dirs:
            flag = False

        if not flag:
            raise RuntimeError("Invalid operation")

        prefix = "bert_pre_trained_model/"
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        with tf.compat.v1.Session() as sess:
            for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
                # Load the variable
                var = tf.train.load_variable(checkpoint_dir, var_name)

                # Set the new name
                new_name = var_name
                # 名字加上前缀
                new_name = prefix + new_name
                var = tf.Variable(var, name=new_name)

            global_step = tf.compat.v1.train.get_or_create_global_step()
            optimizer = AdamWeightDecayOptimizer(
                learning_rate=1e-5,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
                weight_decay_rate=0.0)

            # Save the variables
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, os.path.join(export_dir_base, "model.ckpt"))

        # 产生新的配置文件
        with tf.io.gfile.GFile(os.path.join(checkpoint_dir, "bert_config.json")) as f:
            config_json = json.load(f)
        new_config_json = {
            "model_type": "bert",
            "vocab_size": config_json["vocab_size"],
            "hidden_size": config_json["hidden_size"],
            "intermediate_size": config_json["intermediate_size"],
            "num_hidden_layers": config_json["num_hidden_layers"],
            "max_position_embeddings": config_json["max_position_embeddings"],
            "num_attention_heads": config_json["num_attention_heads"],
            "type_vocab_size": config_json["type_vocab_size"],
        }
        with tf.io.gfile.GFile(os.path.join(export_dir_base, "config.json"), "w") as f:
            json.dump(new_config_json, f)
        copy_file_to_new_path(checkpoint_dir, export_dir_base, "vocab.txt")
