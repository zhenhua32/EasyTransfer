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


"""
AppZoo是将一些常用的应用模型（如基于BERT/Roberta/ALBERT的文本分类、文本匹配、阅读理解、序列标注等）集中在一个命令中，
用户可以抛去繁琐的配置文件，无需接触代码，只需调用几行命令即可迅速试验相应的模型。本教程给出一个新闻分类BERT模型，让用户快速上手。
更详细AppZoo的介绍见 《AppZoo详解》。

就是说是常见模型的命令行用法
"""


import traceback
import tensorflow as tf

from easytransfer.app_zoo.base import ApplicationModel
from easytransfer.app_zoo.conversion import ConversionModel
from easytransfer.app_zoo.feature_extractor import BertFeatureExtractor
from easytransfer.app_zoo.serialization import SerializationModel
from easytransfer.app_zoo.text_classify import BertTextClassify, TextCNNClassify
from easytransfer.app_zoo.text_match import BertTextMatch, BertTextMatchTwoTower, \
    DAMTextMatch, DAMPlusTextMatch, BiCNNTextMatch, HCNNTextMatch
from easytransfer.app_zoo.text_comprehension import BERTTextComprehension, BERTTextHAEComprehension
from easytransfer.app_zoo.sequence_labeling import BertSequenceLabeling


_name_to_app_model = {
    "serialization": SerializationModel,
    "conversion": ConversionModel,
    "feat_ext_bert": BertFeatureExtractor,
    "text_classify_bert": BertTextClassify,
    "text_classify_cnn": TextCNNClassify,
    "text_match_bert": BertTextMatch,
    "text_match_bert_two_tower": BertTextMatchTwoTower,
    "text_match_dam": DAMTextMatch,
    "text_match_damplus": DAMPlusTextMatch,
    "text_match_bicnn": BiCNNTextMatch,
    "text_match_hcnn": HCNNTextMatch,
    "text_comprehension_bert": BERTTextComprehension,
    "text_comprehension_bert_hae": BERTTextHAEComprehension,
    "sequence_labeling_bert": BertSequenceLabeling
}


def get_application_model(config):
    try:
        assert config.model_name in _name_to_app_model
        tf.compat.v1.logging.info(config.model_name)
        app_model = _name_to_app_model.get(config.model_name)(user_defined_config=config)
        tf.compat.v1.logging.info(app_model)
        return app_model
    except Exception:
        traceback.print_exc()
        raise RuntimeError


