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
import six
from easytransfer.engines.distribution import Process

class CSVWriter(Process):
    """ Writer csv format

    Args:

        output_glob : output file fp
        output_schema : output_schema

    """

    def __init__(self, output_glob, output_schema, input_queue=None, **kwargs):
        job_name = 'DistTableWriter'
        super(CSVWriter, self).__init__(job_name, 1, input_queue)

        self.writer = tf.gfile.Open(output_glob, "w")

        self.output_schema = output_schema

    def close(self):
        tf.logging.info('Finished writing')
        self.writer.close()

    def process(self, features):
        """
        features 看起来是个字典
        因为从 Process 类的文档和代码上看, 如果 batch_size 参数被设置为 1, 那么会将 input_list 的第一个值作为参数
        if self.batch_size == 1:
            out = self.process(input_list[0])
        else:
            out = self.process(input_list)
        """
        def str_format(element):
            # 格式化成字符串
            if isinstance(element, float) or isinstance(element, int) \
                    or isinstance(element, str):
                return str(element)
            # 是空数组就返回空字符串
            if element == []:
                return ''
            # 如果是数组, 且数组里的第一个元素的类型不是数组, 就直接用逗号 join
            if isinstance(element, list) and not isinstance(element[0], list):
                return ','.join([str(t) for t in element])
            # 如果第一个元素是数组, 就把里面的用逗号join, 然后在外面用分号join
            elif isinstance(element[0], list):
                return ';'.join([','.join([str(t) for t in item]) for item in element])
            else:
                raise RuntimeError("type {} not support".format(type(element)))

        ziped_list = []
        # 输出格式 output_schema, 也是用逗号分隔的
        for idx, feat_name in enumerate(self.output_schema.split(",")):
            # 然后从 features 中获取对应的值
            batch_feat_value = features[feat_name]
            curr_list = []
            # 遍历这个值
            for feat in batch_feat_value:
                # 如果是一维的, 就变成一个元素的数组加进去
                if len(batch_feat_value.shape) == 1:
                    curr_list.append([feat])
                else:
                    # 否则调用 tolist 转换成数组加进去
                    curr_list.append(feat.tolist())
            # 然后把整个 curr_list 加入到 ziped_list 中
            ziped_list.append(curr_list)

        # ziped_list 就是有 N 列, 每列是一个 feat_name 的所有结果
        for ele in zip(*ziped_list):
            str_list = []
            for curr in ele:
                str_list.append(str_format(curr))
            # 然后就是每行写入所有列名的值
            self.writer.write("\t".join(str_list) + "\n")

