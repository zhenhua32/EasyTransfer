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

def avgloss_logger_hook(max_steps, loss, model_dir, log_step_count_steps, task_index):
    """
    返回一个类, 这个类初始化时, 无需任何参数
    """
    class _LoggerHook(tf.estimator.SessionRunHook):
        """Logs loss and runtime."""

        def __init__(self):
            # 平均损失
            self.avg_loss = None
            # 最大步长
            self.max_steps = max_steps
            # 衰减率
            self.decay = 0.99
            # 文件写入器
            self.writer = tf.compat.v1.summary.FileWriter(model_dir+"/avg_loss")
            # 每多少个步长记录一次
            self.log_step_count_steps  = log_step_count_steps
            # 任务索引
            self.task_index = task_index

        def begin(self):
            """
            开始时, 初始化 _step
            """
            self._step = -1

        def before_run(self, run_context):
            """
            在运行前
            """
            # 每次运行前加 1
            self._step += 1
            # loss 是函数传入的, 表示要添加到 Session.run() 的参数
            return tf.estimator.SessionRunArgs([loss])

        def after_run(self, run_context, run_values):
            """
            在运行后
            """
            # 更新 avg_loss
            loss_value = run_values.results[0]
            if self.avg_loss is None:
                self.avg_loss = loss_value
            else:
                # 指数移动平均, 当前的取 0.99 然后加上 0.01 * loss_value
                #Exponential Moving Average
                self.avg_loss = self.avg_loss * self.decay + (1 - self.decay) * loss_value

            # 每 log_step_count_steps 个步长记录一次, 仅当 task_index 为 0 时
            if self._step % self.log_step_count_steps == 0 and self.task_index == 0:
                # 计算当前进度
                progress = float(self._step) / self.max_steps * 100.0
                # 收集数据
                summary = tf.compat.v1.Summary()
                # 添加一个 avg_loss 项
                summary.value.add(tag='avg_loss', simple_value=self.avg_loss)
                # 写入文件
                self.writer.add_summary(summary, self._step)
                # 并且在日志中打印
                tf.compat.v1.logging.info(
                    'progress = %.2f%%, avg_loss = %.6f' % (progress, float(self.avg_loss)))

    return _LoggerHook()
