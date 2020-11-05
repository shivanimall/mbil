# coding=utf-8
# Copyright 2019 The Watch Try Learn Toy Authors.
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

"""Run the meta-learning collect/eval process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import gin
import run_meta_env

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_tasks', 10, 'Number of tasks to run.')
flags.DEFINE_string('input_data_path', None, 'Path to existing task dataset.')
flags.DEFINE_string(
    'output_data_path', '/tmp/example_out.h5', 'Path to existing task dataset.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin bindings.')
flags.DEFINE_multi_string('gin_config', None, 'Path to gin configs.')


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  run_meta_env.run_meta_env(
      input_data_path=FLAGS.input_data_path,
      output_data_path=FLAGS.output_data_path,
      num_tasks=FLAGS.num_tasks,
  )


if __name__ == '__main__':
  app.run(main)
