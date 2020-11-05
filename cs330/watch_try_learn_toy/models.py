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

"""TF models code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def merge_first_n_dims(structure, n):
  def _helper(tensor):
    shape = tf.shape(tensor)
    return tf.reshape(tensor, tf.concat([[-1], shape[n:]], axis=0))
  return tf.nest.map_structure(_helper, structure)


def expand_batch_dims(structure, batch_sizes):
  def _helper(tensor):
    shape = tf.shape(tensor)
    return tf.reshape(tensor, tf.concat([batch_sizes, shape[1:]], axis=0))
  return tf.nest.map_structure(_helper, structure)


def multi_batch_apply(f, num_batch_dims, *args):
  """Applies f to inputs with more than 1 batch dim."""
  batch_sizes = tf.shape(tf.nest.flatten(args)[0])[:num_batch_dims]
  merged_args = merge_first_n_dims(args, num_batch_dims)
  outputs = f(*merged_args)
  return expand_batch_dims(outputs, batch_sizes)


def fc_layer_net(
    input_tensor,
    output_size,
    hidden_layers=(400, 300),
    final_activation=None):
  """Builds a net with fully connected hidden layers."""
  weight_reg = tf.contrib.layers.l2_regularizer(.01)
  output = slim.stack(
      input_tensor,
      slim.fully_connected,
      hidden_layers,
      normalizer_fn=slim.layer_norm,
      weights_regularizer=weight_reg,
      biases_initializer=tf.constant_initializer(.01),
      weights_initializer=tf.truncated_normal_initializer(stddev=.01))
  output = slim.fully_connected(
      output,
      output_size,
      activation_fn=final_activation,
      weights_regularizer=weight_reg,
      biases_initializer=tf.constant_initializer(.01),
      weights_initializer=tf.truncated_normal_initializer(stddev=.01))
  return output
