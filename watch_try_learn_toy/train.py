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

"""Train MIL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags
import gin
import tensorflow as tf
import tensorflow_probability as tfp
from watch_try_learn_toy import data_processing
from watch_try_learn_toy import models
from watch_try_learn_toy import policies
from watch_try_learn_toy import run_meta_env

FLAGS = flags.FLAGS
HOME = os.path.expanduser('~')

flags.DEFINE_string(
    'model_dir',
    os.path.expanduser('~/meta_reacher/example_train'),
    'Model output directory')
flags.DEFINE_string(
    'train_data_path',
    '{:s}/meta_reacher/datasets/randomtrial_easy_train.h5'.format(HOME),
    'Path to train dataset.')
flags.DEFINE_string(
    'eval_data_path',
    '{:s}/meta_reacher/datasets/randomtrial_easy_eval.h5'.format(HOME),
    'Path to eval dataset.')
flags.DEFINE_integer(
    'train_summary_period', 100, 'Write train summary after this many steps.')
flags.DEFINE_integer(
    'eval_summary_period', 500, 'Write eval summary after this many steps.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin bindings.')
flags.DEFINE_multi_string('gin_config', None, 'Path to gin configs.')


def custom_summary(val, tag):
  return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])


@gin.configurable
def env_eval(
    eval_data_path,
    inf_policy,
    eval_writer,
    step,
    num_eval_tasks=50,
    trial_policy=None):
  """Evaluate the current policy in the environment."""
  if trial_policy is None:
    trial_policy = inf_policy
  demo_ret, trial_ret, inf_ret = run_meta_env.run_meta_env(
      trial_policy=trial_policy,
      inference_policy=inf_policy,
      input_data_path=eval_data_path,
      num_tasks=num_eval_tasks)
  eval_writer.add_summary(custom_summary(demo_ret, 'env/demo_ret'), step)
  eval_writer.add_summary(custom_summary(trial_ret, 'env/trial_ret'), step)
  eval_writer.add_summary(custom_summary(inf_ret, 'env/inf_ret'), step)
  return demo_ret, trial_ret, inf_ret


@gin.configurable
def train_trial_policy(input_ph):
  """Learns a stochastic trial policy that conditions on demo only."""
  condition_demo_features = input_ph['features']['condition_demo']['features']
  inference_features = input_ph['features']['inference_ep']['features']
  inference_labels = input_ph['labels']['inference_ep']['labels']
  condition_demo_ctx = condition_demo_features['obs'][:, :, -1:, :]
  condition_demo_ctx = tf.tile(condition_demo_ctx, [1, 1, 50, 1])
  fc_inputs = tf.concat([inference_features['obs'], condition_demo_ctx], -1)
  with tf.variable_scope('trial_policy', reuse=tf.AUTO_REUSE):
    fc_net_fn = functools.partial(
        models.fc_layer_net,
        output_size=2, hidden_layers=(100, 100))
    mus = models.multi_batch_apply(fc_net_fn, 3, fc_inputs)
    sigma_logits = tf.get_variable(
        'sigma_logits', shape=2, dtype=tf.float32,
        initializer=tf.initializers.constant(0.541))
  sigma = tf.math.softplus(sigma_logits)
  tf.summary.scalar('sigma/0', sigma[0])
  tf.summary.scalar('sigma/1', sigma[1])
  sigma = tf.tile(
      tf.reshape(sigma, [1, 1, 1, 2]),
      tf.concat([tf.shape(mus)[:-1], [1]], 0))
  dist = tfp.distributions.MultivariateNormalDiag(loc=mus, scale_diag=sigma)
  actions = dist.sample()
  nll = tf.reduce_mean(-1 * dist.log_prob(inference_labels['actions']))
  tf.summary.scalar('NLL', nll)
  outputs = {
      'pre_trial_inf_pred': actions,
      'inf_pred': actions,  # No adaptation.
  }
  return nll, outputs


@gin.configurable
def train_mil_baseline(input_ph, ignore_embedding=False):
  """Learns a deterministic policy that conditions on demo only."""
  condition_demo_features = input_ph['features']['condition_demo']['features']
  inference_features = input_ph['features']['inference_ep']['features']
  inference_labels = input_ph['labels']['inference_ep']['labels']
  condition_demo_ctx = condition_demo_features['obs'][:, :, -1:, :]
  condition_demo_ctx = tf.tile(condition_demo_ctx, [1, 1, 50, 1])
  if ignore_embedding:
    fc_inputs = inference_features['obs']
  else:
    fc_inputs = tf.concat([inference_features['obs'], condition_demo_ctx], -1)
  with tf.variable_scope('trial_policy', reuse=tf.AUTO_REUSE):
    fc_net_fn = functools.partial(
        models.fc_layer_net,
        output_size=2, hidden_layers=(100, 100))
    actions = models.multi_batch_apply(fc_net_fn, 3, fc_inputs)
  bc_loss = tf.losses.mean_squared_error(
      labels=inference_labels['actions'],
      predictions=actions)
  tf.summary.scalar('bc_loss', bc_loss)
  outputs = {
      'pre_trial_inf_pred': actions,
      'inf_pred': actions,
  }
  return bc_loss, outputs


@gin.configurable
def train_retrial_policy(input_ph, embedding_size=32):
  """Learns a deterministc retrial policy that conditions on demo and trial."""
  condition_demo_features = input_ph['features']['condition_demo']['features']
  condition_trial_features = input_ph['features']['condition_trial']['features']
  condition_trial_labels = input_ph['features']['condition_trial']['labels']
  condition_trial_input = tf.concat([
      condition_trial_features['obs'],
      condition_trial_labels['obs_tp1'],
      condition_trial_labels['actions']], -1)
  condition_demo_ctx = condition_demo_features['obs'][:, :, -1:, :]
  condition_demo_ctx = tf.tile(condition_demo_ctx, [1, 1, 50, 1])
  with tf.variable_scope('trial_embedding', reuse=tf.AUTO_REUSE):
    embedding_input = tf.concat(
        [condition_trial_input, condition_demo_ctx], axis=-1)
    embedding_fn = functools.partial(
        models.fc_layer_net, output_size=embedding_size,
        hidden_layers=(100, 100))
    embedding = models.multi_batch_apply(embedding_fn, 3, embedding_input)
    # Average over time dim.
    embedding = tf.tile(
        tf.reduce_mean(embedding, axis=-2, keepdims=True),
        [1, 1, 50, 1])
  inference_features = input_ph['features']['inference_ep']['features']
  inference_labels = input_ph['labels']['inference_ep']['labels']
  fc_inputs = tf.concat([
      inference_features['obs'],
      condition_demo_ctx,
      embedding], -1)
  with tf.variable_scope('retrial_policy', reuse=tf.AUTO_REUSE):
    fc_net_fn = functools.partial(
        models.fc_layer_net,
        output_size=2, hidden_layers=(100, 100))
    actions = models.multi_batch_apply(fc_net_fn, 3, fc_inputs)
  bc_loss = tf.losses.mean_squared_error(
      labels=inference_labels['actions'],
      predictions=actions)
  tf.summary.scalar('bc_loss', bc_loss)
  outputs = {
      'pre_trial_inf_pred': None,
      'inf_pred': actions,
  }
  return bc_loss, outputs


@gin.configurable
def train_and_eval(
    model_dir,
    train_data_path,
    eval_data_path,
    train_summary_period,
    eval_summary_period,
    train_steps=100000,
    batch_size=100,
    outer_lr=1e-3,
    train_fn=train_trial_policy):
  """Training loop."""
  input_ph = data_processing.get_placeholders()
  outer_loss, outputs = train_fn(input_ph)

  # Top level tf.nest lacks this function.
  flattened_feature_phs = dict(
      tf.contrib.framework.nest.flatten_with_joined_string_paths(
          input_ph['features']))

  opt = tf.train.AdamOptimizer(outer_lr)
  train_op = opt.minimize(outer_loss)
  merged_summaries = tf.summary.merge_all()
  with tf.Session() as sess:
    inf_policy = policies.LiveSessionPolicy(
        sess, flattened_feature_phs, outputs)
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(
        os.path.join(model_dir, 'train'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval'))
    sess.run(tf.global_variables_initializer())
    for i in range(train_steps):
      data = data_processing.read_task_data(
          train_data_path, batch_size=batch_size)
      feed_dict = data_processing.pack_feed_dict(input_ph, data)
      sess.run(train_op, feed_dict)
      if i % train_summary_period == 0:
        summary = sess.run(merged_summaries, feed_dict)
        train_writer.add_summary(summary, i)
        train_writer.flush()
      if i % eval_summary_period == 0:
        eval_data = data_processing.read_task_data(
            eval_data_path, batch_size=100)
        eval_feed_dict = data_processing.pack_feed_dict(input_ph, eval_data)
        summary = sess.run(merged_summaries, eval_feed_dict)
        eval_writer.add_summary(summary, i)
        _, _, _ = env_eval(eval_data_path, inf_policy, eval_writer, i)
        eval_writer.flush()
    saved_model = os.path.join(model_dir, 'saved_model')
    tf.saved_model.simple_save(
        sess, saved_model,
        inputs=flattened_feature_phs,
        outputs=outputs)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  train_and_eval(
      FLAGS.model_dir,
      FLAGS.train_data_path,
      FLAGS.eval_data_path,
      FLAGS.train_summary_period,
      FLAGS.eval_summary_period,
  )


if __name__ == '__main__':
  app.run(main)
