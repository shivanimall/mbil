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

"""Policy class definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import numpy as np
import tensorflow as tf


class MetaPolicy(object):
  """Meta-learning Policy."""

  def action(self, obs):
    raise NotImplementedError('Implement this in subclass.')

  def reset(self):
    pass

  def adapt(self, condition_data):
    pass

  def reset_task(self):
    pass


@gin.configurable
class RandomPolicy(MetaPolicy):
  """Takes random actions."""

  def action(self, obs):
    del obs
    return np.random.uniform(low=-1., high=1., size=2)


@gin.configurable
class SavedModelPolicy(MetaPolicy):
  """Loads a trained tf saved_model."""

  def __init__(self, model_dir=gin.REQUIRED):
    saved_model_dir = os.path.join(model_dir, 'saved_model')
    self.sess = tf.Session(graph=tf.Graph())
    meta_graph_def = tf.saved_model.loader.load(
        self.sess,
        [tf.saved_model.tag_constants.SERVING],
        saved_model_dir)
    signature_def = meta_graph_def.signature_def
    signature_key = (
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    feature_phs, outputs = {}, {}
    for input_key in signature_def[signature_key].inputs.keys():
      name = signature_def[signature_key].inputs[input_key].name
      feature_phs[input_key] = self.sess.graph.get_tensor_by_name(name)
    for output_key in signature_def[signature_key].outputs.keys():
      name = signature_def[signature_key].outputs[output_key].name
      outputs[output_key] = self.sess.graph.get_tensor_by_name(name)
    self._feature_phs, self._outputs = feature_phs, outputs
    self._condition_data = None

  def adapt(self, condition_data):
    self._condition_data = condition_data

  def action(self, obs):
    inf_feature = np.tile(obs[None], [50, 1])[None][None]
    task_spec = self._condition_data['task_spec']
    condition_demo = self._condition_data['demo']
    feed_dict = {
        self._feature_phs[
            'condition_demo/features/obs']: condition_demo['obs'][None],
        self._feature_phs[
            'condition_demo/labels/actions']: condition_demo['actions'][None],
        self._feature_phs['inference_ep/features/obs']: inf_feature,
        self._feature_phs['input_scaling']: task_spec.input_scaling[None],
        self._feature_phs['target_idx']: task_spec.target_idx[None],
    }
    output = self._outputs['pre_trial_inf_pred']
    if 'trial' in self._condition_data and (
        self._condition_data['trial'] is not None):
      condition_trial = self._condition_data['trial']
      feed_dict.update({
          self._feature_phs[
              'condition_trial/features/obs']: condition_trial['obs'][None],
          self._feature_phs['condition_trial/labels/obs_tp1']: condition_trial[
              'obs_tp1'][None],
          self._feature_phs['condition_trial/labels/actions']: condition_trial[
              'actions'][None],
      })
      output = self._outputs['inf_pred']
    outputs = self.sess.run(output, feed_dict)
    return outputs[0, 0, 0]


@gin.configurable
class PretrainedDemoPolicy(SavedModelPolicy):
  """A policy trained for single task reacher, to generate demo data."""

  def action(self, obs):
    task_spec = self._condition_data['task_spec']
    if task_spec.target_idx == 0:
      single_task_obs = np.concatenate([
          obs[:4],
          obs[4:6],
          obs[8:10],
          obs[10:13],
      ])
    else:
      single_task_obs = np.concatenate([
          obs[:4],
          obs[6:8],
          obs[8:10],
          obs[13:16],
      ])
    feed_dict = {self._feature_phs['obs_ph']: [single_task_obs]}
    action = self.sess.run(self._outputs, feed_dict)['pol']
    return np.squeeze(action) / task_spec.input_scaling


class LiveSessionPolicy(MetaPolicy):
  """Uses a model in a live session (ie, while training)."""

  def __init__(self, sess, feature_phs, outputs):
    self.sess = sess
    self._feature_phs = feature_phs
    self._outputs = outputs
    self._condition_data = None

  def adapt(self, condition_data):
    self._condition_data = condition_data

  def action(self, obs):
    inf_feature = np.tile(obs[None], [50, 1])[None][None]
    task_spec = self._condition_data['task_spec']
    condition_demo = self._condition_data['demo']
    feed_dict = {
        self._feature_phs[
            'condition_demo/features/obs']: condition_demo['obs'][None],
        self._feature_phs[
            'condition_demo/labels/actions']: condition_demo['actions'][None],
        self._feature_phs[
            'condition_demo/labels/obs_tp1']: condition_demo['obs_tp1'][None],
        self._feature_phs[
            'condition_demo/labels/rewards']: condition_demo['reward'][None],
        self._feature_phs['inference_ep/features/obs']: inf_feature,
        self._feature_phs['input_scaling']: task_spec.input_scaling[None],
        self._feature_phs['target_idx']: task_spec.target_idx[None],
    }
    output = self._outputs['pre_trial_inf_pred']
    if 'trial' in self._condition_data and (
        self._condition_data['trial'] is not None):
      condition_trial = self._condition_data['trial']
      feed_dict.update({
          self._feature_phs[
              'condition_trial/features/obs']: condition_trial['obs'][None],
          self._feature_phs['condition_trial/labels/obs_tp1']: condition_trial[
              'obs_tp1'][None],
          self._feature_phs['condition_trial/labels/actions']: condition_trial[
              'actions'][None],
          self._feature_phs['condition_trial/labels/rewards']: condition_trial[
              'reward'][None],
      })
      output = self._outputs['inf_pred']
    outputs = self.sess.run(output, feed_dict)
    return outputs[0, 0, 0]
