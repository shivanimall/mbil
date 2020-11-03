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

"""Data processing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.lib.recfunctions as rfn
import tables
import tensorflow as tf


def batch_episode(episode_data):
  """Batch transitions of a single episode."""
  zipped_episode_data = zip(*episode_data)[:5]  # Leave out the debug data.
  return tuple(np.stack(x) for x in zipped_episode_data)


def batch_episode_list(list_of_episodes):
  """Batches multiple episodes together as a recarray."""
  batched_episode_data_list = []
  for episode_data in list_of_episodes:
    batched_episode_data = batch_episode(episode_data)
    batched_episode_data_list.append(batched_episode_data)
  zipped_batched_data = zip(*batched_episode_data_list)
  batched_batched_data = tuple(np.stack(x) for x in zipped_batched_data)
  names = ('obs', 'actions', 'reward', 'obs_tp1', 'done')
  dtype = [(name, batched_batched_data[i].dtype, batched_batched_data[i].shape)
           for i, name in enumerate(names)]
  return np.array(batched_batched_data, dtype=dtype)


def pack_metarollout(
    task_spec, condition_demo_data, condition_trial_data, inference_ep_data):
  """Pack metarollout data into a numpy array."""
  dtype, data = [], []
  if condition_demo_data:
    dtype.append(('condition_demo', condition_demo_data.dtype))
    data.append(condition_demo_data)
  if condition_trial_data:
    dtype.append(('condition_trial', condition_trial_data.dtype))
    data.append(condition_trial_data)
  dtype.append(('inference_ep', inference_ep_data.dtype))
  data.append(inference_ep_data)
  recarray = np.array([tuple(data)], dtype=np.dtype(dtype)).view(np.recarray)
  return rfn.append_fields(
      recarray, 'task_spec', np.expand_dims(task_spec, 0),
      usemask=False, asrecarray=True)


class MetaRolloutWriter(object):
  """Write metarollouts using PyTables."""

  def __init__(self, path):
    """Set the pytables path."""
    self._path = path

  def write_task(self, task_data):
    """Write the data from one or more tasks."""
    with tables.open_file(self._path, 'a') as h5file:
      if '/task_data' in h5file:
        h5file.root.task_data.append(task_data)
      else:
        h5file.create_table(h5file.root, 'task_data', obj=task_data)
      h5file.root.task_data.flush()


def read_task_data(path, batch_size=None):
  """Read task (or batch of tasks)."""
  with tables.open_file(path, 'r') as h5file:
    nrows = h5file.root.task_data.nrows
    idx = np.random.choice(nrows, size=batch_size, replace=True)
    return h5file.root.task_data[idx].view(np.recarray)


def get_episode_placeholders():
  return {
      'features': {
          'obs': tf.placeholder(tf.float32, [None, 1, 50, 16]),
      },
      'labels': {
          'actions': tf.placeholder(tf.float32, [None, 1, 50, 2]),
          'obs_tp1': tf.placeholder(tf.float32, [None, 1, 50, 16]),
          'rewards': tf.placeholder(tf.float32, [None, 1, 50]),
      }
  }


def make_meta_feature_label(dict_structure):
  """Split dict into meta-feature and meta-label."""
  features = {
      'condition_demo': dict_structure['condition_demo'],
      'condition_trial': dict_structure['condition_trial'],
      'inference_ep': {'features': dict_structure['inference_ep']['features']},
      'input_scaling': dict_structure['input_scaling'],
      'target_idx': dict_structure['target_idx'],
  }
  labels = {
      'inference_ep': {'labels': dict_structure['inference_ep']['labels']},
      'input_scaling': dict_structure['input_scaling'],
      'target_idx': dict_structure['target_idx'],
  }
  return {'features': features, 'labels': labels}


def get_placeholders():
  return make_meta_feature_label({
      'condition_demo': get_episode_placeholders(),
      'condition_trial': get_episode_placeholders(),
      'inference_ep': get_episode_placeholders(),
      'input_scaling': tf.placeholder(tf.float32, [None, 2]),
      'target_idx': tf.placeholder(tf.float32, [None]),
  })


def read_episode_data(episode_data):
  return {
      'features': {
          'obs': episode_data['obs'],
      },
      'labels': {
          'actions': episode_data['actions'],
          'obs_tp1': episode_data['obs_tp1'],
          'rewards': episode_data['reward'],
      }
  }


def pack_feed_dict(ph, data):
  processed_data = make_meta_feature_label({
      'condition_demo': read_episode_data(data['condition_demo']),
      'condition_trial': read_episode_data(data['condition_trial']),
      'inference_ep': read_episode_data(data['inference_ep']),
      'input_scaling': data['task_spec']['input_scaling'],
      'target_idx': data['task_spec']['target_idx'],
  })
  return dict(zip(tf.nest.flatten(ph), tf.nest.flatten(processed_data)))
