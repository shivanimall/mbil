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

from absl import logging
import gin
import numpy as np
from watch_try_learn_toy import data_processing
from watch_try_learn_toy import policies
from watch_try_learn_toy import reacher_env


def run_episode(env, policy):
  """Roll out a single episode in env with policy; return the episode data."""
  obs = env.reset()
  policy.reset()
  done = False
  episode_data = []
  while not done:
    action = policy.action(obs)
    obs_tp1, rew, done, debug = env.step(action)
    episode_data.append((obs, action, rew, obs_tp1, done, debug))
    obs = obs_tp1
  return episode_data


def compute_avg_ret(batched_episode_data):
  return np.mean(np.sum(batched_episode_data.reward, axis=-1))


def run_task(
    task_idx,
    env,
    demo_policy,
    trial_policy,
    inference_policy,
    num_condition_demos,
    num_condition_trials,
    num_inference_eps,
    condition_demo_data=None,
    task_spec=None):
  """Run a single meta-rollout."""
  del task_idx
  task_spec = env.reset_task(task_spec)
  if demo_policy:
    demo_policy.reset_task()
  if trial_policy:
    trial_policy.reset_task()
  inference_policy.reset_task()

  if num_condition_demos and not condition_demo_data:
    condition_demo_data = []
    demo_policy.adapt({'task_spec': task_spec})
    for _ in range(num_condition_demos):
      episode_data = run_episode(env, demo_policy)
      condition_demo_data.append(episode_data)
    condition_demo_data = data_processing.batch_episode_list(
        condition_demo_data)

  condition_trial_data = None
  if num_condition_trials:
    condition_trial_data = []
    trial_policy.adapt({'task_spec': task_spec, 'demo': condition_demo_data})
    for _ in range(num_condition_trials):
      episode_data = run_episode(env, trial_policy)
      condition_trial_data.append(episode_data)
    condition_trial_data = data_processing.batch_episode_list(
        condition_trial_data)

  inference_ep_data = []
  inference_policy.adapt({
      'task_spec': task_spec,
      'demo': condition_demo_data,
      'trial': condition_trial_data,
  })
  for _ in range(num_inference_eps):
    episode_data = run_episode(env, inference_policy)
    inference_ep_data.append(episode_data)
  inference_ep_data = data_processing.batch_episode_list(inference_ep_data)

  return data_processing.pack_metarollout(
      task_spec, condition_demo_data, condition_trial_data, inference_ep_data)


@gin.configurable
def run_meta_env(
    demo_policy=None,
    trial_policy=None,
    inference_policy=None,
    input_data_path=None,
    output_data_path=None,
    num_tasks=10,
    num_condition_demos=1,
    num_condition_trials=0,
    num_inference_eps=1):
  """Run num_tasks meta-rollouts."""
  env = reacher_env.MetaReacherEnv()
  assert isinstance(inference_policy, policies.MetaPolicy), 'Not a MetaPolicy!'
  writer = None
  if output_data_path is not None:
    writer = data_processing.MetaRolloutWriter(output_data_path)
  demo_rets, trial_rets, inf_rets = [], [], []
  for task_idx in range(num_tasks):
    task_spec = None
    condition_demo_data = None
    if input_data_path is not None:
      task_data = data_processing.read_task_data(input_data_path)
      task_spec = task_data.task_spec
      condition_demo_data = task_data.condition_demo
    task_data = run_task(
        task_idx,
        env,
        demo_policy,
        trial_policy,
        inference_policy,
        num_condition_demos,
        num_condition_trials,
        num_inference_eps,
        condition_demo_data=condition_demo_data,
        task_spec=task_spec)

    if writer is not None:
      writer.write_task(task_data)

    if num_condition_demos:
      demo_rets.append(compute_avg_ret(task_data.condition_demo))
    if num_condition_trials:
      trial_rets.append(compute_avg_ret(task_data.condition_trial))
    inf_rets.append(compute_avg_ret(task_data.inference_ep))

  demo_ret = np.mean(demo_rets) if demo_rets else 0
  trial_ret = np.mean(trial_rets) if trial_rets else 0
  inf_ret = np.mean(inf_rets)
  logging.info(
      'Demo: %.2f\tTrial: %.2f\tInference: %.2f',
      demo_ret, trial_ret, inf_ret)
  return demo_ret, trial_ret, inf_ret
