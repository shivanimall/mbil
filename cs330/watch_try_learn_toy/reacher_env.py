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

"""Gym (meta)reacher env."""

import random
from dm_control import mujoco
import gin
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os
import mujoco_py

TARGET_NAMES = ['target1', 'target2']  # MetaReacher target names.


@gin.configurable
class MetaReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  """Gym reacher env, with 2 potential targets and randomized dynamics."""

  def __init__(
      self,
      frame_skip=2,
      max_episode_steps=50,
      reward_threshold=-3.75,
      min_target_separation=0.1,
      scale_input_magnitudes=False):
    """Initialize the MetaReacherEnv.

    Args:
      frame_skip: Controls self.dt, the period between timesteps. Specifically:
        dt = .01 * frame_skip.
      max_episode_steps: The number of timesteps allowed in an episode.
      reward_threshold: This argument is ignored.
      min_target_separation: The minimum distance allowed between 2 targets.
        For reference, the diameter of a target is .01.
      scale_input_magnitudes: If True, randomly scale input direction
        *and* magnitude each task. Otherwise, only change direction.
    """
    utils.EzPickle.__init__(self)
  
    model_path = 'mujoco_assets/reacher_2target.xml'
    fullpath = os.path.join(os.path.dirname(__file__), model_path)
    self.frame_skip = frame_skip
    self.model = mujoco_py.load_model_from_path(fullpath)
    self.sim = mujoco_py.MjSim(self.model)
    self.data = self.sim.data
    self.viewer = None
    self._viewers = {}

    self._ep_num_in_task = -1
    self.ep_step_num = 0
    self._min_target_separation = min_target_separation
    self._scale_input_magnitudes = scale_input_magnitudes
    

    self.max_episode_steps = max_episode_steps
    self.reward_threshold = reward_threshold

    with open(model_path, 'r') as f:
      xml_string = f.read()

    self.frame_skip = frame_skip
    #self.physics = mujoco_env.CustomPhysics.from_xml_string(xml_string)
    #self.physics = mujoco.CustomPhysics.from_xml_string(xml_string)
    self.physics = mujoco.Physics.from_xml_string(xml_string)
    self.camera = mujoco.MovableCamera(self.physics, height=480, width=640)

    self.viewer = None

    self.metadata = {
        #'render.modes': ['human', 'rgb_array'],
        'render.modes': ['rgb_array'],
        'video.frames_per_second': int(np.round(1.0 / self.dt))
    }

    self._task_spec = None
    self.reset_task()  # Need to have a task_spec before the step() call.
    self.init_qpos = self.physics.data.qpos.ravel().copy()
    self.init_qvel = self.physics.data.qvel.ravel().copy()
    observation, _, done, _ = self.step(np.zeros(self.physics.model.nu))
    assert not done
    self.obs_dim = observation.size

    bounds = self.physics.model.actuator_ctrlrange.copy()
    low = bounds[:, 0]
    high = bounds[:, 1]
    self.action_space = spaces.Box(low, high, dtype=np.float32)

    high = np.inf * np.ones(self.obs_dim)
    low = -high
    self.observation_space = spaces.Box(low, high, dtype=np.float32)

    self.seed()
    #self.camera_setup()

  @property
  def dt(self):
      return 0.01 * self.frame_skip

  def step(self, a):
    if a.shape != (2,):
      raise ValueError('Expected shape (2,), got {:s}.'.format(a.shape))
    scaled_a = self._task_spec.input_scaling * a
    target_name = TARGET_NAMES[self._task_spec.target_idx]
    vec = self.get_body_com('fingertip')-self.get_body_com(target_name)
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(a).sum()
    reward = reward_dist + reward_ctrl
    self.do_simulation(scaled_a, self.frame_skip)
    ob = self._get_obs()
    done = False
    if self.ep_step_num >= (self.max_episode_steps - 1):
      done = True
    self.ep_step_num += 1
    return ob, reward, done, dict(
        reward_dist=reward_dist,
        reward_ctrl=reward_ctrl,
        task_spec=self._task_spec)

  def reset_model(self):
    self._ep_num_in_task += 1
    self.ep_step_num = 0

    # Randomize the pos/vel of the reacher joints and the targets.
    qpos = self.np_random.uniform(
        low=-0.1, high=0.1, size=self.physics.model.nq) + self.init_qpos
    # Generate random locations for the goals that are sufficiently far
    # away from each other.
    goals = []
    while True:
      goal = self.np_random.uniform(low=-.2, high=.2, size=2)
      if np.linalg.norm(goal) < 2:
        if not goals or np.linalg.norm(
            goals[0] - goal) > self._min_target_separation:
          goals.append(goal)
      if len(goals) == 2:
        break
    # Set the 2 goal locations.
    qpos[-4:-2] = goals[0]
    qpos[-2:] = goals[1]

    qvel = self.init_qvel + self.np_random.uniform(
        low=-.005, high=.005, size=self.physics.model.nv)
    qvel[-4:] = 0  # Goals should have 0 velocity.
    self.set_state(qpos, qvel)
    return self._get_obs()

  def _get_obs(self):
    theta = self.physics.data.qpos.flat[:2]
    return np.concatenate([
        np.cos(theta),
        np.sin(theta),
        self.physics.data.qpos.flat[2:],
        self.physics.data.qvel.flat[:2],
        self.get_body_com('fingertip') - self.get_body_com('target1'),
        self.get_body_com('fingertip') - self.get_body_com('target2'),
    ])

  def reset_task(self, task_spec=None):
    self._ep_num_in_task = 0
    if task_spec is None:
      target_idx = random.choice([0, 1])
      magnitude = np.random.uniform(0.5, 2.0, size=2)
      input_scaling = np.sign(np.random.uniform(-1., 1., size=2))
      if self._scale_input_magnitudes:
        input_scaling = input_scaling * magnitude
      task_spec = np.array((target_idx, input_scaling), dtype=[
          ('target_idx', int),
          ('input_scaling', input_scaling.dtype, input_scaling.shape)
      ])
      self._task_spec = task_spec.view(np.recarray)
    else:
      self._task_spec = task_spec
    return self._task_spec

  def camera_setup(self):
    # pylint: disable=protected-access
    self.camera._render_camera.trackbodyid = -1
    self.camera._render_camera.distance = self.physics.model.stat.extent * .3
    self.camera._render_camera.lookat[0] += 0
    self.camera._render_camera.lookat[1] += 0
    self.camera._render_camera.lookat[2] += 0.5
    self.camera._render_camera.elevation = -90
    self.camera._render_camera.azimuth = 0
    # pylint: enable=protected-access
