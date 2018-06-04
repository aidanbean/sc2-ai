#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
introduction:
this agent use a simple deep q network of two fully connected layers,
for network configuration, please refer to dqn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

from .utils import preprocess_screen

_PLAYER_RELATIVE = features.PlayerRelative.ALLY


class DQNAgent(base_agent.BaseAgent):

    def __init__(self):
        super(DQNAgent, self).__init__()
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        pass

    def setup(self, obs_spec, action_spec):
        """tf.session setup"""
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        """reward discount reset"""
        self.episodes += 1

    def step(self, obs):
        """
        get observation, return action using RL
        obs = observation spec in lib/features.py : 218
        """
        super(DQNAgent, self).step(obs)
        # obs.observation.screen_feature is (17, 64, 64)
        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen_input = np.expand_dims(preprocess_screen(screen), axis=0)



        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])




