#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions

class BaseAgent(object):
    """A base agent to write custom scripted agents.
    It can also act as a passive agent that does nothing but no-ops.
    """

    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

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
        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
