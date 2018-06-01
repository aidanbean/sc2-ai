#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
introduction:
this agent use a simple deep q network of two fully connected layers,
for network configuration, please refer to basic_dqn.py
"""

from pysc2.agents import base_agent


class BasicDQNAgent(base_agent.BaseAgent):

    def __init__(self):
        super(BasicDQNAgent, self).__init__()
        pass

    def step(self, obs):
        super(BasicDQNAgent, self).step(obs)
        pass




