"""reference: https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

the purpose for writing/learning this script is to gain a better understanding on
how to navigate the pysc2 environmental features and produce actions"""

from pysc2.agents import base_agent
from pysc2.lib import actions

import time

## function/action id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

## get features id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

## unit ids
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45

## parameters
# what are they for?
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]


class SimpleAgent(base_agent.BaseAgent):
    def step(self, ops):
        """a agent that just mine very quickly
        and got defeated on full map"""
        super(SimpleAgent, self).step(ops)
        time.sleep(0.5)
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])