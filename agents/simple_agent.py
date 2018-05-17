"""reference: https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

some background info:
     - scv is the workder unit for terran

the purpose for writing/learning this script is to gain a better understanding on
how to navigate the pysc2 environmental features and produce actions"""

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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
_PLAYER_SELF = 1 # ???
_NOT_QUEUED = [0]
_QUEUED = [1]


class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None # this store our spawn location
    supply_depot_built = False
    scv_selected = False

    def step(self, obs):
        """a agent that just mine very quickly
        and got defeated on full map"""
        super(SimpleAgent, self).step(obs)
        time.sleep(0.5)

        if self.base_top_left is None:
            # return a list of coordinates of non zero units
            player_x, player_y = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

            # if the mean of y coordinates for all units is less than 31, base_top_left is assigned True
            self.base_top_left = player_y.mean() <= 31


        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]

                # get all x, y ccordinates for all scv units
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                # select the 1st one using actions, for no reason
                target = [unit_x[0], unit_y[0]]
                self.scv_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])


        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def transformLocation(self, x, x_distance, y, y_distance):
        """move units further from command center"""
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]