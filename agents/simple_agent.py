"""reference: https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

some background info:
     - scv is the workder unit for terran

the purpose for writing/learning this script is to gain a better understanding on
how to navigate the pysc2 environmental features and produce actions"""

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

## action id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id # rally = group them together
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

## get features id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

## unit ids
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21

## parameters
_PLAYER_SELF = 1 # ???
_NOT_QUEUED = [0]
_QUEUED = [1]
_SUPPY_USED = 3
_SUPPLY_MAX = 4


class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None # this store our spawn location
    scv_selected = False

    # construction related
    supply_depot_built = False
    barrack_built = False

    # attacked and control
    barracks_rallied = False
    barracks_selected = False
    army_selected = False
    army_rallied = False


    def step(self, obs):
        """a agent that just mine very quickly
        and got defeated on full map"""
        super(SimpleAgent, self).step(obs)

        # locate base
        if self.base_top_left is None:
            # return a list of coordinates of non zero units
            player_x, player_y = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

            # if the mean of y coordinates for all units is less than 31, base_top_left is assigned True
            self.base_top_left = player_y.mean() <= 31

        # build depot
        if not self.supply_depot_built:
            # select a worker if you havent
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]

                # get all x, y ccordinates for all scv units
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                # select the 1st one using actions, for no reason
                target = [unit_x[0], unit_y[0]] # target is the location for executing action in the screen
                self.scv_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # if worker selected, build depot
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                # command center has multiple x, y coordinates b/c frames
                # the 0, 20 means we build the depot above or below 20 pixel from the command center
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                self.supply_depot_built = True
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        # if build barrack avaiable, select a location and build
        elif not self.barrack_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target  = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                self.barrack_built = True
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        # control barrack
        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                # if there is a barrack, pick a location in screen
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.barracks_selected = True
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # group the units together
            else:
                self.barracks_rallied = True
                if self.base_top_left:
                    # the rally location is hard code for simplicity
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])
                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])

        elif obs.observation["player"][_SUPPY_USED] < obs.observation["player"][_SUPPLY_MAX] and \
                _TRAIN_MARINE in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_NOT_QUEUED])

        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
                elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    self.army_rallied = True
                    self.army_selected = False

                    if self.base_top_left:
                        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])


        return actions.FunctionCall(_NOOP, [])

    def transformLocation(self, x, x_distance, y, y_distance):
        """move units further from command center"""
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]