"""
An example scripted agent specifically for solving the
BuildMarines map (mini-game).  Much of this code was influenced from
https://github.com/skjb/pysc2-tutorial/blob/master/Building%20a%20Basic%20Agent/simple_agent.py

To run this agent, do:
python -m pysc2.bin.agent --map BuildMarines --agent agents.example_agent.BuildMarines
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# Relevant actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

# Features
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Terran unit ID's
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45

_NOT_QUEUED = [0]
_QUEUED = [1]

_SUPPLY_USED = 3
_SUPPLY_MAX = 4


class BuildMarines(base_agent.BaseAgent):
    
    # basic local flag variables
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False

    def step(self, obs):
        super(BuildMarines, self).step(obs)

        # first, check if there is a supply depot.
        if not self.supply_depot_built:

            # need to select SCV first before we can build one.
            if not self.scv_selected:

                # first, get a matrix representation of screen (pixels), based on unit type
                unit_type = obs.observation["screen"][_UNIT_TYPE]

                # gets x & y coordinates of all Terran SCV unit types
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                # get the coordinates of the first SCV in the list
                target = [unit_x[0], unit_y[0]]

                # call the "select point" function, to select the unit (SCV) at that point.
                # "_NOT_QUEUED" means we are not waiting for a previous action to execute.
                self.scv_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # (next game step) if SCV selected, check if we can build supply depot.
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:

                unit_type = obs.observation["screen"][_UNIT_TYPE]

                # get all the x and y coordinates that fall in the bounds of the command center
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                # target location of supply depot will be 20 pixels below center (mean) of command center
                target = [int(unit_x.mean()), int(unit_y.mean()) + 20]

                # call "build supply depot" function, to end this game step.
                self.supply_depot_built = True
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        # once we have built supply depot, build the barracks.
        elif not self.barracks_built:

            # SCV should still be selected, so this essentially checks if we have enough minerals
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                # target location of barracks will be 30 pixels to the right of the center (mean) of command center
                target = [int(unit_x.mean()) + 30, int(unit_y.mean())]

                # build the barracks at the target location
                self.barracks_built = True
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        # select barracks before we can train marines
        elif not self.barracks_selected:

            unit_type = obs.observation["screen"][_UNIT_TYPE]

            # get the list of coordinates that fall within the bounds of the barracks
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

            # this waits until barracks has finished building.
            # (barracks will only have valid coordinates when done building)
            if unit_y.any():

                # select the center coordinates (mean) of barracks
                target = [int(unit_x.mean()), int(unit_y.mean())]
                self.barracks_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        # if we have supply left, train a marine
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in \
                obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        # otherwise, don't do any action (game goes on)
        return actions.FunctionCall(_NO_OP, [])
