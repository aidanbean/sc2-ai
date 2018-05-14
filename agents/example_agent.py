"""
An example scripted agent specifically for solving the
BuildMarines map (mini-game).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# Relevant Terran actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

# Terran unit ID's
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45
_TERRAN_MARINE = 48

class BuildMarines(base_agent.BaseAgent):

  def step(self, obs):
    super(BuildMarines, self).step(obs)
    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
