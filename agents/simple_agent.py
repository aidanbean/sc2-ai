"""https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c"""

from pysc2.agents import base_agent
from pysc2.lib import actions

import time

class SimpleAgent(base_agent.BaseAgent):
    def step(self, ops):
        """a agent that just mine very quickly
        and got defeated on full map"""
        super(SimpleAgent, self).step(ops)
        time.sleep(0.5)
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])