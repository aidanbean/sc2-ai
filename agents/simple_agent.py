"""https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c"""

from pysc2.agents import base_agent
from pysc2.lib import actions

class SimpleAgent(base_agent.BaseAgent):
    def step(self, ops):
        super(SimpleAgent, self).step(ops)
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])