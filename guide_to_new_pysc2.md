## a guide to the newest version(1.2) of pysc2

### avaiable action:
obs.observation.available_action

### timestep class:
`['step_type', 'reward', 'discount', 'observation']`

      A `TimeStep` namedtuple containing:
        step_type: A `StepType` of `FIRST`.
        reward: `None`, indicating the reward is undefined.
        discount: `None`, indicating the discount is undefined.
        observation: A NumPy array, or a dict, list or tuple of arrays
          corresponding to `observation_spec()`.
          
### def parse_agent_interface_format():
this function wrap the ground level layers to create the environment in bin/agent.py for the agent to play 

### obs.observation.screen_feature:
it's of size (17 = # features, 64, 64 = screen size)