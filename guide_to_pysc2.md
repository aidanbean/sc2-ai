# A guide to component of pysc2 

## running a custom agent:
`$ python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards`

## component of the the pysc2 API:
### more important:
`agent`: stores the agent class

`bin`: the core of running the agent

`env`: environment protocol

`lib`: define feature layers

`maps`: setting on Maps

### less important
`run_configs`: setting on running the game

`tests`: unit tests

## Bin:
define two game mode: human play and agent play

human play: bin/play.py

agent play: bin/agent.py

### agent.py:
set map (type, size), difficulty, race, etc for running an agent-played game. Important settings are: `max_agent_steps, step_mul (dafult: APM=8), agent (default: random_agent), diffculty (default: None=VeryEasy), map(map name)`

### Other files:
`map_list` : list all maps

`gen_actions`: Generate the action definitions for actions.py

`actions`: Print the valid actions

`replay_actions`: Dump out stats about all the actions that are in use in a set of replays

## Env:

### environment.py:
Define a base for environment parameters.

`timestep`: precept from environment from a step of interaction, a triple of (step_type, observation, reward).
    Attribute contains reward, discount, step type.

`steptype`: first, mid, last(end of sequence)

`Base`: abstract class of RL environment
 - `reset`: Starts a new sequence and returns the first `TimeStep` of this sequence
 - `step`: Updates the environment according to the action and returns a `TimeStep`
 - `observation_spec`: Defines the observations provided by the environment
 - `action_spec`: Defines the actions that should be provided to `step`


## Python:
`__future__`: avoid confusion on different import tools, avoid incompatiablility in different version of python

