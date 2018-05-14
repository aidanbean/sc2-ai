# A guide to component of pysc2 

Official pysc2 documentation can be found
[here](https://github.com/deepmind/pysc2/)

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

## Features
"Features" help expose the current state of the game.  Features include:
- Minimap features ("minimap")
- Screen features ("screen")
- General player information ("player")
- Selection ("single_select" or "multi_select")
- Avaliabe actions ("available_actions")
- etc.  More can be found in `pysc2.lib.features`

**Step** Function:

The **Step** function is called every step of the game.  It allows access to the `obs` argument which
allows access to the game Observations, which include features.  For example, if you wanted to select a particular unit,
you would have to look at the `unit_type` layer within the `screen` feature layer.  To do this:

```python
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
unit_type = obs.observation["screen"][_UNIT_TYPE]
``` 

`unit_type` will contain a matrix with the same dimensions as the game resolution.  Each entry in the matrix will be a 
number representing the unit type (there are ~500 unit types). Then, to filter by particular units, e.g. a Terran SCV 
(which is unit #45), we can do this:

```python
unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
```

`unit_x` and `unit_y` will then contain all the coordinates of the screen where SCVs are located.
