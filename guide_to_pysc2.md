# A guide to components of pysc2 

## Introduction:
This `md` provided an un-official guidance for using pysc2
to develop agents. It included introduction to some source code 
document in the pysc2, as well as notes on how to navigate them.

Official pysc2 documentation can be found
[here](https://github.com/deepmind/pysc2/)

## remark:
If you would like to contribute to the document, try look at if there existing section you can add to, if not create a section `## section`;
to create a sub-section, `### subsection` under the section you would like to add.

The outline for this `.md` is:
```md
## section/directory of file that will be explained
    ### sub-section/ files
    ===========
    description
    ===========

```

## running a custom agent:
Go the directory where you agent source code is stored, and run the following command:

`$ python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards`

## Directories:
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
 - `observation_spec`: Defines the observations provided by the environment. Redefined in `lib.features` specific for SC2.
 - `action_spec`: Defines the actions that should be provided to `step`

 ### base_env_wrapper.py:
 `BaseEnvWapper` is a child of `Base` class from `environment.py`, which is used for overwrite the `Base`

 ### sc2_env.py (the most important file in env):
`SC2Env`: As a derived class from `environment.Base`, it re-defines all function members from the `Base` to create the SC2 environment parameters.  

Some environment parameters: `discount (default=1), score_index (-1: choose win/loss as reward, >=0: choose some scores from one of the score_cumulative)`.

Full parameter list can be found [here](https://github.com/deepmind/pysc2/blob/3e0749630aebbcc2f9a62613fcdf149095d4d6d0/pysc2/env/sc2_env.py#L88)

`observation_spec`: import from `lib.feature`, contains 12 features. Here are some important feature:

- ``screen`: `screen.SCREEN_FEATURES` contain 13 sub-feature layers. found [here](https://github.com/deepmind/pysc2/blob/3e0749630aebbcc2f9a62613fcdf149095d4d6d0/pysc2/lib/features.py#L140)

- `observation_spec` is in line 140 in features.py

A summary of action/observation feature can be found [here](https://github.com/deepmind/pysc2/blob/master/docs/environment.md)

## Actions:

`actions.py` lives in `lib/`. Around 500 action, mostly binary choice, controlled 
by agrument `queued[] = [True, False]`.

### run_loop.py:
a run loop for agent/environment interaction, used to `bin/agent.py` for running agent. 

function `run_loop` gets an agent, create an environment, pass in initial 
timestep to the agent, run function `agent.step()` with a `timestep` as arugment
, which consisit of 4 parts:
- `timestep.step_type`

- `timestep.reward`

- `timestep.discount`

- `timestep.observation`: a dictionary, which contains `observation_spec`

## Agent template :
```python 
class SimpleAgent(base_agent.BaseAgent):
    """simple agent"""
    def step(self, timesteps):
        super(SimpleAgent, self).step(timesteps)
        #############
        generate action based on timesteps.observation, timesteps.reward
        #############
        return actions.FunctionCall(function_id, args)


class PolicyGradientAgent(base_agent.BaseAgent):
    """policy gradient that iteratively update an initial policy to 
    maximize utility function"""
    def setup(self, obs_spec, action_spec): # overwrite function in BaseAgent

    def create_policy_net(self): # 

    def create_training_method(self): # 

    def train_policy_net(self): # 

    def step(self, obs): # overwrite function in Base, choose action based on pi net
```

### A few notes:
An action in pysc2 is a composition of smaller parameters, these parameters are 
sort of like smaller actions.  

## Creating a agent:

### Tips:
- You have as much time as you want in one timestep, the game will only preceded to next 
time frame after you output your action with the `step` function.

- In the game build marine, you will need to first build supply depot, then build the 
barrack to BUILD MARINES.

- In `def step(self, ops)`, `ops` contains the state information, for more information, refer to the FEATURE section 
down below.

- To get an action id: `actions.FUNCTIONS.action1.id`. To get an action by id: `action.functionCall(actions.FUNCTIONS.action1.id)`

- `SCREEN_FEATURES` are the 13 minimap features, to get a feature index: 
`features.*_FEATURES.feature1.index`. To get a feature: `feature1 = obs.observation[fea_location][feature1 index]`.

- `QUEUED` is a bool argument for most of the actions to indicate whether 
this action should be executed now or after pervious action

- The screen is `84x84`, with the top left being `[0, 0]`. spawn location is either 
top left or bottom right.

- `PLAYER_RELATIVE`: a list of units arranged relative to the 
current player.

- Coordinate information are extracted from 6 frame in total. If there is 18 units in the game, you will get 
`18 x 6` coordinates for the same 18 units that is moving in the 6 frames. The coordinates of units are 
returned in the order (y, x), but you must pass in the values in the order (x, y).

- To execute a action, you must call `actions.FunctionCall("function", "arguments for this function")` in `def step()`

- variables store in agent class with be saved to the next eposide. 

## Python:
`__future__`: avoid confusion on different import tools, avoid incompatiablility in different version of python

## Features:
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
