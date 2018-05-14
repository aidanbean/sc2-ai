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


## Python:
`__future__`: avoid confusion on different import tools, avoid incompatiablility in different version of python

