---
title: "Getting Started"
layout: single
permalink: /getting-started/
excerpt: "How to quickly install and setup Minimal Mistakes for use with GitHub Pages."
last_modified_at: 2018-06-07T15:58:49-04:00
toc: true
toc_sticky: true
toc_label: "Table of Contents"
toc_icon: "cog"
---
This page is about getting started running the agent.  Please navigate to our <a href="https://github.com/aidanbean/ECS170-AI" target="_blank">GitHub</a> repository and follow these steps.
## Running the agent
Under folder `agents/`

Dueling DQN agent:
```shell
`﻿python -m main --agent=dueling_agent.dueling_agent.DuelingAgent`
```

DQN agent:
```shell
`﻿python -m main --agent=dqn_agent.dqn_agent.DQNAgent`
```

## Setting game configuration
Under file `agents/main.py`

## Setting agent configuration
Under file `agents/run_loop.py`

## Description

Both DQN agents are capable of running the full game as they are extended (with different degree of changes) with network structure from DeepMind's paper. Yet, the current reward function defined in `agents/*_agent/utils.py` is only for minigame `BuildMarines`. If you want the agent to perform better in the full game or other minig-games, please feel free to define new reward functions respectively.


## Version
The agents are bug-free under following environment configuration.

`python`: 2.7 or 3.*

`pysc2`: after 5/31/2018

`tensorflow`: latest version on gpu or cpu

`StarCraft II`: 4.1.2
