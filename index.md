---
title: "StarCraft II AI"
layout: single
permalink: /
date: 2018-06-07T10:46:41-04:00
header:
  overlay_color: "#000"
  overlay_filter: "0.3"
  overlay_image: /assets/images/wall006-1920x1200.jpg
  caption: "Photo credit: [**Blizzard**](http://us.battle.net/sc2/en/media/wallpapers/?view=wall006)"
excerpt: "Exploring Sparse Rewards with Deep Reinforcement Learning<br /><br /><br />"
classes: wide
---
## Introduction

  StarCraft II has several significant obstacles for developing a game-playing agent that makes it a difficult challenge in artificial intelligence.   Firstly,  the action space is large which results in a combinatorial explosion of possible game states. Secondly, the state space is high-dimensional given there are hundreds of elements in each game frame. Thirdly, game actions might result in long-term consequences that cannot be well captured even by state-of-art learning methods.  Finally, the goal of the game, which might appear obvious to human players, is obscure for many learning methods to achieve.

  We decided to implement simple agents first then gradually increase the complexity level. In the beginning, we implemented scripted agents and a Q  table  agent  that  are  only  capable  of  playing the minigame [BuildMarines](https://github.com/deepmind/pysc2/blob/master/docs/mini_games.md). At the later stage, we implement the DuelingDQN agent that is capable of playing the full game with no restriction on state space or action space.

We choose Dueling DQN is because Dueling DQN performs  generally  better  than  baseline  DQN  in  Atari  games. It required less computation power compare to parallel A3C. Further, in RTS games, we often need to consider the advantage  of chosen actions, namely how much better an action is compared to another.  Dueling DQN encoded this mechanism. And the BuildMarines minigame,  which didn’t offer much state changes but valued much on choosing the correct action is the perfect playground to test Dueling DQN.
