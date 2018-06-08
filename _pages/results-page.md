---
title: "Results"
layout: single
permalink: /results-page/
excerpt: "How to quickly install and setup Minimal Mistakes for use with GitHub Pages."
last_modified_at: 2018-06-07T15:58:49-04:00
toc: true
toc_sticky: true
toc_label: "Table of Contents"
toc_icon: "cog"
---

## Analysis
Both DQN agents learned to build multiple depots and barracks quickly in episode 1. But they learned to build Marines relatively slower. Perhaps it’s because we select non-spatial action and spatial action simultaneously, and we can build depot or barrack in any coordinates of the map. However, build marine action requires the agent to first select the barrack then select the build marine action, which is a complex action requiring the agent much more training to learn. Since building a barracks and depots also contains positive reward, the agent tends to build these infrastructure instead of trying to click on them.
## Screenshots
![Training1](/assets/images/sc1.png)


![Training2](/assets/images/sc2.png)


![Training3](/assets/images/sc3.png)
