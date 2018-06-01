#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def run_loop(agents, env, max_frames, max_episodes):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec) # start tf session, build network 

    try:
        while not max_episodes or total_episodes < max_episodes:
            # iterate episode
            total_episodes += 1
            timesteps = env.reset() # start a new game
            for a in agents:
                a.reset() # reset reward discount
            while True:
                total_frames += 1
                actions = [agent.step(timestep)
                                     for agent, timestep in zip(agents, timesteps)]
                if max_frames and total_frames >= max_frames:
                    return
                if timesteps[0].last():
                    break
                timesteps = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


