#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time



def run_loop(agents, env, max_frames, max_episodes, screen_size, save_replay):
    """A run loop to have agents and an environment interact."""

    # record frame, episode and time
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    # obtain obs spec, action spec, setup agent
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(
            obs_spec=obs_spec,
            action_spec=act_spec,
            screen_size=screen_size,
            learning_rate=0.001,
            reward_decay=0.9,
            max_epilson=0.9,
            init_epilson=0.5,
            replace_target_iter=20,
            memory_size=1000,
            batch_size=32,
            drop_out=0.1,
            apply_drop_out=True,
            e_greedy_increment=0.01,
            sess=None
        )    # start tf session, build network, etc

    # learning loop
    try:
        # for each episode
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            timesteps = env.reset() # start a new episode with the env

            # reset for every agent - there is only one agent here
            for a in agents:
                a.reset()   # what need to be reset ? e_greedy ? reward ?

            # while the game is not terminated
            while True:
                total_frames += 1

                # record s
                obs = timesteps

                # use step (choose action) function to return an action based on observation in time step
                actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]

                # return the run loop function if max frame reached, default is None
                if max_frames and total_frames >= max_frames:
                    # save score
                    for a in agents:
                        a.save_endgame_score()

                    if save_replay:
                        env.save_replay(agents[0].__name__)
                    return
                # terminate if terminated state reach
                if timesteps[0].last():
                    for a in agents:
                        a.get_endgame_score(obs=timesteps[0])
                    break

                # progress the environment using actions returned from agent
                timesteps = env.step(actions)
                # store experience replay
                for a in agents:
                    a.store_transition(obs=obs[0], a=actions[0], obs_=timesteps[0])

                # agent learn from replay after every # of steps / frame
                if (total_frames > 600) and (total_frames % 200 == 0):
                    for a in agents:
                        a.learn()

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


