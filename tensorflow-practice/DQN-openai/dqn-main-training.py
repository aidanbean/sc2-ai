import gym
from dqn import DQN

# hyper parameter
ENV_NAME = "CartPole-v0"
EPISODE = 10000 # from start to end
STEP = 300 # step limitation in a episode
TEST = 10 # test the agent performance for 10 games every 100 episodes


if __name__ == '__main__':
    # init env and agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # at the beginning of each episide, re-init the environment
        state= env.reset()

        for step in range(STEP):

            # start training / playing in this episode / game until this game over
            action = agent.egreedy_action(state)
            # perhaps env.step return an extra arg, but we ignore it with _
            next_state, reward, done, _ = env.step(action)
            reward_agent = -1 if done else 0.1
            agent.preceive(state, action, reward, next_state, done)
            state = next_state

            if done:
                # when the episode is complete
                break

        # test for every 100 episode
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    # the only difference than training. we directly return action produced by DQN
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action=action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print("episode: {} avg reward: {}".format(episode, ave_reward))
            if ave_reward >= 200:
                break



