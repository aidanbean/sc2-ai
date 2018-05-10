import gym 
import numpy as np 

def run():
	for i_episode in range(20):
		observation = env.reset()
		for t in range(100):
			env.render()

			# step returns 4 values
			# obervation(object): env specific object representing your observation of environment
			# reward(float): amount of reward achieved by previous action 
			# done(bool): whetthers its time to resest the environemtn again 
			# info(dict): diagnostic info for debugging
			action = env.action_space.sample()
			print(observation)
			observation, reward, done, info = env.step(action)

			if done:
				print("Episode finished after{} timesteps".format(t+1))
				break


def get_spaces():
	print(env.action_space) # discrete(2)
	print(env.observation_space) # box(4, )

	# box represents n-dimensional box, valid observations will be an array of 4 numbers

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	run()