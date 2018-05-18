import gym 
import numpy as np

# establish gym environment 
def main():
	"""
	using frozen lake gym environment to implement simple q table learning algo 
	most important 2 things: 
	1. How the action step is randomly chosen from current states action space 
	2. bellman equation to figure out the Q table score for this state 
	"""
	env = gym.make('FrozenLake-v0')

	# init variables 
	# Q table is all zeros, with size of (observation, action) space size 
	Q = np.zeros((env.observation_space.n, env.action_space.n))

	# learning parameters 
	lr = 0.8
	y = 0.95
	num_episodes = 2000

	# list to contain total rewards and steps per episode
	rewardList = []

	# run basic Q learning algorithm 
	for i in range(num_episodes):
	    state = env.reset()
	    rewardAll = 0
	    done = False 
	    j = 0
	    
	    # Q table learning algorithm 
	    while j < 99:
	        j += 1
	        
	        env.render()
	        # randomly choose available action from current state's actions
	        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1/(i+1)))
	        
	        # take step in environment
	        state1, reward, done, _ = env.step(action) 
	        
	        # bellman equation 
	        Q[state, action] = Q[state, action] + lr*(reward + y * np.max(Q[state,:]) - Q[state, action])
	        
	        # update reward and state
	        rewardAll += reward 
	        state = state1
	        if done is True:
	            break 
	            
	    rewardList.append(rewardAll)

	print("Score over time: " + str(sum(rewardList)/num_episodes))
	print("Final Q Table: ")
	print(Q)


if __name__ == '__main__':
	main()