## DQN agent

this agent use the same network structure as defined in the deepmind's 
paper. 

### configuration: 
- for configuration of layers, please refer to https://arxiv.org/abs/1708.04782
- the only difference is: 1. we use gradient clipping to gradient explosion. 2.
we seperately get Q_target = y for target network, 
 and Q_evaluation from evaluation network. And update parameters of 
 target network from the evaluation network after certain learning 
 iteration.  
 
 ### running agent:
 - this agent's reward function is explicitly for buildmarines. reward 
 calculator method is in the utils.py in dqn_agent folder. 
 Although our agent is capable of playing any SC full game or minigame,
 a general reward function is not defined here due to limited time. 
 but, it gives perfect opportunties for us in the future 
 to define a reward function for other agent task. and that makes perfect sense ! 
 As a person, we dont aim for the same object when doing different jobs!
 - the default map is BuildMarines. render is set to true. 
 - change the game configuration in main.py in agents/ folder. change agent 
 configuration in agents/run_loop.py. 