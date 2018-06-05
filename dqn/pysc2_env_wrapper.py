"""An env wrapper to print the available actions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env import base_env_wrapper
import numpy as np

class PySC2EnvWrapper(base_env_wrapper.BaseEnvWrapper):
    """An env wrapper for our DQN."""

    def __init__(self, env):
        super(PySC2EnvWrapper, self).__init__(env)
        self._action_spec = self.action_spec()[0]
        self._obsevation_spec = self.observation_spec()[0]
        # self.all_obs
        
    def reset(self, *args, **kwargs):
        features="rgb_screen"
        all_obs = super(PySC2EnvWrapper, self).reset(*args, **kwargs)
        self.all_obs = all_obs
        obs = all_obs[0].observation
        if features is "rgb_screen":
            features = obs["rgb_screen"].astype(np.float32)
        elif features is "rgb_minimap":
            features = obs["rgb_minimap"].astype(np.float32)
            
        return features

    def step(self, *args, **kwargs):
        features="rgb_screen"
        all_obs = super(PySC2EnvWrapper, self).step(*args, **kwargs)
        self.all_obs = all_obs
        obs = all_obs[0].observation
        reward = all_obs[0].reward
        done = all_obs[0].last()
        if features is "rgb_screen":
            features = obs["rgb_screen"].astype(np.float32)
        elif features is "rgb_minimap":
            features = obs["rgb_minimap"].astype(np.float32)
            
        return features, reward, done
    
    ''' Trying to test random agent; can ignore
    def random_step(self, features="rgb_screen"):
        obs = self.all_obs[0].observation
        function_id = np.random.choice(obs.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        action = actions.FunctionCall(function_id, args)
        all_obs = super(PySC2EnvWrapper, self).step(action)
        self.all_obs
        reward = all_obs[0].reward
        done = all_obs[0].last()
        if features is "rgb_screen":
            features = obs["rgb_screen"].astype(np.uint8)
        elif features is "rgb_minimap":
            features = obs["rgb_minimap"].astype(np.uint8)
            
        return features, reward, done
    '''
