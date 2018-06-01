""" intro: a two layer nerual network for q-value perdiction and value iteration,
    approximator: adam algorithm
    the Q value approximator of this QDN is Q(s) instead of Q(s, a)
    in other words, its calculation is action independent,
    the approximator will produce q value for all actions,
    the agent will often perfer the one with best q value"""

import tensorflow as tf
import numpy as np
import random
from collections import deque

INITIAL_EPSILON = 0.3 # 0.3 chance to explore at the beginning
FINAL_EPSILON = 0.01
HIDDEN_SIZE = 50
REPLAY_SIZE = 10000 # size of the experience replay
BATCH_SIZE = 32
GAMMA = 0.7

class DQN():
    def __init__(self, env):
        """besides the following, other parameters contains:
            self.state_input, self.action_input, self.q_value_layer"""
        # experinece replay storage
        self.replay_buffer = deque()

        # parameter
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_net()
        self.create_training_method() # create optimizer

        # init session
        self.sess = tf.InteractiveSession() # basically the same as session()
        self.sess.run(tf.global_variables_initializer())
        pass

    def weight_var(self, shape):
        """create weight for a layer"""
        init = tf.truncated_normal(shape=shape) # std_dev default is 1
        return tf.Variable(init)

    def bias_var(self, shape):
        """create a bias for neuron
        Constant 2-D tensor populated with scalar value -1.
            tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                        [-1. -1. -1.]]"""
        init = tf.constant(value=0.01, shape=shape)
        return tf.Variable(init)

    def create_Q_net(self):
        """construct weight and bias, define input layer and hidden layer, and output layer(q value) """

        # weights, hidden layer size 20， every neuron has 4 weight, total dimension [4 x 20]
        w1 = self.weight_var([self.state_dim, HIDDEN_SIZE])
        # add bias to all neurons
        b1 = self.bias_var([HIDDEN_SIZE])
        w2 = self.weight_var([HIDDEN_SIZE, self.action_dim])
        b2 = self.bias_var([self.action_dim])

        # input layers
        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        h_layer = tf.nn.relu(features=tf.matmul(self.state_input, w1) + b1)
        self.q_value = tf.matmul(h_layer, w2) + b2 # this is consider an operation, must eval in tf
        pass

    def create_training_method(self):
        # one hot representation, need action input to get the predict q value
        self.action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim]) # [0, 1]
        # target q value from experience replay for this action
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None])
        # current q value for this action
        q_action = tf.reduce_sum(tf.matmul(self.q_value, tf.transpose(self.action_input)), reduction_indices=1)

        self.cost = tf.reduce_mean(input_tensor=tf.square(self.y_input - q_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=self.cost)
        pass

    def preceive(self, state, action, reward, next_state, done):
        """store preceived transition"""
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_net()
        pass

    def train_Q_net(self):
        self.time_step += 1

        # get minibatch from replay D
        minibatch = random.sample(population=self.replay_buffer, k=BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        q_value_batch = self.q_value.eval(feed_dict={self.state_input: next_state_batch})  # max_a' Q(s', a')

        # calculate y
        y_batch = [] # sample a batch of target y value, and use GD to optimize loss between y and q
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                # for terminal
                y_batch.append(reward_batch[i])
            else:
                # for non-terminal
                y_batch.append(reward_batch[i] + GAMMA * np.max(a=q_value_batch[i]))

        # self.optimizer.run(feed_dict={self.y_input: y_batch, self.state_input: state_batch, self.action_input: action_batch}) # current q value has been obtained
        self.optimizer.run(feed_dict={self.y_input: y_batch, self.state_input: state_batch, self.action_input: action_batch})
        pass

    def egreedy_action(self, state):
        """allow epsilon chance to explore in training"""
        q_value = self.q_value.eval(feed_dict={self.state_input: [state]})[0]

        if random.random() <= self.epsilon:
            action_id = random.randint(0, self.action_dim - 1)
        else:
            action_id = np.argmax(q_value) # Returns the indices of the maximum values along an axis = action id

        # exploration gets smaller as trainning progress
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return action_id
        pass

    def action(self, state):
        """directly return 'best' action in testing """
        return np.argmax(self.q_value.eval(feed_dict={self.state_input: [state]})[0])
        pass
