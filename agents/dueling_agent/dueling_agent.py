#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
introduction: this agent uses dueling deep q network
running the agent: python -m main

*note: this agent is for buildMarines only, define your own reward function
if you want an agent with different objective

reference:
    paper:
        deepmind: https://arxiv.org/abs/1708.04782
        dueling dqn: https://arxiv.org/abs/1511.06581
        a3c: https://arxiv.org/abs/1602.01783
    code:
        https://github.com/XHUJOY/pysc2-agents
        https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5.3_Dueling_DQN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from .utils import preprocess_screen, screen_channel, buildmarines_reward
rewards_file = open("rewards.txt", "w")
loss_file = open("loss.txt", "w")


class DuelingAgent(object):

    def __init__(self):
        # self.reward = 0
        self.episodes = 0
        # self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.isize = len(actions.FUNCTIONS)
        pass

    def setup(
            self,
            obs_spec,
            action_spec,
            screen_size,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            drop_out=0.2,
            apply_drop_out=False,
            e_greedy_increment=None,
            sess=None
    ):
        """
        1. this function is run before the episode iteration starts
        2. set up tf session, network structure with input args,
        experience replay storage and optimizer for training
        """

        print("start setting up Dueling DQN agent...")
        print("configuration: ")
        print("screen size: {}".format(screen_size))
        print("learning rate: {}".format(learning_rate))
        print("reward decay: {}".format(reward_decay))
        print("replace target net after {} iterations".format(replace_target_iter))
        print("memory size: {}".format(memory_size))
        print("batch size: {}".format(batch_size))
        print("drop out rate: {}".format(drop_out))
        print("apply drop out ? {}".format(apply_drop_out))

        self.obs_spec = obs_spec
        self.action_spec = action_spec

        # learning setting
        self.ssize = screen_size # input cnn size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.drop_out = drop_out
        self.apply_drop_out = apply_drop_out

        self.learn_step_counter = 0
        # self.memory = np.zeros((self.memory_size, self.ssize*2+2))
        self.memory = deque()
        self.summary = []

        # build model
        self.build_model()

        # this operation is for replace target net params with eval net params
        t_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.replace_target_op = [tf.assign(ref=t, value=e) for t, e in zip(t_params, e_params)]

        # init tf session
        self.init_session(sess=sess)

    def init_session(self, sess):
        """handle tf session setup, tf log and tf graph"""
        # set up session
        print("initialize session ...")
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        self.summary_writer = tf.summary.FileWriter("../logs/", self.sess.graph)
        print("session initialized")
        pass

    def build_network(self):
        """
        define convolution network layers (two conv, two pool, one fully-connected)
        two cnn nets with above config and
        input: screen feature

        [None, screen_channel(), self.ssize, self.ssize] -> [None, self.ssize, self.ssize, screen_channel()]
        traditional input dims:  Batch  size x  Height  x  Width  x  Channels
        tradition weight: Height  x  Width  x  Input   Channels  x  Output   Channels
        output: spatial action output, non-spatial action output


        """

        print("building network...")
        # Extract features

        sconv1 = tf.layers.conv2d(
            inputs=tf.transpose(a=self.screen, perm=[0, 2, 3, 1]),
            filters=16,
            kernel_size=[5, 5],
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            name='sconv1'   # 1st conv2d feature layer
        )

        # pooling can be inserted here

        sconv2 = tf.layers.conv2d(
            inputs=sconv1,
            filters=32,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            name='sconv2')  # 2nd conv2d feature layer

        # Compute spatial actions
        spatial_action = tf.layers.conv2d(
            inputs=sconv2,
            filters=1,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding='same',
            activation=None,
            bias_initializer=tf.constant_initializer(0.1),
            name='saptial_action')
        spatial_action = tf.nn.softmax(tf.layers.flatten(spatial_action))

        # Compute non spatial actions and value
        info_fc = tf.layers.dense(
            inputs=tf.layers.flatten(inputs=self.info),
            units=256,
            activation=tf.tanh,
            bias_initializer=tf.constant_initializer(0.1),
            name='info_fc')

        feat_fc = tf.concat([layers.flatten(sconv2), info_fc], axis=1)
        feat_fc = tf.layers.dense(
            inputs=feat_fc,
            units=256,
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            name='feat_fc')

        # define non spatial dropout
        if self.apply_drop_out:
            feat_fc = tf.layers.dropout(
                inputs=feat_fc,
                rate=self.drop_out
            )

        # non_spatial_action = tf.layers.dense(inputs=feat_fc,
        #                                      units=self.isize,
        #                                      activation=tf.nn.softmax,
        #                                      name='non_spatial_action')

        # compute q value of non-spatial action using dueling net

        with tf.variable_scope('Value_non_spatial_action'):
            print("initialize value layer for non spatial action ...")
            self.value_non_spatial_action = tf.layers.dense(inputs=feat_fc,
                                                 units=1,
                                                 activation=tf.nn.softmax,
                                                 name='Value_non_spatial_action')
        with tf.variable_scope('Advantage_non_spatial_action'):
            print("initialize advantage layer for non spatial action ...")
            self.advantage_non_spatial_action = tf.layers.dense(inputs=feat_fc,
                                                 units=self.isize,
                                                 activation=tf.nn.softmax,
                                                 name='Advantage_non_spatial_action')
        with tf.variable_scope('non_spatial_action'):
            non_spatial_action = self.value_non_spatial_action + \
                                 (self.advantage_non_spatial_action -
                                  tf.reduce_mean(self.advantage_non_spatial_action, axis=1, keep_dims=True))

        # original A3C q value calculation
        q = tf.reshape(
            tensor=tf.layers.dense(
                inputs=feat_fc,
                units=1,
                activation=None,
                name='q'),
            shape=[-1]) # a shape of [-1] flattens into 1-D

        return spatial_action, non_spatial_action, q

    

    def build_model(self):
        """
        define evaluation net, target net
        define optimizer for evaluation net
        """

        print("building model...")
        # ---------------------------evaluation net for spatial, non-spatial---------------------------
        # cnn input features
        self.screen = tf.placeholder(tf.float32, [None, screen_channel(), self.ssize, self.ssize], name='screen')
        self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')
        # build eval net for spatial, non-spatial and return q_eval scope name = eval_net, collection name = eval...
        with tf.variable_scope('eval_net'):
            # c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.spatial_action, self.non_spatial_action, self.q_eval = self.build_network()
        # self.spatial_action, self.non_spatial_action, self.q_eval = self.build_network()

        # target value
        self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
        self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2], name='spatial_action_selected')
        self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],name= 'valid_non_spatial_action')
        self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
        self.q_target = tf.placeholder(tf.float32, [None], name='q_target')

        # A3C loss calculation in deepmind paper
        # action log probability
        spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
        spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))

        non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
        valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
        valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
        non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
        non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

        self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
        self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

        # compute loss with gradient clipping
        action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
        advantage = tf.stop_gradient(self.q_target - self.q_eval)
        policy_loss = - tf.reduce_mean(action_log_prob * advantage)
        value_loss = - tf.reduce_mean(self.q_eval * advantage)

        loss = policy_loss + value_loss

        self.summary.append(tf.summary.histogram("policy_loss",policy_loss))
        self.summary.append(tf.summary.histogram("value_loss",value_loss))
        self.summary.append(tf.summary.histogram("policy_loss",loss))

        self.summary.append(tf.summary.scalar("policy_loss",policy_loss))
        self.summary.append(tf.summary.scalar("value_loss",value_loss))
        self.summary.append(tf.summary.scalar("policy_loss",loss))

        loss_file.write(str(loss))


        # Build the optimizer
        opt = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, epsilon=1e-10)
        grads = opt.compute_gradients(loss)
        cliped_grad = []
        for grad, var in grads:
            self.summary.append(tf.summary.histogram(var.op.name, var))
            self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
            grad = tf.clip_by_norm(grad, 10.0)
            cliped_grad.append([grad, var])
        self.train_op = opt.apply_gradients(cliped_grad)
        self.summary_op = tf.summary.merge(self.summary)

        # # dueling net optimizer method
        # self.q_target = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='q_target')
        # with tf.variable_scope('loss'):
        #     self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        # with tf.variable_scope('train'):
        #     self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------------------target net for spatial, non-spatial---------------------------

        with tf.variable_scope('target_net'):
            _, _, self.q_next = self.build_network()

        pass

    def reset(self):
        """reward discount reset"""

        self.episodes += 1

    def step(self, obs):
        """
        choose action
        get observation, return spatial, nonspatial action using RL
        obs = observation spec in lib/features.py : 218
        """
        # for v in tf.get_default_graph().as_graph_def().node:
        #     print(v.name)

        # obs.observation.screen_feature is (17, 64, 64)
        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen = np.expand_dims(preprocess_screen(screen), axis=0) # return (bs=1, channel=42, h=64, w=64)

        # get available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        # run session to obtain spatial action output and non spatial action array
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action], feed_dict={self.screen: screen, self.info: info})

        # select action and spatial target
        non_spatial_action = non_spatial_action.ravel() # flatten
        spatial_action = spatial_action.ravel() # flatten
        valid_actions = obs.observation['available_actions']    # available action index
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)  # position to move
        target = [int(target // self.ssize), int(target % self.ssize)]

        # e-greedy action selection (IN THIS NETWORK, WE EXPLORE ONLY IF A RANDOM FRACTION IS ABOVE EPSILON)
        if np.random.random() > self.epsilon:
            # randomly select non-spatial action
            act_id = np.random.choice(valid_actions)

            # randomly select spatial action
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):    # in fact, only screen
                act_args.append([target[1], target[0]]) # y x to x y
            else:
                act_args.append([0])  # [0] means not queue

        # self.steps += 1
        # self.reward += obs.reward

        # print("return action with id: {} and args {} ".format(act_id, act_args))
        return actions.FunctionCall(act_id, act_args)

    def store_transition(self, obs, a, obs_):
        """
        store the transition (s, a, r, s') in experience replay
        define reward function
        get s=screen, s'=screen', and action, put them in memory
        currently, a in non-spatial
        1. break observation into screen state and reward
        reward function: # marines + # minerial + # depot + # barracks
        """

        # # get s
        # screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        # s = np.expand_dims(preprocess_screen(screen), axis=0) # return (bs=1, channel=42, h=64, w=64)
        #
        # # get s_
        # screen = np.array(obs_.observation.feature_screen, dtype=np.float32)
        # s_ = np.expand_dims(preprocess_screen(screen), axis=0) # return (bs=1, channel=42, h=64, w=64)

        # get r for BuildMarines
        r = buildmarines_reward(obs_)

        # store transition
        # transition = (s, [a, r], s_)
        transition = (obs, a, r, obs_)
        self.memory.append(transition)

        # remove old memory if full
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        pass

    def learn(self):
        """when certain number of replay size reach, learn from minibatch replay"""
        print("\nstart learning...")

        # replace target net parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\nreplaced target net parameters...')

        # sample mini-batch
        sample_indices = np.random.choice(len(self.memory), size=self.batch_size)
        batch_memory = deque(list(np.array(self.memory)[sample_indices]))
        print("\nselecting minibatch of size: {}..." .format(len(batch_memory)))

        # extract s = [], a = [], s' = [], r = []
        screens = []
        screens_next = []
        infos = []
        infos_next = []
        rewards = []


        # actions
        valid_spatial_action = np.zeros([self.batch_size], dtype=np.float32)
        spatial_action_selected = np.zeros([self.batch_size, self.ssize ** 2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([self.batch_size, len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([self.batch_size, len(actions.FUNCTIONS)], dtype=np.float32)

        for i, [obs, a, r, obs_] in enumerate(batch_memory):
            # s current state from obs
            screen = np.array(obs.observation.feature_screen, dtype=np.float32)
            screen = np.expand_dims(preprocess_screen(screen), axis=0)  # return (bs=1, channel=42, h=64, w=64)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            # s_ next state from obs_
            screen_next = np.array(obs_.observation.feature_screen, dtype=np.float32)
            screen_next = np.expand_dims(preprocess_screen(screen_next), axis=0)  # return (bs=1, channel=42, h=64, w=64)
            info_next = np.zeros([1, self.isize], dtype=np.float32)
            info_next[0, obs_.observation['available_actions']] = 1

            # append to s list, s_ list
            screens.append(screen)
            infos.append(info)
            screens_next.append(screen_next)
            infos_next.append(info_next)

            # get reward r
            rewards.append(r)
            rewards_file.write(str(r))

            # get action 'a'
            act_id = a.function
            act_args = a.arguments

            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        screens = np.concatenate(screens, axis=0) # (32, size of s)
        infos = np.concatenate(infos, axis=0)
        screens_next = np.concatenate(screens_next, axis=0)
        infos_next = np.concatenate(infos_next, axis=0)

        rewards = np.transpose(np.array(rewards)) # (32, r)

        # get q_next = Q(s', a': theta) to calculate y
        q_next = self.sess.run(self.q_next, feed_dict={self.screen: screens_next, self.info: infos_next})
        # q_next = self.sess.run(self.q_eval, feed_dict={self.screen: screens_next, self.info: infos_next})
        q_target = rewards + self.gamma * q_next

        # train
        feed = {self.screen: screens,
                self.info: infos,
                self.q_target: q_target,
                self.valid_spatial_action: valid_spatial_action,
                self.spatial_action_selected: spatial_action_selected,
                self.valid_non_spatial_action: valid_non_spatial_action,
                self.non_spatial_action_selected: non_spatial_action_selected
                }

        _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
        self.summary_writer.add_summary(summary, self.learn_step_counter)
        _ = self.sess.run(self.train_op, feed_dict=feed)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        pass


if __name__ == '__main__':
    agent = DuelingAgent()
    agent.setup(
        obs_spec=1,
        action_spec=1,
        screen_size=64,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        batch_size=32,
        e_greedy_increment=None,
        sess=None
    )

    for v in tf.get_default_graph().as_graph_def().node:
        print(v.name)
    pass

    rewards_file.close()
    loss_files.close()





