#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
introduction: this agent use a dueling deep q network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from utils import preprocess_screen, screen_channel

_PLAYER_RELATIVE = features.PlayerRelative.ALLY


class DuelingAgent(object):

    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
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
            e_greedy_increment=None,
            output_graph=False,
            sess=None
    ):
        """
        1. this function is run before the episode iteration starts
        2. set up tf session, network structure with input args,
        experience replay storage and optimizer for training
        """
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

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.ssize*2+2))

        # build model
        self.build_model()

        # this operation is for replace target net params with eval net params
        # t_params = tf.get_collection('target_net_params')
        # e_params = tf.get_collection('eval_net_params')
        t_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.replace_target_op = [tf.assign(ref=t, value=e) for t, e in zip(t_params, e_params)]

        # init tf session
        self.init_session(sess=sess, output_graph=output_graph)

    def init_session(self, sess, output_graph):
        """handle tf session setup, tf log and tf graph"""
        # set up session
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
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
        non_spatial_action = tf.layers.dense(inputs=feat_fc,
                                             units=len(actions.FUNCTIONS),
                                             activation=tf.nn.softmax,
                                             name='non_spatial_action')

        # compute q value using dueling net
        with tf.variable_scope('Value'):
            self.V = tf.layers.dense(
                inputs=feat_fc,
                units=1,
                activation=None,
                bias_initializer=tf.constant_initializer(0.1),
                name='V')
        with tf.variable_scope('Advantage'):
            self.A = tf.layers.dense(
                inputs=feat_fc,
                units=len(actions.FUNCTIONS),
                activation=None,
                bias_initializer=tf.constant_initializer(0.1),
                name='A')
        with tf.variable_scope('Q'):
            q = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
        # q = tf.reshape(
        #     tensor=tf.layers.dense(
        #         inputs=feat_fc,
        #         units=1,
        #         activation=None,
        #         name='q'),
        #     shape=[-1]) # a shape of [-1] flattens into 1-D

        return spatial_action, non_spatial_action, q

    def build_model(self):
        """
        define evaluation net, target net
        define optimizer for evaluation net
        """

        # ---------------------------evaluation net for spatial, non-spatial---------------------------
        # cnn input features
        self.screen = tf.placeholder(tf.float32, [None, screen_channel(), self.ssize, self.ssize], name='screen')
        self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')
        # build eval net for spatial, non-spatial and return q_eval scope name = eval_net, collection name = eval...
        with tf.variable_scope('eval_net'):
            # c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.spatial_action, self.non_spatial_action, self.q_eval = self.build_network()

        # target value
        # self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
        # self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2], name='spatial_action_selected')
        # self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],name= 'valid_non_spatial_action')
        # self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
        self.q_target = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='q_target')

        # # action log probability
        # spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
        # spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
        #
        # non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
        # valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
        # valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
        # non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
        # non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
        #
        # # compute loss
        # action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
        # advantage = tf.stop_gradient(self.q_target - self.q_eval)
        # policy_loss = - tf.reduce_mean(action_log_prob * advantage)
        # value_loss = - tf.reduce_mean(self.q_eval * advantage)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------------------target net for spatial, non-spatial---------------------------

        with tf.variable_scope('target_net'):
            _, _, self.q_next = self.build_network()

        pass

    def reset(self):
        """reward discount reset"""

        self.episodes += 1

    def step(self, obs):
        """
        get observation, return action using RL
        obs = observation spec in lib/features.py : 218
        """

        # obs.observation.screen_feature is (17, 64, 64)
        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen_input = np.expand_dims(preprocess_screen(screen), axis=0) # return (bs=1, channel=42, h=64, w=64)

        # get available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        # run session to obtain spatial action output and non spatial action array
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict={self.screen: screen_input, self.info: info})

        # select action and spatial target
        non_spatial_action = non_spatial_action.ravel() # flatten
        spatial_action = spatial_action.ravel() # flatten
        valid_actions = obs.observation['available_actions']
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        target = [int(target // self.ssize), int(target % self.ssize)]

        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def store_transition(self, obs, a, obs_):
        """store the transition in experience replay"""

        pass

    def learn(self):
        """when certain number of replay size reach, learn from minibatch replay"""
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
        output_graph=False,
        sess=None
    )





