import numpy as np
import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)


class DuelingDQN:
    def __init__(
            self,
            output_dims, # the input dimension in [] form
            input_dims,  # the output dimension in [] form
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            sess=None,
    ):
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, input_dims*2+2))
        self._build_model()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_network(self, s, c_names, n_l1, w_init, b_init):
        """build network layers for the model"""
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.input_dims, n_l1], initializer=w_init, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

        with tf.variable_scope('Value'):
            w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_init, collections=c_names)
            b2 = tf.get_variable('b2', [1, 1], initializer=b_init, collections=c_names)
            self.V = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Advantage'):
            w2 = tf.get_variable('w2', [n_l1, self.output_dims], initializer=w_init, collections=c_names)
            b2 = tf.get_variable('b2', [1, self.output_dims], initializer=b_init, collections=c_names)
            self.A = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Q'):
            out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)

        return out

    def _build_model(self):
        """build target net and eval net"""
        
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.input_dims], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.output_dims], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_init, b_init = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = self._build_network(self.s, c_names, n_l1, w_init, b_init)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.input_dims], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = self._build_network(self.s_, c_names, n_l1, w_init, b_init)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.output_dims)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample a minibatch
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.input_dims:]}) # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.input_dims]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.input_dims].astype(int)
        reward = batch_memory[:, self.input_dims + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.input_dims],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

