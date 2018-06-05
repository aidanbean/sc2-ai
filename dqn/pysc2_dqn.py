#import gym
#from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import plotting

from collections import deque, namedtuple

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2_env_wrapper import PySC2EnvWrapper

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ADD = [0]
_NOT_QUEUED = [0]

step_mul = 8

"""
Tasks:

TODO: Create a mapping between action id and the composed action
TODO: Select all marines if they are not selected; Note: DefeatRoaches, DefeatZerglingsAndBanelings
      can have mid-episode state change
TODO: Make sure PySC2EnvWrapper returns the exact same results to be passed as OpenAI gym env
TODO: Train the network 
TODO: Demonstrate results
TODO: Write Reports and other Deliverables

1. State processing 
process the state, which is just a picture in Atari. 
For us the input may be a vector of actions, stats, etc
Needs to be done to figure out how to interface with the Q Estimator Network

2. Building the Q network
Atari used image processing, 3 conv layers, 1 fully connected 
We can use a simple feed forward, or something crazy haha. Capsule net? 

3. Picking a Action Policy
The implementation provided is a epsilon greedy policy 
https://stats.stackexchange.com/questions/248131/epsilon-greedy-policy-improvement?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
provides a probablility for possible actions
We could pick our own or a different policy, i think A3C policy is a little diffeerent 

4. Integration into Q learning model 
Put everything together.
Load replay memory, Big for loop for running through number of episodes 

"""



"""
Actions: Move_screen (MS), Attack_screen (AS)
We use a mapping between each Q network prediction and a fully composed action with complete arguments
We discretize the screen into a 12 x 12 grid where each grid cell is 7 x 7. Each action maps 
to either Move_screen or Attack_screen and the center of one of these grid cells. Note: the action 
is never queued.

0 = 331/Move_screen (3/queued [0]; 0/screen [0, 3])
1 = 331/Move_screen (3/queued [0]; 0/screen [0, 10])
...
23 = 331/Move_screen (3/queued [0]; 0/screen [73, 73])
24 = 12/Attack_screen (3/queued [0]; 0/screen [0, 3])
25 = 12/Attack_screen (3/queued [0]; 0/screen [0, 10])
...
47 = 12/Attack_screen (3/queued [0]; 0/screen [73, 73])
"""

VALID_ACTIONS = [i for i in range(48)]

''' Don't need StateProcessor; do any processing in PySC2EnvWrapper
class StateProcessor():
    """
    process a raw starcraft 2 minimap image.
    """

    def __init__(self):
        # build the tf graph
        with tf.variable_scope("state_processor"):
            # input will be the 64 x 64 x 3 minimap rgb values
            self.input_state = tf.placeholder(shape=[64, 64, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            # after this, we don't need to crop because we are using whole minimap.
            # still resize to 84 x 84 for compatibility purposes:
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Arguments:
            sess: a tf session object
            state: a [64, 64, 3] starcraft2 minimap rgb state
        Returns:
            processed [84, 84, 1] grayscale state
        """
        return sess.run(self.output, feed_dict={self.input_state: state})
'''

class Estimator():
    """Q-Value Estimator Network"""

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        build tf graph
        3 convolutional layers 2 fully connected layer
        We can manipulate the network outself.
        MSE loss
        RMSPropOptimizer from the DQN paper
        We can pick a new one ourself.
        Set to minimize
        """

        # placeholders
        # input is rgb screen or minimap image
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.uint8, name="X")
        # target TD values
        self.Y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer ID of action selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # 3 convolutional layers
        # (input, num_output, kernel_size, stride, activation)
        conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # flatten into a fully connected layer
        flattened = tf.contrib.layers.flatten(conv3)
        # fc1
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        # predict a action - fc2
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # get predictions for chosen action
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # calculate loss
        # mean of all the differences
        self.losses = tf.squared_difference(self.Y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # optimizer parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # summaries for tensorboard
        # visual learning
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("lost_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            # max q value is the max value from predictions
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        predict action value

        Args:
            sess: tf session
            s: State inpuf of shape [batch_size, 84, 84, 3]

        Returns:
            Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing estimated action value
        """
        return sess.run(self.predictions, feed_dict={self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        update the estimator toward given targets
        Args:
            tf session
            s: state input of shape [batch_size, 84, 84, 3]
            a: chosen action of shape [batch_size]
            y: targets of shape [batch_size]

        Returns:
            calculated loss on the batch
        """
        feed_dict = {self.X_pl: s, self.Y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss


# 2 networks that share same parameters in DQN algorithm
# copy the paramters to target network on each, t, steps
def copy_model_parameters(sess, estimator1, estimator2):
    """
    copies model parameters of one estimator to another
    Args:
        tf session:
        estimator1: estimator to copy from
        estimator2: estimator to copy to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    create epsilon-greedy policy based on Q function and epsilon
    https://stats.stackexchange.com/questions/248131/epsilon-greedy-policy-improvement?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    Args:
        estimator: returns q values for given state
        nA: number of actions in the environment

    Returns:
        A function that takes the sess, observation, epsilon as function
        returns probabilities for each action in from of a numpy array length of nA
    """

    # we can define our own policy here too. f
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

#TODO: Complete function
def compose_action_from_id():
    ''' For reference; return actions.FunctionCall(function_id, args)
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        function_id = numpy.random.choice(obs.observation.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
                return actions.FunctionCall(function_id, args)
    ''' 
    mapping = {}
    for k in range(24):
        for i in range(3,84,7):
            for j in range(3,84,7):
                function_id1 = _MOVE_SCREEN
                function_id2 = _ATTACK_SCREEN
                args1 = [_NOT_QUEUED,[i, j]]
                args2 = [_NOT_QUEUED,[i, j]]
                mapping[k] = actions.FunctionCall(function_id1, args1)
                mapping[k+24] = actions.FunctionCall(function_id2, args2)
    return mapping

# DQN algorithm
def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    DQN algorithm with fff-policy Temporal Differnce control
    returns EpisodeStats object with 2 numpy arrays for episode_lengths and episode_rewards
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    replay_memory = []

    # useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # get current time step
    total_t = sess.run(tf.train.get_global_step())

    # epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # q policy we are following
    policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))

    action_mapping = compose_action_from_id()
    # load initial experience into replay memory
    print("Populating replay memory...")
    state = env.reset()

    state, _, _ = env.step(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ADD]))

    # make the minimap data the state.
    #state = state[0].observation["rgb_minimap"].astype(np.uint8)
    #state = state_processor.process(sess, state)
    #state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        if i % 1000 == 0:
            print("iteration " + str(i))
        # according to policy, create a action probability array
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
        # randomly select an action according to action probs from policy
        action = np.random.choice(np.arange(len(VALID_ACTIONS)), p=action_probs)
        # action = action_mapping[action]
        action = action_mapping[0]

        # openAI gym take a step in action space
        next_state, reward, done = env.step(action)
        # process image data
        #next_state = state_processor.process(sess, next_state)
        #next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        # add action to replay memory
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            # if found goal, start over
            state = env.reset()
            state, _, _ = env.step(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ADD]))
            # make the minimap data the state.
            #state = state[0].observation["rgb_minimap"].astype(np.uint8)
            #state = state_processor.process(sess, state)
            #state = np.stack([state] * 4, axis=2)

        else:
            # if not found goal, update state to next state
            state = next_state

    # record videos
    # ad env monitor wrapper
    # Does this work for PySC2?
    #env = Monitor(env, directory=monitor_path, video_callable=lambda count: count % record_video_every == 0,
    #              resume=True)

    for i_episode in range(num_episodes):
        # save the current checkpoint
        if i_episode % 100 == 0:
            print("episode: " + str(i_episode))
        saver.save(tf.get_default_session(), checkpoint_path)

        # reset openAI environment
        state = env.reset()
        state, _, _ = env.step(actions.FunctionCall(_SELECT_ARMY, [_SELECT_ADD]))
        #state = state_processor.process(sess, state)
        #state = np.stack([state] * 4, axis=2)
        loss = None
        # main forloop after loading initial state
        for t in itertools.count():

            # epsilon for this timestep
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            # add epsilon to tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # maybe update the target estimator
            # update means copying parameters from q estimator -> target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # take the next step in the environment
            # similar to earlier when loading replay memory with first step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(VALID_ACTIONS)), p=action_probs)
            action = action_mapping[action]
            action = action_mapping[0]
            next_state, reward, done = env.step(action)
            #next_state = state_processor.process(sess, next_state)
            #next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            # if replay memory is full, pop
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # sample minibatch from replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # calculate qvalues and targets
            # Q ALGO RIGHT HERE LMAO
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.max(
                q_values_next, axis=1)

            # gradient descent
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1

    # Add summaries to tensorboard
    episode_summary = tf.Summary()
    episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward",
                              tag="episode_reward")
    episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length",
                              tag="episode_length")
    q_estimator.summary_writer.add_summary(episode_summary, total_t)
    q_estimator.summary_writer.flush()

    yield total_t, plotting.EpisodeStats(
        episode_lengths=stats.episode_lengths[:i_episode + 1],
        episode_rewards=stats.episode_rewards[:i_episode + 1])

    return stats


if __name__ == '__main__':
    agent_interface_format = features.parse_agent_interface_format(
        rgb_screen=84,
        rgb_minimap=84,
    )

    env = sc2_env.SC2Env(
        map_name="MoveToBeacon", #CollectMineralShards, DefeatRoaches, DefeatZerglingsAndBanelings
        step_mul=step_mul,
        agent_interface_format=agent_interface_format)

    env = PySC2EnvWrapper(env)
    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/")#{}".format(env.spec.id))
    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")
    # State processor
    #state_processor = StateProcessor()

    # Run it!
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in deep_q_learning(sess,
                                        env,
                                        q_estimator=q_estimator,
                                        target_estimator=target_estimator,
                                        experiment_dir=experiment_dir,
                                        num_episodes=10000,
                                        replay_memory_size=500000,
                                        replay_memory_init_size=1000,
                                        update_target_estimator_every=10000,
                                        epsilon_start=1.0,
                                        epsilon_end=0.1,
                                        epsilon_decay_steps=500000,
                                        discount_factor=0.99,
                                        batch_size=32):
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))