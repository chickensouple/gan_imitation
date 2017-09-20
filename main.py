import numpy as np
import tensorflow as tf
import gym
from DDPG import DDPG

class Discriminator(object):
    def __init__(self, input, action_dim):
        self.state = tf.placeholder(tf.float32, [None, action_dim])
        self.input = input
        self._build_model()

    def _build_model(self):
        action_layer1 = tf.contrib.layers.fully_connected(self.input, 
            num_outputs=32, 
            activation_fn=tf.nn.relu,
            scope='action_layer1')

        state_layer1 = tf.contrib.layers.fully_connected(self.state, 
            num_outputs=64, 
            activation_fn=tf.nn.relu,
            scope='state_layer1')

        layer1 = tf.concat([action_layer1, state_layer1], 1)
        layer2 = tf.contrib.layers.fully_connected(layer1,
            num_outputs = 128,
            activation_fn=tf.nn.relu,
            scope='layer2')

        logit = tf.contrib.layers.fully_connected(layer2,
            num_outputs = 2,
            scope='logit')

        probs = tf.nn.softmax(logit)
        self.accept_prob = probs[0]



class Generator(object):
    def __init__(self, noise_dim, state_dim, action_range, sess):
        self.noise_dim = noise_dim
        self.noise = tf.placeholder(tf.float32, [None, noise_dim])
        self.state = tf.placeholder(tf.float32, [None, state_dim])
        self.sess = sess
        self.action_range = action_range

        self._build_model()

    def _build_model(self):
        noise_layer1 = tf.contrib.layers.fully_connected(self.noise, 
            num_outputs=32, 
            activation_fn=tf.nn.relu,
            scope='noise_layer1')

        state_layer1 = tf.contrib.layers.fully_connected(self.state, 
            num_outputs=64, 
            activation_fn=tf.nn.relu,
            scope='state_layer1')

        layer1 = tf.concat([noise_layer1, state_layer1], 1)
        layer2 = tf.contrib.layers.fully_connected(layer1,
            num_outputs = 128,
            activation_fn=tf.nn.relu,
            scope='layer2')

        self.output = tf.contrib.layers.fully_connected(layer2,
            num_outputs = 2,
            activation_fn=tf.nn.sigmoid,
            scope='actions')

    def get_action(self, state):
        noise = np.random.random(self.noise_dim)
        
        fd = {self.state: state, self.noise: noise}
        output = self.sess.run(self.output, feed_dict=fd)

        action_scale = action_range[1] - action_range[0]
        output[0] = action_scale * output[0] + action_range[0]

        noise = np.random.randn(output[1])

        return output[0] + noise


 
 # re-used for optimizing all networks
def optimizer(loss, var_list, num_decay_steps=400, initial_learning_rate=0.03):
    decay = 0.95
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer
   

def get_expert_traj(env, ddpg):
    state = env.reset()

    states = []
    actions = []

    while True:
        # get action
        action = ddpg.actor.policy(state)
        action = np.clip(action, 
            ddpg.action_mean - ddpg.action_scale, 
            ddpg.action_mean + ddpg.action_scale)
        action = np.array([action.squeeze()])

        [new_state, reward, done, _] = env.step(action)
        new_state = np.reshape(new_state, (1, ddpg.state_dim))
        total_reward += reward

        states.append(state)
        actions.append(action)
        
        state = new_state

        if done:
            break

    return states, actions


def get_generator_traj(env, gen):
    state = env.reset()

    states = []
    actions = []

    while True:
        # get action
        action = gen.get_action(state)

        [new_state, reward, done, _] = env.step(action)
        new_state = np.reshape(new_state, (1, ddpg.state_dim))
        total_reward += reward

        states.append(state)
        actions.append(action)
        
        state = new_state

        if done:
            break

    return states, actions


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    noise_dim = 2

    sess = tf.InteractiveSession()
    
    expert_actions = tf.placeholder(tf.float32, [None, state_dim])
    gen_actions = tf.placeholder(tf.float32, [None, state_dim])

    with tf.variable_scope('Generator'):
        gen = Generator(noise_dim, state_dim, [env.action_space.low, env.action_space.high], sess)
    with tf.variable_scope('Discriminator') as scope:
        D_fake = Discriminator(gen_actions, action_dim)
        scope.reuse_variables()
        D_real = Discriminator(expert_actions, action_dim)

    eps = 1e-2  # to prevent log(0) case
    loss_d = tf.reduce_mean(-tf.log(D_real.accept_prob + eps) - tf.log(1 - D_fake.accept_prob + eps))
    loss_g = tf.reduce_mean(-tf.log(D_fake.accept_prob + eps))
    
    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

    LR = 1e-3
    opt_d = optimizer(loss_d, d_params, 400, LR)
    opt_g = optimizer(loss_g, g_params, 400, LR / 2)

    tf.global_variables_initializer().run()

    ddpg = DDPG(state_dim, 
        action_dim, 
        [env.action_space.low, env.action_space.high], 
        sess=sess)
    ddpg.load_model()


    for i in range(1000):

        # get expert policies
        ddpg_states, ddpg_act = get_expert_traj(env, ddpg)

        # get generator policies
        gen_states, gen_act = get_generator_traj(env, gen)

        # updating discriminator
        fd = {expert_actions: ddpg_act,
            D_real.states: ddpg_states,
            gen_actions: gen_act,
            D_fake.state: gen_states}
        sess.run(opt_d, feed_dict=fd)

        # updating generator
        sess.run(opt_g, feed_dict=fd)

