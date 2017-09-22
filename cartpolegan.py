import numpy as np
import tensorflow as tf
import gym
from DDPG import DDPG
import time

class ConditionalGAN(object):
    def __init__(self, state_dim, noise_dim, action_dim, sess):
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.action_dim = action_dim
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [None, state_dim])
        self.noise = tf.placeholder(tf.float32, [None, noise_dim])
        self.action = tf.placeholder(tf.float32, [None, action_dim])

        with tf.variable_scope('gen'):
            self.gen = ConditionalGAN._build_generator(self.state, self.noise, action_dim)

        with tf.variable_scope('discr') as scope:
            self.discr_gen = ConditionalGAN._build_discriminator(self.state, self.gen)
            scope.reuse_variables()
            self.discr_dist = ConditionalGAN._build_discriminator(self.state, self.action)

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discr')

        # for pretraining
        self.target_prob = tf.placeholder(tf.float32, [None])
        self.sloss = tf.reduce_mean(tf.square(self.target_prob - tf.squeeze(self.discr_dist)))
        self.sopt = tf.train.AdamOptimizer(1e-2).minimize(self.sloss)

        eps = 1e-2
        self.discr_loss = -tf.reduce_mean(tf.log(self.discr_dist + eps) + tf.log(1.0 - self.discr_gen + eps))
        self.gen_loss = -tf.reduce_mean(tf.log(self.discr_gen + eps))

        lr = 1e-3
        self.discr_opt = tf.train.AdamOptimizer(lr).minimize(self.discr_loss, var_list=discr_vars)
        self.gen_opt = tf.train.AdamOptimizer(lr * 0.5).minimize(self.gen_loss, var_list=gen_vars)

    def pretrain(self, state, action, targets):
        fd = {self.state: state, self.action: action, self.target_prob: targets}
        loss, _ = self.sess.run([self.sloss, self.sopt], feed_dict=fd)
        return loss

    def train_discr(self, state, noise, action):
        fd = {self.state: state, self.noise: noise, self.action: action}
        loss, _ = self.sess.run([self.discr_loss, self.discr_opt], feed_dict=fd)
        return loss

    def train_gen(self, state, noise):
        fd = {self.state: state, self.noise: noise}
        loss, _ = self.sess.run([self.gen_loss, self.gen_opt], feed_dict=fd)
        return loss

    def sample_gen(self, state, noise):
        fd = {self.state: state, self.noise: noise}
        samples = self.sess.run(self.gen, feed_dict=fd)
        return samples


    def get_discr_prob(self, state, action):
        fd = {self.state: state, self.action: action}
        probs = self.sess.run(self.discr_dist, feed_dict=fd)
        return probs

    @staticmethod
    def _build_generator(state, noise, action_dim):
        state_layer1 = tf.contrib.layers.fully_connected(state, 
            num_outputs=64, 
            activation_fn=tf.nn.elu,
            scope='state_layer1')

        noise_layer1 = tf.contrib.layers.fully_connected(noise, 
            num_outputs=32, 
            activation_fn=tf.nn.elu,
            scope='noise_layer1')


        layer1 = tf.concat([noise_layer1, state_layer1], 1)
        layer2 = tf.contrib.layers.fully_connected(layer1,
            num_outputs=128,
            activation_fn=tf.nn.relu,
            scope='layer2')

        output = tf.contrib.layers.fully_connected(layer2,
            num_outputs=action_dim,
            activation_fn=tf.nn.sigmoid,
            scope='actions')
        return output

    @staticmethod
    def _build_discriminator(state, action):
        state_layer1 = tf.contrib.layers.fully_connected(state, 
            num_outputs=64, 
            activation_fn=tf.nn.relu,
            scope='state_layer1')

        action_layer1 = tf.contrib.layers.fully_connected(action, 
            num_outputs=32, 
            activation_fn=tf.nn.relu,
            scope='action_layer1')

        layer1 = tf.concat([action_layer1, state_layer1], 1)
        layer2 = tf.contrib.layers.fully_connected(layer1,
            num_outputs = 128,
            activation_fn=tf.nn.relu,
            scope='layer2')

        prob = tf.contrib.layers.fully_connected(layer2,
            num_outputs = 1,
            activation_fn=tf.nn.sigmoid,
            scope='prob')
        return prob

class GaussianDist(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def sample(self, num):
        samples = np.random.randn(num) * self.var + self.mean
        return samples

def get_expert_traj(env, ddpg):
    state = env.reset()

    states = []
    actions = []

    total_reward = 0
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

        states.append(np.squeeze(state))
        actions.append(action)
        
        state = new_state

        if done:
            break

    return states, actions

def plot_discr(gan):
    num = 200

    state = np.zeros((num, gan.state_dim))
    x = np.array([np.linspace(-3, 3, num)]).T
    probs = gan.get_discr_prob(state, x)
    plt.scatter(x, probs, label='discr')


def plot_gen(gan):
    num = 200
    state = np.zeros((num, gan.state_dim))
    noise = np.random.random((num, gan.noise_dim))

    samples = gan.sample_gen(state, noise)

    hist, bin_edges = np.histogram(samples, density=True)
    x = np.array((bin_edges[:-1] + bin_edges[1:]) * 0.5)

    plt.scatter(x, hist, label='gen')

def plot_true(dist):
    num = 200
    x = np.array([np.linspace(-3, 3, num)]).T
    y = (1.0 / np.sqrt(2 * np.pi * dist.var)) * np.exp(-np.square(x - dist.mean) / (2 * dist.var))
    plt.scatter(x, y, label='true')



def rollout_gan(env, gan, render=True, max_iter=1000):
    state = env.reset()

    total_reward = 0
    for _ in range(max_iter):
        noise = np.random.random((1, gan.noise_dim))
         
        action = gan.sample_gen(np.reshape(state, (1, 3)), noise)
        #print "this is action executed" + str(action)
        if (render):
            env.render()
            time.sleep(0.01)
        [state, reward, done, info] = env.step(action)
        
        total_reward += reward

        if done:
            break
    return total_reward




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import sys
    import pickle
    from utils import rollout

    noise_dim = 2

    env = gym.make('Pendulum-v0')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]


    parser = argparse.ArgumentParser(description="Supervised Training")
    parser.add_argument('--type', dest='type', action='store',
        required=True,
        choices=['train', 'test'],
        help="type")
    parser.add_argument('--file', dest='file', action='store',
        default='data/model.ckpt',
        help='file to save model in')


    args = parser.parse_args(sys.argv[1:])

    # setting up network
    sess = tf.InteractiveSession()

    ddpg = DDPG(state_dim,action_dim,[env.action_space.low, env.action_space.high], sess=sess)
    gan = ConditionalGAN(state_dim, noise_dim, action_dim, sess)
    tf.global_variables_initializer().run()

    if args.type == 'train':
        get_state = lambda x: x
        for i in range(10000):
            total_reward = ddpg.update(env, get_state)
            print("Iteration " + str(i) + " reward: " + str(total_reward))
            if i % 20 == 0:
                [_, _, rewards] = rollout(env, ddpg.curr_policy(), get_state, render=True)
                total_reward = np.sum(np.array(rewards))
                print("Test reward: " + str(total_reward))

            if i % 50 == 0:
                ddpg.save_model(args.file)

        policy = ddpg.curr_policy()
        rollout(env, policy, get_state, render=True)
        ddpg.save_model(args.file)
    elif args.type == 'test':
        ddpg.load_model(args.file)

        # training
        for i in range(1000):
            ddpg_states, ddpg_act = get_expert_traj(env, ddpg)
            batch_size = len(ddpg_states)
            ddpg_states = np.array(ddpg_states)
            ddpg_act = np.array(ddpg_act)

            noise = np.random.random((batch_size, noise_dim))

            for j in range(10):
                disc_loss = gan.train_discr(ddpg_states, noise, ddpg_act)

            noise = np.random.random((batch_size, noise_dim))
            gen_loss = gan.train_gen(ddpg_states, noise)

            print("Iteration gan_loss " + str(i) + ": " + str(gen_loss))
            print("Iteration disc_loss" + str(i) + ": " + str(disc_loss))
            

            # merged = tf.summary.merge_all()

            # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
            #                           sess.graph)

            # tf.summary.scalar('gen_loss', gen_loss)
            # tf.summary.scalar('disc_loss', disc_loss)

            #testing####
            if i % 50 == 0:
                reward = rollout_gan(env, gan)
                print("testing reward: " + str(reward))


        plt.legend()
        plt.show()

