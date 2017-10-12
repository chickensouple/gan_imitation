import numpy as np
import tensorflow as tf
import gym
import time


class MLP(object):
    def __init__(self, state_dim, action_dim, action_range, sess):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [None, state_dim])
        self.action = tf.placeholder(tf.float32, [None, action_dim])

        self.output = self._build_model(self.state, action_range)
        self.loss = tf.reduce_mean(tf.square(self.output - self.action))

        self.opt = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def train(self, states, actions):
        fd = {self.state: states, self.action: actions}
        loss, _ = self.sess.run([self.loss, self.opt], feed_dict=fd)
        return loss

    def get_action(self, states):
        fd = {self.state: states}
        output = self.sess.run(self.output, feed_dict=fd)
        return output

    def _build_model(self, input, action_range):
        layer1 = tf.contrib.layers.fully_connected(input, 
            num_outputs=64+32, 
            activation_fn=tf.nn.elu,
            scope='layer1')

        layer2 = tf.contrib.layers.fully_connected(layer1,
            num_outputs=128,
            activation_fn=tf.nn.relu,
            scope='layer2')

        layer3 = tf.contrib.layers.fully_connected(layer2,
            num_outputs=64,
            activation_fn=tf.nn.relu,
            scope='layer3')

        layer4 = tf.contrib.layers.fully_connected(layer3,
            num_outputs=32,
            activation_fn=tf.nn.relu,
            scope='layer4')

        output = tf.contrib.layers.fully_connected(layer4,
            num_outputs=action_dim,
            activation_fn=tf.nn.sigmoid,
            scope='actions')

        output = output * (action_range[1] - action_range[0]) + action_range[0]
        
        return output


    def save_model(self, filename='/tmp/model.ckpt'):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filename)
        print("Model saved in file: %s" % filename)

    def load_model(self, filename='/tmp/model.ckpt'):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print("Model loaded from file: %s" % filename)



def rollout_mlp(env, mlp, render=True, max_iter=1000):
    state = env.reset()

    total_reward = 0
    for _ in range(max_iter):
        action = mlp.get_action(np.reshape(state, (1, mlp.state_dim)))

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

    noise_dim = 100

    env = gym.make('Pendulum-v0')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_range = [env.action_space.low, env.action_space.high]


    parser = argparse.ArgumentParser(description="Supervised Training")
    parser.add_argument('--type', dest='type', action='store',
        required=True,
        choices=['train', 'test'],
        help="type")
    parser.add_argument('--file', dest='file', action='store',
        default='data/mlp_model.ckpt',
        help='file to save model in')

    args = parser.parse_args(sys.argv[1:])

    # setting up network
    sess = tf.InteractiveSession()
    mlp = MLP(state_dim, action_dim, action_range, sess)
    tf.global_variables_initializer().run()

    # read in expert data
    expert_data = pickle.load(open('data/data.p', 'rb'))
    num_expert_data = len(expert_data['states'])


    batch_size = 512
    if args.type == 'train':
        # training
        for i in range(1000000):
            data_idx = np.random.randint(num_expert_data, size=batch_size)

            expert_states = expert_data['states'][data_idx, :]
            expert_actions = expert_data['actions'][data_idx, :]

            loss = mlp.train(expert_states, expert_actions)

            print("Iteration " + str(i) + " loss: " + str(loss))


            if i % 50 == 0:
                reward = rollout_mlp(env, mlp)
                print("testing reward: " + str(reward))


            if i % 500 == 0:
                mlp.save_model(args.file)

