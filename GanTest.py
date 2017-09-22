import numpy as np
import tensorflow as tf

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
        self.gen_loss = -tf.reduce_mean(tf.log(1.0 - self.discr_gen + eps))

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
        fd = {self.state: state, self.noise: noise, self.action: action}
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
            activation_fn=tf.nn.relu,
            scope='state_layer1')

        noise_layer1 = tf.contrib.layers.fully_connected(noise, 
            num_outputs=32, 
            activation_fn=tf.nn.relu,
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


def plot_discr(gan):
    num = 200

    state = np.zeros((num, gan.state_dim))
    x = np.array([np.linspace(-3, 3, num)]).T
    probs = gan.get_discr_prob(state, x)
    plt.scatter(x, probs)

def plot_true(dist):
    num = 200
    x = np.array([np.linspace(-3, 3, num)]).T
    y = (1.0 / np.sqrt(2 * np.pi * dist.var)) * np.exp(-np.square(x - dist.mean) / (2 * dist.var))
    plt.scatter(x, y)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    state_dim = 2
    noise_dim = 2
    action_dim = 1

    # setting up network
    sess = tf.InteractiveSession()
    gan = ConditionalGAN(state_dim, noise_dim, action_dim, sess)
    tf.global_variables_initializer().run()
    
    # true distribution
    true_dist = GaussianDist(0.6, 0.2)

    # get initial distribution
    num_samples = 1000
    state = np.zeros((num_samples, state_dim))
    noise = np.random.random((num_samples, noise_dim))
    initial_sample = gan.sample_gen(state, noise)
    # n, bins, patches = plt.hist(initial_sample, range=[-2, 8], alpha=0.5, label='untrained')

    # plot_discr(gan)

    # pretraining
    pretrain_batch_size = 500
    state = np.zeros((pretrain_batch_size, state_dim))
    for i in range(10000):
        action = np.array([true_dist.sample(pretrain_batch_size)]).T

        hist, bin_edges = np.histogram(action, density=True)
        probs = hist * np.diff(bin_edges)
        prob_idx = np.argmin(np.maximum(action - bin_edges[:-1], 0), axis=1)
        targets = probs[prob_idx]

        # plt.scatter(action, targets)
        # plt.show()



        if i % 50 == 0:
            plt.cla()
            plot_discr(gan)
            plot_true(true_dist)
            plt.show(block=False)
            plt.pause(0.01)

        loss = gan.pretrain(state, action, targets)

        print("Pretrain Iteration " + str(i) + ": " + str(loss))

    plot_discr(gan)
    plt.show()

    exit()

    batch_size = 128
    state = np.zeros((batch_size, state_dim))

    # training
    for i in range(1000):
        noise = np.random.random((batch_size, noise_dim))
        action = np.array([true_dist.sample(batch_size)]).T

        for j in range(10):
            gan.train_discr(state, noise, action)

        noise = np.random.random((batch_size, noise_dim))
        gen_loss = gan.train_gen(state, noise)

        print("Iteration " + str(i) + ": " + str(gen_loss))

        if i % 50 == 0:
            x = np.array([np.linspace(0, 5, batch_size)]).T
            probs = gan.get_discr_prob(state, x)
            plt.scatter(x, probs)
            plt.show(block=False)
            plt.pause(0.1)

    state = np.zeros((num_samples, state_dim))
    noise = np.random.random((num_samples, noise_dim))
    final_samples = gan.sample_gen(state, noise)
    n, bins, patches = plt.hist(final_samples, alpha=0.5, label='trained')

    plt.cla()
    true_samples = true_dist.sample(num_samples)
    n, bins, patches = plt.hist(true_samples, alpha=0.5, label='true')

    plt.legend()
    plt.show()

