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
        self.discr_loss = tf.reduce_mean(tf.log(self.discr_dist + eps) + tf.log(1.0 - self.discr_gen + eps))
        self.gen_loss = tf.reduce_mean(tf.log(1.0 - self.discr_gen + eps))

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


def stratified_sample(size, range=[0, 1]):
    col = np.linspace(range[0], range[1], size[0])
    
    samples = np.tile(col, [size[1], 1]).T
    samples += np.random.random(size) / (size[0] * (range[1] - range[0]))

    return samples

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    state_dim = 2
    noise_dim = 1
    action_dim = 1

    # setting up network
    sess = tf.InteractiveSession()
    gan = ConditionalGAN(state_dim, noise_dim, action_dim, sess)
    tf.global_variables_initializer().run()
    
    # true distribution
    true_dist = GaussianDist(1.2, 1.)

    # pretraining
    pretrain_batch_size = 500
    for i in range(200):
        samples = np.array([true_dist.sample(pretrain_batch_size)]).T

        hist, bin_edges = np.histogram(samples, density=True)
        action = np.array([(bin_edges[:-1] + bin_edges[1:]) * 0.5]).T
        state = np.zeros((len(action), state_dim))

        if i % 50 == 0:
            plt.cla()
            plot_discr(gan)
            plot_true(true_dist)
            plt.show(block=False)
            plt.pause(0.01)

        loss = gan.pretrain(state, action, hist)

        print("Pretrain Iteration " + str(i) + ": " + str(loss))


    batch_size = 128
    state = np.zeros((batch_size, state_dim))

    # training
    for i in range(2000):
        # noise = np.random.random((batch_size, noise_dim))
        noise = stratified_sample((batch_size, noise_dim))

        action = np.array([true_dist.sample(batch_size)]).T

        for j in range(10):
            gan.train_discr(state, noise, action)

        noise = stratified_sample((batch_size, noise_dim))
        gen_loss = gan.train_gen(state, noise)

        print("Iteration " + str(i) + ": " + str(gen_loss))

        if i % 50 == 0:
            plt.cla()
            plot_discr(gan)
            plot_true(true_dist)
            plot_gen(gan)
            plt.legend()
            plt.show(block=False)
            plt.pause(0.1)

    plt.cla()
    plot_discr(gan)
    plot_true(true_dist)
    plot_gen(gan)
    plt.legend()

    plt.legend()
    plt.show()

