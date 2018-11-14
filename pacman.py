#%%
import gym
import numpy as np
import tensorflow as tf

env = gym.make('MsPacman-ram-v0')

# learning parameters

# gamma
y = .99
# alpha
alpha = 0.7
# epsilon
e = 0.1
memory_size = 1000
num_episodes = 2000

# 128 bytes of atari console's RAM
env_features = 128

train_batch_size = 64
random_threshold = 1000
# %%


def weight_initialize(shape, name="weights"):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def bias_initialize(length, name="bias"):
    return tf.Variable(tf.constant(0.1, shape=[length]), name=name)


def create_layer(inputs, input_number, output_number, relu=True, name="layer"):
    weights = weight_initialize(shape=[input_number, output_number], name="{}_weights".format(name))
    biases = bias_initialize(length=output_number, name="{}_biases".format(name))
    layer = tf.matmul(inputs, weights) + biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer
#%%


table = np.zeros((memory_size, env_features * 2 + 2))


def transition_store(state, action, reward, observation):
    if 'table_index' not in globals():
        global table_index
        table_index = 0
    transition = np.hstack((state, [action, reward], observation))
    index = table_index % memory_size
    table[index: ] = transition
    table_index += 1

#%%
class Q_Network():
    def __init__(self, trainable=False, name="DQNetwork"):
        self.env_obs = tf.placeholder(shape=[None, env_features], dtype=tf.float32)
        self.layer_1 = create_layer(self.env_obs, env_features, 128, name="input_layer")
        self.layer_2 = create_layer(self.layer_1, 128, 512, name="hidden_1")
        self.layer_3 = create_layer(self.layer_2, 512, 512, name="hidden_2")
        self.layer_4 = create_layer(self.layer_3, 512, 128, name="hidden_3")
        self.layer_5 = create_layer(self.layer_4, 128, env.action_space.n, relu=False, name="output_1")
        self.q_output = tf.nn.softmax(self.layer_5, name="output_softmax")
        self.prediction = tf.argmax(self.q_output)
        if trainable:
            self.q_next = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_next, self.q_output))
            tf.summary.scalar('loss function', self.loss)
            self.training_operation = tf.train.AdamOptimizer(0.001).minimize(self.loss, name="adam_optim")


#%%
main_Q_network = Q_Network(trainable=True, name="main_net")
target_Q_network = Q_Network(name="target_name")

#%%
with tf.Session() as sess:
    frames = 0;
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("train_summary", sess.graph)
    final_reward = 0
    saver.save(sess, 'mspacman/model')
    for i in range(num_episodes):
        state = env.reset()
        done = False
        saver.restore(sess, "mspacman/model")
        while not done:
            env.render()
            frames += 1
            if frames < random_threshold or np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = sess.run([main_Q_network.prediction], feed_dict={main_Q_network.env_obs:[state]})
                action = np.argmax(action)

            observation, reward, done, _info = env.step(action)
            transition_store(state, action, reward, observation)
            if frames >= random_threshold and frames % 4 == 0:
                if table_index > memory_size:
                    sample_index = np.random.choice(memory_size, size = train_batch_size)
                else:
                    sample_index = np.random.choice(table_index, size=train_batch_size)
                batch = table[sample_index, :]
                obs = batch[:, :env_features]
                obs_next = batch[:, -env_features:]
                q_next = sess.run(target_Q_network.q_output, feed_dict={target_Q_network.env_obs: obs_next})
                q_eval_next = sess.run(main_Q_network.q_output, feed_dict={main_Q_network.env_obs: obs_next})
                q_eval = sess.run(main_Q_network.q_output, feed_dict={main_Q_network.env_obs: obs})
                q_target = q_eval.copy()
                batch_index = np.arange(train_batch_size, dtype=np.int32)
                eval_act_index = batch[:, env_features].astype(int)
                _reward = batch[:, env_features + 1]
                max_action = np.argmax(q_eval_next, axis=1)
                next_sel = q_next[batch_index, max_action]
                q_target[batch_index, eval_act_index] += alpha * (_reward + y * next_sel - q_target[batch_index, eval_act_index])
                sess.run(main_Q_network.training_operation, feed_dict={main_Q_network.env_obs: obs, main_Q_network.q_next: q_target})
                if done:
                    summary = sess.run(merged, feed_dict={main_Q_network.env_obs: obs, target_Q_network.env_obs: obs, main_Q_network.q_next: q_target})
                    writer.add_summary(summary, i)
            state = observation
            final_reward += reward

            if done:
                saver.save(sess, 'mspacman/model')
                print('Mean Reward in iteration {} is: '.format(i), final_reward/(i+1))
#%%