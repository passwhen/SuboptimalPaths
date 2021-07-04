import argparse
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)


ALG_NAME = 'DQN'
ENV_ID = 'GridWorld'

Q_LR = 3e-3
REPLAY_BUFFER_SIZE = 1024

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        return state, action, reward, next_state, done

    def buffer_len(self):
        return len(self.buffer)


class QNetwork(Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=num_actions, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def evaluate(self, state):
        state = state.astype(np.float32)
        action = self.forward(state)

        action = np.argmax(action.numpy(), 1)
        return action

    def get_action(self, state):
        state = state.astype(np.float32)
        action = self.forward([state])
        action = np.argmax(action.numpy(), 1)
        return action

    def sample_action(self):
        a = np.random.randint(0, 4, 1)
        return a


class DQN:
    def __init__(
            self, state_dim, action_dim, hidden_dim, replay_buffer, target_update_interval=2,
            q_lr=Q_LR
    ):
        self.replay_buffer = replay_buffer

        # 初始化网络
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim)
        print('Q Network: ', self.q_net)

        # 初始化
        self.target_q_net = self.target_ini(self.q_net, self.target_q_net)
        self.q_net.train()
        self.target_q_net.eval()

        self.update_cnt = 0
        self.target_update_interval = target_update_interval

        self.q_optimizer = tf.optimizers.Adam(q_lr)
        self.q_optimizer_ = tf.optimizers.Adam(q_lr * 0.05)

    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, gamma=0.8, soft_tau=1e-1, op=False):
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        new_next_action = self.q_net.evaluate(next_state)

        target = self.target_q_net(state).numpy()
        next_target = self.target_q_net(next_state).numpy()
        target_q_value = []
        for i in range(batch_size):
            target_q_value.append(next_target[i][new_next_action[i]])

        target_q_value = np.array(target_q_value)
        target_q_value = reward + (1 - done) * gamma * target_q_value
        target[range(batch_size), np.reshape(action, batch_size).astype(np.int)] = target_q_value

        with tf.GradientTape() as q_tape:
            predicted_q_value = self.q_net(state)
            q_value_loss = tf.losses.mean_squared_error(target, predicted_q_value)
        q_grad = q_tape.gradient(q_value_loss, self.q_net.trainable_weights)
        if not op:
            self.q_optimizer.apply_gradients(zip(q_grad, self.q_net.trainable_weights))
        else:
            self.q_optimizer_.apply_gradients(zip(q_grad, self.q_net.trainable_weights))

        if self.update_cnt % self.target_update_interval == 0:
            self.target_q_net = self.target_soft_update(self.q_net, self.target_q_net, soft_tau)

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.q_net.trainable_weights, extend_path('model_q_net.npz'))
        tl.files.save_npz(self.target_q_net.trainable_weights, extend_path('model_target_q_net.npz'))

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net.npz'), self.q_net)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net.npz'), self.target_q_net)











