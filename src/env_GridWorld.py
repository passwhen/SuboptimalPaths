import gym
from numpy import random
import numpy as np
from gym.envs.classic_control import rendering
import copy

class GridWorld(gym.Env):
    def __init__(self):
        self.viewer = None
        self.size = 20
        # 状态空间
        self.states = np.ones((self.size, self.size), dtype=int)
        self.state_dim = 2
        # 动作空间
        self.action_dim = 4
        # 初始状态
        self.state_init = [2, 2]
        # 目标状态
        self.state_target = [17, 10]
        # 墙的位置
        self.states[0, :] = 0
        self.states[19, :] = 0
        self.states[:, 0] = 0
        self.states[:, 19] = 0
        self.wall = [[1, 4], [1, 7], [1, 11], [1, 14], [1, 16],
                     [2, 4], [2, 7], [2, 11], [2, 14], [2, 16],
                     [3, 7], [3, 11], [3, 14], [3, 16],
                     [4, 1], [4, 2], [4, 3], [4, 4], [4, 6], [4, 7], [4, 8],
                     [4, 9], [4, 11], [4, 13], [4, 14], [4, 16], [4, 18],
                     [6, 4],
                     [7, 1], [7, 2], [7, 3], [7, 4], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10],
                     [7, 12], [7, 13], [7, 15], [7, 16], [7, 17], [7, 18],
                     [8, 4], [8, 7], [8, 10], [8, 13], [8, 16],
                     [9, 4], [9, 5], [9, 7], [9, 16],
                     [10, 7], [10, 13], [10, 14],
                     [11, 4], [11, 7], [11, 10], [11, 13], [11, 16],
                     [12, 1], [12, 2], [12, 3], [12, 4], [12, 6], [12, 7], [12, 9], [12, 10],
                     [12, 11], [12, 12], [12, 13], [12, 15], [12, 16], [12, 17], [12, 18],
                     [15, 1], [15, 2], [15, 3], [15, 4], [15, 6], [15, 7], [15, 8], [15, 10],
                     [15, 11], [15, 12], [15, 13], [15, 14], [15, 16],
                     [16, 7], [16, 13], [16, 16],
                     [17, 4], [17, 16], [18, 4], [18, 7], [18, 13], [18, 16]]
        for i, j in self.wall:
            self.states[i, j] = 0
        # self.states[self.wall] = 0
        # 奖励衰减
        self.reward_dec = 20
        # 初始化
        self.state = self.state_init.copy()
        self.hp_init = 5
        self.hp = self.hp_init
        self.reward_init = 1000
        self.reward = self.reward_init

    def reset(self):
        self.state = self.state_init.copy()
        self.hp = self.hp_init
        self.reward =self.reward_init
        return np.array(self.state)

    def step(self, action):
        a = action

        # reward
        if self.reward > 60:
            self.reward -= self.reward_dec
        info = 0
        pre_state = self.state.copy()
        done = False
        # 0 --> 上
        # 1 --> 右
        # 2 --> 下
        # 3 --> 左
        state = self.state.copy()
        if a == 0:
            state[0] = self.state[0] - 1
        elif a == 1:
            state[1] = self.state[1] + 1
        elif a == 2:
            state[0] = self.state[0] + 1
        elif a == 3:
            state[1] = self.state[1] - 1
        else:
            print("error")

        if self.states[state[0], state[1]] == 1:
            self.state = state.copy()
            if self.state == self.state_target:
                reward = self.reward
                done = True
                info = 1
            else:
                reward = 0
        else:
            state = self.state.copy()
            reward = -1
            # self.hp -= 1
            # if self.hp <= 0:
            #     done = True
            #     info = -1
        return reward, np.array(pre_state), np.array(state), done, info

    def virtual_step(self, state_in, action):
        a = action

        state_in = copy.deepcopy(state_in.astype(np.int))
        # reward
        info = 0
        pre_state = state_in.copy()
        done = False
        state = state_in.copy()
        if a == 0:
            state[0] = state_in[0] - 1
        elif a == 1:
            state[1] = state_in[1] + 1
        elif a == 2:
            state[0] = state_in[0] + 1
        elif a == 3:
            state[1] = state_in[1] - 1
        else:
            print("error")

        if self.states[state[0], state[1]] == 1:
            if (state == self.state_target).all():
                reward = self.reward
                done = True
                info = 1
            else:
                reward = 0
        else:
            state = state_in.copy()
            reward = -1
            # hp = self.hp - 1
            # if hp == 0:
            #     done = True
            #     info = -1
        return reward, np.array(pre_state), np.array(state), done, info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    env = GridWorld()
    for i in range(100):
        env.reset()
        ext_reward = 0
        while True:
            action = random.randint(0, env.action_dim - 1)
            # print(action)
            reward, pre_state, next_state, done, info = env.step(action)
            ext_reward += reward
            if done:
                env.reset()
                print("time:" + str(i) + "  reward: " + str(ext_reward) + "  End:", next_state)
                break

        env.close()
