from numpy import random
import numpy as np
from env_GridWorld import GridWorld
from agent import ReplayBuffer, DQN
from prior import Prior
import copy
import seaborn as sns
import time


def distance(state_0, state_1):
    x_0 = state_0[0]
    y_0 = state_0[1]
    x_1 = state_1[0]
    y_1 = state_1[1]
    return (pow(x_0 - x_1, 2) + pow(y_0 - y_1, 2)) ** 0.5


def save_list(table, path):
    file = open(path, 'a')

    for fp in table:
        file.write(str(fp))

        file.write('\n')

    file.close()


def store_reward(line, path):
    file = open(path, 'a')
    s = str(line).replace('[', '').replace(']', '').replace(",", ' ') + '\n'
    file.write(s)
    file.close()


def one_hot(s, dim=20):
    s = copy.deepcopy(s.astype(np.int))
    state_one_hot = np.zeros(2 * dim)
    state_one_hot[s[0]] = 1
    state_one_hot[dim + s[1]] = 1
    return np.array(state_one_hot, dtype=np.float32)


class Run:
    def __init__(self):
        self.name = ""
        # 参数
        self.epsilon = 0.3
        # self.episodes = 10000
        self.episodes = 5000
        # self.max_steps = 64
        self.max_steps = 128
        self.batch = 128
        self.update_itr = 5

        # 环境初始化
        self.env = GridWorld()
        self.state_dim = 40
        self.action_dim = 4

        # 初始化状态
        state = self.env.reset()
        self.state = copy.deepcopy(state.astype(np.float32))
        self.pre_state = None

        # 缓存区
        self.replay_buffer = ReplayBuffer()
        # 网络
        self.hidden_dim = 128
        self.agent = DQN(self.state_dim, self.action_dim, self.hidden_dim, self.replay_buffer)
        state = state.astype(np.float32)
        self.agent.q_net([one_hot(state)])
        self.agent.target_q_net([one_hot(state)])

        # 先验路径
        self.prior = Prior()
        self.prior.reset()

        # 结束符
        self.is_end_init = False
        self.is_end = self.is_end_init

        # 外部奖励
        self.ext_reward_init = 0
        self.ext_reward = self.ext_reward_init

        # 奖励列表
        self.episode_list = []
        self.all_step_list = []
        self.sub_ext_reward_list = []
        self.ext_reward_list = []
        self.total_time_step = []
        self.reward_std_list = []
        self.time_list = []
        self.time_init = time.time()

        # 总时间步
        self.count_step = -1
        # 总轮次
        self.episode = 0
        # 死亡次数
        self.dead_times = 0
        # 到达终点次数
        self.success_times = 0
        # visited
        self.visited_list = []
        self.visited_init = np.zeros((20, 20), dtype=int)
        self.visited = copy.deepcopy(self.visited_init)
        self.visited_times = 0

    def visited_update(self):
        for i in self.visited_list:
            x = int(i[0])
            y = int(i[1])
            self.visited[x][y] += 1
        self.show_visited()
        self.visited = copy.deepcopy(self.visited_init)

    def save_visited(self):
        state = copy.deepcopy(self.state)
        self.visited_list.append(state)
        if len(self.visited_list) >= self.max_steps * 100:
            self.visited_update()
            self.visited_list = []

    def show_visited(self):
        self.visited_times += 1
        visited = np.array(self.visited)
        fig = sns.heatmap(visited, cmap='GnBu', cbar=False)
        heatmap = fig.get_figure()
        fig_path = "../data/visited_" + self.name + str(self.visited_times * 100) + ".png"
        heatmap.savefig(fig_path, dpi=400)

    def reinforcement_learning(self):
        self.name = "DQN"
        self.success_times = 0
        self.time_init = time.time()
        for ep in range(self.episodes):
            if ep % 100 == 0:
                print(ep)

            # 初始化
            self.state = copy.deepcopy(self.env.reset().astype(np.float32))
            self.pre_state = None
            ep_reward = 0
            for step in range(self.max_steps):

                self.count_step += 1
                # 引导探索
                if self.success_times >= 400:
                    self.epsilon = 0.1
                elif self.success_times >= 200:
                    self.epsilon = 0.2
                else:
                    self.epsilon = 0.3

                if np.random.uniform() < self.epsilon:
                    action = self.agent.q_net.sample_action()
                else:
                    action = self.agent.q_net.get_action(one_hot(self.state))

                reward, pre_state, state, done, info = self.env.step(action)
                self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
                self.state = copy.deepcopy(state.astype(np.float32))
                self.save_visited()
                ep_reward += reward

                self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state), 1 if done else 0)

                if self.replay_buffer.buffer_len() > self.batch and self.count_step % self.batch == 0:
                    for _ in range(self.update_itr):
                        self.agent.update(self.batch)

                # if done:
                if done or step == self.max_steps - 1:
                    if info == 1:
                        self.success_times += 1
                    elif info == -1:
                        self.dead_times += 1
                    # 奖励列表
                    self.sub_ext_reward_list.append(ep_reward)
                    if len(self.sub_ext_reward_list) == 10:
                        avg = np.mean(self.sub_ext_reward_list)
                        std = np.std(self.sub_ext_reward_list)
                        store_reward(self.sub_ext_reward_list, "../data/" + self.name + "_reward.txt")
                        self.sub_ext_reward_list = []

                        if len(self.ext_reward_list) == 0:
                            self.ext_reward_list.append(avg)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)
                            self.time_list.append(time.time() - self.time_init)
                        else:
                            self.ext_reward_list.append(self.ext_reward_list[-1] * 0.9 + avg * 0.1)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)
                            self.time_list.append(time.time() - self.time_init)

                        if len(self.ext_reward_list) >= 100:
                            save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
                            save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
                            save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
                            save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
                            save_list(self.time_list, "../data/" + self.name + "_time_list.txt")
                            self.episode_list = [self.episode_list[-1]]
                            self.ext_reward_list = [self.ext_reward_list[-1]]
                            self.total_time_step = [self.total_time_step[-1]]
                            self.reward_std_list = [self.reward_std_list[-1]]
                            self.time_list = [ self.time_list[-1]]

                        print(
                            "reinforcement;",
                            'step: %i;' % self.count_step,
                            'episode: %i;' % self.episode,
                            "last reward: %i;" % ep_reward,
                            "success: %i;" % self.success_times,
                            "death: %i;" % self.dead_times,
                            "last state: %r;" % self.state.tolist(),
                            "avg: %r;" % avg,
                            "avg reward: %r;" % (self.ext_reward_list[-1] if len(self.ext_reward_list) > 0 else 0),
                        )
                    self.episode += 1

                    # 重置
                    self.state = copy.deepcopy(self.env.reset().astype(np.float32))
                    self.pre_state = None
                    ep_reward = 0

                    break

        save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
        save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
        save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
        save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
        save_list(self.time_list, "../data/" + self.name + "_time_list.txt")
        self.agent.save()
        print("finish")

    def optimization(self, num=0):
        self.name = "optimization"
        # 模仿学习阶段
        lam = 0.8
        times = 0
        for ep in range(self.episodes):
            if ep % 100 == 0:
                print(ep)
            times = ep

            # 初始化
            self.state = copy.deepcopy(self.env.reset().astype(np.float32))
            self.pre_state = None
            self.prior.reset()

            ep_reward = 0
            if np.random.uniform() < lam:
                imitate = True
            else:
                imitate = False

            if lam <= 0.2:
                break

            for step in range(self.max_steps):
                self.count_step += 1
                # 学习
                if imitate:
                    self.epsilon = 0.5
                    action_prior = self.prior.choose_action(num)

                    if np.random.uniform() < self.epsilon:
                        action_net = self.agent.q_net.sample_action()
                    else:
                        action_net = self.agent.q_net.get_action(one_hot(self.state))

                    reward_net, pre_state, state_net, done_net, info_net = self.env.virtual_step(self.state, action_net)

                    reward, pre_state, state, done, info = self.env.step(action_prior)

                    # 网络输出和先验途径不同时给予惩罚
                    if not (state == state_net).all():
                        pre_state = copy.deepcopy(pre_state.astype(np.float32))
                        state_net = copy.deepcopy(state_net.astype(np.float32))
                        self.replay_buffer.push(one_hot(pre_state), action_net, reward_net - 1, one_hot(state_net),
                                                1 if done_net else 0)

                    # 实际利用先验路径前进
                    self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
                    self.state = copy.deepcopy(state.astype(np.float32))
                    self.replay_buffer.push(one_hot(self.pre_state), action_prior, reward + 1, one_hot(self.state),
                                            1 if done else 0)

                    if self.replay_buffer.buffer_len() > self.batch and self.count_step % self.batch == 0:
                        for _ in range(self.update_itr):
                            self.agent.update(self.batch)

                    if done:
                        self.episode += 1
                        # 重置
                        self.state = copy.deepcopy(self.env.reset().astype(np.float32))
                        self.prior.reset()
                        self.pre_state = None
                        ep_reward = 0

                        break

                else:
                    # 引导探索
                    if self.success_times >= 400:
                        self.epsilon = 0.1
                    elif self.success_times >= 200:
                        self.epsilon = 0.2
                    else:
                        # 保证探索性
                        self.epsilon = 0.3

                    if np.random.uniform() < self.epsilon:
                        action = self.agent.q_net.sample_action()
                    else:
                        action = self.agent.q_net.get_action(one_hot(self.state))

                    reward, pre_state, state, done, info = self.env.step(action)
                    self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
                    self.state = copy.deepcopy(state.astype(np.float32))
                    self.save_visited()
                    ep_reward += reward

                    self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state),
                                            1 if done else 0)

                    if self.replay_buffer.buffer_len() > self.batch and self.count_step % self.batch == 0:
                        for _ in range(self.update_itr):
                            self.agent.update(self.batch)

                    # if done:
                    if done or step == self.max_steps - 1:
                        if info == 1:
                            self.success_times += 1
                        elif info == -1:
                            self.dead_times += 1
                        # 奖励列表
                        self.sub_ext_reward_list.append(ep_reward)
                        if len(self.sub_ext_reward_list) == 10:
                            avg = np.mean(self.sub_ext_reward_list)
                            std = np.std(self.sub_ext_reward_list)
                            store_reward(self.sub_ext_reward_list, "../data/" + self.name + "_reward.txt")
                            self.sub_ext_reward_list = []
                            temp = self.prior.reward(num)

                            if len(self.ext_reward_list) == 0:
                                self.ext_reward_list.append(avg)
                                self.episode_list.append(self.episode)
                                self.total_time_step.append(self.count_step)
                                self.reward_std_list.append(std)
                            else:
                                self.ext_reward_list.append(self.ext_reward_list[-1] * 0.9 + avg * 0.1)
                                # 逐渐切换到强化学习，避免奖励突变
                                if self.ext_reward_list[-1] >= 1.2 * temp:
                                    lam = 0
                                elif self.ext_reward_list[-1] >= 0.95 * temp:
                                    lam = 0.1
                                elif self.ext_reward_list[-1] >= 0.8 * temp:
                                    lam = 0.2
                                elif self.ext_reward_list[-1] >= 0.4 * temp:
                                    lam = 0.5
                                else:
                                    lam = 0.8
                                self.episode_list.append(self.episode)
                                self.total_time_step.append(self.count_step)
                                self.reward_std_list.append(std)

                            if len(self.ext_reward_list) >= 100:
                                save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
                                save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
                                save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
                                save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
                                self.episode_list = [self.episode_list[-1]]
                                self.ext_reward_list = [self.ext_reward_list[-1]]
                                self.total_time_step = [self.total_time_step[-1]]
                                self.reward_std_list = [self.reward_std_list[-1]]

                            print(
                                "imitate;",
                                'step: %i;' % self.count_step,
                                'episode: %i;' % self.episode,
                                "last reward: %i;" % ep_reward,
                                "success: %i;" % self.success_times,
                                "death: %i;" % self.dead_times,
                                "last state: %r;" % self.state.tolist(),
                                "avg: %r;" % avg,
                                "avg reward: %r;" % (self.ext_reward_list[-1] if len(self.ext_reward_list) > 0 else 0),
                            )
                        self.episode += 1

                        # 重置
                        self.state = copy.deepcopy(self.env.reset().astype(np.float32))
                        self.prior.reset()
                        self.pre_state = None
                        ep_reward = 0

                        break

        # 继续优化
        self.success_times = 0
        for ep in range(times, self.episodes):
            if ep % 100 == 0:
                print(ep)

            # 初始化
            self.state = copy.deepcopy(self.env.reset().astype(np.float32))
            self.pre_state = None
            test = False
            ep_reward = 0
            for step in range(self.max_steps):

                self.count_step += 1
                # epsilon过高会导致智能体直接挂掉，无法正确获取到外部奖励
                self.epsilon = 0.1

                if np.random.uniform() < self.epsilon:
                    action = self.agent.q_net.sample_action()
                else:
                    action = self.agent.q_net.get_action(one_hot(self.state))

                reward, pre_state, state, done, info = self.env.step(action)
                self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
                self.state = copy.deepcopy(state.astype(np.float32))
                self.save_visited()
                ep_reward += reward

                self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state), 1 if done else 0)

                if self.replay_buffer.buffer_len() > self.batch and self.count_step % self.batch == 0:
                    self.agent.update(self.batch, op=True)

                # if done:
                if done or step == self.max_steps - 1:
                    if info == 1:
                        self.success_times += 1
                    elif info == -1:
                        self.dead_times += 1
                    # 奖励列表
                    self.sub_ext_reward_list.append(ep_reward)
                    if len(self.sub_ext_reward_list) == 10:
                        avg = np.mean(self.sub_ext_reward_list)
                        std = np.std(self.sub_ext_reward_list)
                        store_reward(self.sub_ext_reward_list, "../data/" + self.name + "_reward.txt")
                        self.sub_ext_reward_list = []

                        if len(self.ext_reward_list) == 0:
                            self.ext_reward_list.append(avg)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)
                        else:
                            self.ext_reward_list.append(self.ext_reward_list[-1] * 0.9 + avg * 0.1)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)

                        if len(self.ext_reward_list) >= 100:
                            save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
                            save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
                            save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
                            save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
                            self.episode_list = [self.episode_list[-1]]
                            self.ext_reward_list = [self.ext_reward_list[-1]]
                            self.total_time_step = [self.total_time_step[-1]]
                            self.reward_std_list = [self.reward_std_list[-1]]

                        print(
                            "optimization;",
                            'step: %i;' % self.count_step,
                            'episode: %i;' % self.episode,
                            "last reward: %i;" % ep_reward,
                            "success: %i;" % self.success_times,
                            "death: %i;" % self.dead_times,
                            "last state: %r;" % self.state.tolist(),
                            "avg: %r;" % avg,
                            "avg reward: %r;" % (self.ext_reward_list[-1] if len(self.ext_reward_list) > 0 else 0),
                        )
                    self.episode += 1

                    # 重置
                    self.state = copy.deepcopy(self.env.reset().astype(np.float32))
                    self.pre_state = None
                    ep_reward = 0

                    break

        save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
        save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
        save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
        save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
        self.agent.save()
        print("finish")

    def milestones(self, interval, num):
        self.name = "milestones"
        # 初始化内在奖励
        milestones_reward = 8
        milestones = []
        visited = []
        leap = []
        done = False
        self.state = copy.deepcopy(self.env.reset().astype(np.float32))
        self.pre_state = None
        self.prior.reset()

        # 设置里程碑
        for step in range(self.max_steps):
            action_prior = self.prior.choose_action(num)

            reward, pre_state, state, done, info = self.env.step(action_prior)
            self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
            self.state = copy.deepcopy(state.astype(np.float32))
            if (step + 1) % interval == 0:
                milestones.append(self.state)
                visited.append(0)
                leap.append(milestones_reward * len(milestones))

            if done:
                break

        # 将终点加入里程碑
        if not (self.state == milestones[-1]).all():
            milestones.append(self.state)
            visited.append(0)
            leap.append(milestones_reward * len(milestones))

        self.time_init = time.time()
        max_top = 0
        for ep in range(self.episodes):
            if ep % 100 == 0:
                print(ep)
            # 下一个搜寻的里程碑
            top = 0
            self.state = copy.deepcopy(self.env.reset().astype(np.float32))
            last_state = copy.deepcopy(self.state)
            self.pre_state = None
            ep_reward = 0
            self.epsilon = 0.3
            for step in range(self.max_steps):
                self.count_step += 1
                if top < len(visited):
                    temp = max(visited[top:])
                    if temp >= 500:
                        self.epsilon = 0.1
                    elif temp >= 300:
                        self.epsilon = 0.2
                    else:
                        self.epsilon = 0.3
                if np.random.uniform() < self.epsilon:
                    action = self.agent.q_net.sample_action()
                else:
                    action = self.agent.q_net.get_action(one_hot(self.state))

                reward, pre_state, state, done, info = self.env.step(action)
                self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
                self.state = copy.deepcopy(state.astype(np.float32))
                self.save_visited()
                ep_reward += reward

                # 内在奖励
                int_r = 0
                for i in range(top, len(milestones)):
                    if distance(self.state, milestones[i]) <= 1:
                        visited[i] += 1
                        last_state = copy.deepcopy(self.state)
                        for j in range(top, i + 1):
                            int_r += leap[j] * (0.8 ** (i - j)) * (0.9 ** distance(self.state, self.env.state_init))
                        top = i + 1
                        break

                if distance(last_state, self.state) >= interval * 1.5:
                    int_r += milestones_reward * 0.8
                    last_state = copy.deepcopy(self.state)

                reward += int_r

                if temp >= 1000:
                    if np.random.uniform() < 0.4:
                        self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state),
                                                1 if done else 0)
                else:
                    self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state),
                                            1 if done else 0)

                if self.replay_buffer.buffer_len() > self.batch and self.count_step % self.batch == 0:
                    if visited[-1] > 300:
                        self.agent.update(self.batch, op=True)
                    else:
                        for _ in range(self.update_itr):
                            self.agent.update(self.batch)

                # if done:
                if done or step == self.max_steps - 1:
                    max_top = max(top, max_top)
                    if info == 1:
                        self.success_times += 1
                    elif info == -1:
                        self.dead_times += 1

                    # 奖励列表
                    self.sub_ext_reward_list.append(ep_reward)
                    if len(self.sub_ext_reward_list) == 10:
                        avg = np.mean(self.sub_ext_reward_list)
                        std = np.std(self.sub_ext_reward_list)
                        store_reward(self.sub_ext_reward_list, "../data/" + self.name + "_reward.txt")
                        self.sub_ext_reward_list = []

                        if len(self.ext_reward_list) == 0:
                            self.ext_reward_list.append(avg)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)
                        else:
                            self.ext_reward_list.append(self.ext_reward_list[-1] * 0.9 + avg * 0.1)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)

                        if len(self.ext_reward_list) >= 100:
                            save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
                            save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
                            save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
                            save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
                            self.episode_list = [self.episode_list[-1]]
                            self.ext_reward_list = [self.ext_reward_list[-1]]
                            self.total_time_step = [self.total_time_step[-1]]
                            self.reward_std_list = [self.reward_std_list[-1]]

                        print(
                            "milestones;",
                            'step: %i;' % self.count_step,
                            'episode: %i;' % self.episode,
                            "last reward: %i;" % ep_reward,
                            "success: %i;" % self.success_times,
                            "death: %i;" % self.dead_times,
                            "last state: %r;" % self.state.tolist(),
                            "top: %r;" % top,
                            "max top: %r;" % max_top,
                            "avg: %r;" % avg,
                            "avg reward: %r;" % (self.ext_reward_list[-1] if len(self.ext_reward_list) > 0 else 0),
                            "visited: %r;" % visited,
                        )
                    self.episode += 1

                    # 重置
                    self.state = copy.deepcopy(self.env.reset().astype(np.float32))
                    self.prior.reset()
                    self.pre_state = None
                    ep_reward = 0
                    top = 0

                    break

        save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
        save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
        save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
        save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
        self.agent.save()
        print("finish")

    def multi_milestones(self, interval):
        # 多条先验路径，添加一个条件，只能获得比当前位置离终点更近的里程碑的奖励
        self.name = "multi_milestones"
        # 奖励大小需要根据interval调整，重要参数
        milestones_reward = 5

        # 第一个里程碑序列
        milestones0 = []
        visited0 = []
        leap0 = []
        done = False
        self.state = copy.deepcopy(self.env.reset().astype(np.float32))
        self.pre_state = None
        self.prior.reset()

        # 设置里程碑
        for step in range(self.max_steps):
            action_prior = self.prior.choose_action(0)

            reward, pre_state, state, done, info = self.env.step(action_prior)
            self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
            self.state = copy.deepcopy(state.astype(np.float32))
            if (step + 1) % interval == 0:
                milestones0.append(self.state)
                visited0.append(0)
                leap0.append(milestones_reward * len(milestones0))

            if done:
                break

        # 将终点加入里程碑
        if not (self.state == milestones0[-1]).all():
            milestones0.append(self.state)
            visited0.append(0)
            leap0.append(milestones_reward * len(milestones0))

        # 第二个里程碑序列
        milestones1 = []
        visited1 = []
        leap1 = []
        done = False
        self.state = copy.deepcopy(self.env.reset().astype(np.float32))
        self.pre_state = None
        self.prior.reset()

        # 设置里程碑
        for step in range(self.max_steps):
            action_prior = self.prior.choose_action(1)

            reward, pre_state, state, done, info = self.env.step(action_prior)
            self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
            self.state = copy.deepcopy(state.astype(np.float32))
            if (step + 1) % interval == 0:
                milestones1.append(self.state)
                visited1.append(0)
                leap1.append(milestones_reward * len(milestones1))

            if done:
                break

        # 将终点加入里程碑
        if not (self.state == milestones1[-1]).all():
            milestones1.append(self.state)
            visited1.append(0)
            leap1.append(milestones_reward * len(milestones1))

        # 目标
        goal = copy.deepcopy(self.state)

        # 统一奖励 将长的奖励改小
        reward = 0
        length = min(len(milestones0), len(milestones1))
        for i in range(length):
            reward += leap0[i]

        count = (1 + len(leap0)) * len(leap0) / 2
        for i in range(0, len(leap0)):
            leap0[i] = (i + 1) / count * reward

        count = (1 + len(leap1)) * len(leap1) / 2
        for i in range(0, len(leap1)):
            leap1[i] = (i + 1) / count * reward

        max_top0 = 0
        max_top1 = 0
        for ep in range(self.episodes):
            if ep % 100 == 0:
                print(ep)
            # 下一个搜寻的里程碑
            top0 = 0
            top1 = 0
            temp0 = max(visited0[top0:])
            temp1 = max(visited1[top1:])

            # 初始化
            self.state = copy.deepcopy(self.env.reset().astype(np.float32))
            last_state = copy.deepcopy(self.state)
            self.pre_state = None
            ep_reward = 0
            self.epsilon = 0.3
            for step in range(self.max_steps):
                self.count_step += 1

                # 判定epsilon大小
                if top0 < len(visited0):
                    temp0 = max(visited0[top0:])
                if top1 < len(visited1):
                    temp1 = max(visited1[top1:])
                if temp0 >= 500 or temp1 >= 500:
                    self.epsilon = 0.1
                elif temp0 >= 300 or temp1 >= 300:
                    self.epsilon = 0.2
                else:
                    self.epsilon = 0.3

                # 行动
                if np.random.uniform() < self.epsilon:
                    action = self.agent.q_net.sample_action()
                else:
                    action = self.agent.q_net.get_action(one_hot(self.state))

                reward, pre_state, state, done, info = self.env.step(action)
                self.pre_state = copy.deepcopy(pre_state.astype(np.float32))
                self.state = copy.deepcopy(state.astype(np.float32))
                self.save_visited()
                ep_reward += reward

                # 内在奖励
                int_r = 0
                for i in range(top0, len(milestones0)):
                    if distance(self.state, milestones0[i]) <= 1 and \
                            distance(last_state, goal) - distance(self.state, goal) > 2:
                        visited0[i] += 1
                        last_state = copy.deepcopy(self.state)
                        for j in range(top0, i + 1):
                            int_r += leap0[j] * (0.8 ** (i - j)) * (0.9 ** distance(self.state, self.env.state_init))
                        top0 = i + 1
                        for k in range(top1, len(milestones1)):
                            if distance(milestones0[i], milestones1[k]) <= 1:
                                visited1[k] += 1
                                top1 = k + 1
                        break

                for i in range(top1, len(milestones1)):
                    if distance(self.state, milestones1[i]) <= 1 and \
                            distance(last_state, goal) - distance(self.state, goal) > 2:
                        visited1[i] += 1
                        last_state = copy.deepcopy(self.state)
                        for j in range(top1, i + 1):
                            int_r += leap1[j] * (0.8 ** (i - j)) * (0.9 ** distance(self.state, self.env.state_init))
                        top1 = i + 1
                        for k in range(top0, len(milestones0)):
                            if distance(milestones1[i], milestones0[k]) <= 1:
                                visited0[k] += 1
                                top0 = k + 1
                        break

                if distance(last_state, self.state) >= interval * 1.5:
                    int_r += milestones_reward * 0.8
                    last_state = copy.deepcopy(self.state)

                reward += int_r

                if temp0 >= 1000 or temp1 >= 1000:
                    if np.random.uniform() < 0.4:
                        self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state),
                                                1 if done else 0)
                else:
                    self.replay_buffer.push(one_hot(self.pre_state), action, reward, one_hot(self.state),
                                            1 if done else 0)

                if self.replay_buffer.buffer_len() > self.batch and self.count_step % self.batch == 0:
                    for _ in range(self.update_itr):
                        if visited0[-1] > 300 or visited1[-1] > 300:
                            self.agent.update(self.batch, op=True)
                        else:
                            self.agent.update(self.batch)

                # if done:
                if done or step == self.max_steps - 1:
                    max_top0 = max(top0, max_top0)
                    max_top1 = max(top1, max_top1)
                    if info == 1:
                        self.success_times += 1
                    elif info == -1:
                        self.dead_times += 1

                    # 奖励列表
                    self.sub_ext_reward_list.append(ep_reward)
                    if len(self.sub_ext_reward_list) == 10:
                        avg = np.mean(self.sub_ext_reward_list)
                        std = np.std(self.sub_ext_reward_list)
                        store_reward(self.sub_ext_reward_list, "../data/" + self.name + "_reward.txt")
                        self.sub_ext_reward_list = []

                        if len(self.ext_reward_list) == 0:
                            self.ext_reward_list.append(avg)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)
                            self.time_list.append(time.time() - self.time_init)
                        else:
                            self.ext_reward_list.append(self.ext_reward_list[-1] * 0.9 + avg * 0.1)
                            self.episode_list.append(self.episode)
                            self.total_time_step.append(self.count_step)
                            self.reward_std_list.append(std)
                            self.time_list.append(time.time() - self.time_init)

                        if len(self.ext_reward_list) >= 100:
                            save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
                            save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
                            save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
                            save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
                            save_list(self.time_list, "../data/" + self.name + "_time_list.txt")
                            self.episode_list = [self.episode_list[-1]]
                            self.ext_reward_list = [self.ext_reward_list[-1]]
                            self.total_time_step = [self.total_time_step[-1]]
                            self.reward_std_list = [self.reward_std_list[-1]]
                            self.time_list = [self.time_list[-1]]

                        print(
                            "milestones;",
                            'step: %i;' % self.count_step,
                            'episode: %i;' % self.episode,
                            "last reward: %i;" % ep_reward,
                            "success: %i;" % self.success_times,
                            "death: %i;" % self.dead_times,
                            "last state: %r;" % self.state.tolist(),
                            "top0: %r;" % top0,
                            "top1: %r;" % top1,
                            "max top0: %r;" % max_top0,
                            "max top1: %r;" % max_top1,
                            "avg: %r;" % avg,
                            "avg reward: %r;" % (self.ext_reward_list[-1] if len(self.ext_reward_list) > 0 else 0),
                            "visited0: %r;" % visited0,
                            "visited1: %r;" % visited1,
                        )
                    self.episode += 1

                    # 重置
                    self.state = copy.deepcopy(self.env.reset().astype(np.float32))
                    self.prior.reset()
                    self.pre_state = None
                    ep_reward = 0
                    top = 0

                    break

        save_list(self.episode_list, "../data/" + self.name + "_episode_list.txt")
        save_list(self.ext_reward_list, "../data/" + self.name + "_ext_reward_list.txt")
        save_list(self.total_time_step, "../data/" + self.name + "_time_step_list.txt")
        save_list(self.reward_std_list, "../data/" + self.name + "_reward_std_list.txt")
        self.agent.save()
        print("finish")

if __name__ == "__main__":
    model = Run()
    model.reinforcement_learning()
    # model.optimization()
    # model.milestones(5, 0)
    # model.multi_milestones(5)
    model.show_visited()
    print("finish")
