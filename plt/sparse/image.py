import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
import pandas as pd


def prior(first, second):
    second = np.unique(second)
    first = first[:len(second)]
    first = first.flatten()
    second = np.repeat(second, 10)
    return first, second

def image():
    sns.set(style="darkgrid", font_scale=1)

    # 强化
    rl_reward = np.loadtxt('DQN_reward.txt')
    rl_time_step = np.loadtxt('DQN_time_step_list.txt')
    rl_reward, rl_time_step = prior(rl_reward, rl_time_step)
    data = {"time_step": rl_time_step, "reward": rl_reward, "Algorithm": "rl"}
    df_rl = pd.DataFrame(data)

    # 优化方法
    op_reward = np.loadtxt('optimization_reward.txt')
    op_time_step = np.loadtxt('optimization_time_step_list.txt')
    op_reward, op_time_step = prior(op_reward, op_time_step)
    data = {"time_step": op_time_step, "reward": op_reward, "Algorithm": "op-1st prior path"}
    df_op = pd.DataFrame(data)

    # 一条先验
    our_with_1path_reward = np.loadtxt('milestones_reward.txt')
    our_with_1path_time_step = np.loadtxt('milestones_time_step_list.txt')
    our_with_1path_reward, our_with_1path_time_step = prior(our_with_1path_reward, our_with_1path_time_step)
    data = {"time_step": our_with_1path_time_step, "reward": our_with_1path_reward, "Algorithm": "our-1sy prior path"}
    df_m1 = pd.DataFrame(data)

    # 两条先验
    our_with_2path_reward = np.loadtxt('multi_milestones_reward.txt')
    our_with_2path_time_step = np.loadtxt('multi_milestones_time_step_list.txt')
    our_with_2path_reward, our_with_2path_time_step = prior(our_with_2path_reward, our_with_2path_time_step)
    data = {"time_step": our_with_2path_time_step, "reward": our_with_2path_reward, "Algorithm": "our-both prior paths"}
    df_m2 = pd.DataFrame(data)

    df = pd.concat([df_rl, df_op, df_m1, df_m2])
    sns.relplot(x="time_step", y="reward", data=df, kind="line", hue="Algorithm")

    plt.ylabel("rewards")
    plt.xlabel("time steps")
    plt.title("Grid world")
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.1)

    plt.savefig("result.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    image()
