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

    m3r = np.loadtxt('m3r.txt')
    m3t = np.loadtxt('m3t.txt')
    m3r, m3t = prior(m3r, m3t)
    data = {"time_step": m3t, "reward": m3r, "Algorithm": "d=3"}
    df_m3 = pd.DataFrame(data)

    m4r = np.loadtxt('m4r.txt')
    m4t = np.loadtxt('m4t.txt')
    m4r, m4t = prior(m4r, m4t)
    data = {"time_step": m4t, "reward": m4r, "Algorithm": "d=4"}
    df_m4 = pd.DataFrame(data)

    m5r = np.loadtxt('m5r.txt')
    m5t = np.loadtxt('m5t.txt')
    m5r, m5t = prior(m5r, m5t)
    data = {"time_step": m5t, "reward": m5r, "Algorithm": "d=5"}
    df_m5 = pd.DataFrame(data)

    m6r = np.loadtxt('m6r.txt')
    m6t = np.loadtxt('m6t.txt')
    m6r, m6t = prior(m6r, m6t)
    data = {"time_step": m6t, "reward": m6r, "Algorithm": "d=6"}
    df_m6 = pd.DataFrame(data)

    df = pd.concat([df_m3, df_m4, df_m5, df_m6])
    sns.relplot(x="time_step", y="reward", data=df, kind="line", hue="Algorithm")

    plt.ylabel("rewards")
    plt.xlabel("time steps")
    plt.title("Grid World")

    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.1)

    plt.savefig("result.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    image()
