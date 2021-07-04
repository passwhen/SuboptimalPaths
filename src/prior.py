import numpy as np


class Prior(object):
    def __init__(self):
        self.init = 0
        self.step = self.init

        self.path0 = [2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                      1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3,
                      3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        self.path1 = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      3]
        self.path = [self.path0, self.path1]

    def reset(self):
        self.step = self.init

    def choose_action(self, num):
        action = self.path[num][self.step]
        self.step += 1
        return np.array([action], dtype=np.float32)

    def reward(self, num):
        return 140 if num == 0 else 300
