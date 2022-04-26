import numpy as np


l = ['a280', 'ali535', 'att48', 'att532', 'bayg29', 'bays29', 'berlin52', 'bier127', 'brazil58']

class Env:
    def __init__(self, arr):
        self.select = None
        self.arr = None
        self.now = 0
        self.finished = 0
        self.setup(arr)

    def setup(self, arr):
        self.arr = np.asarray(arr)
        self.select = np.zeros(self.arr.shape[0])
        self.now = 0
        self.finished = 0
        self.select[0] = 1

    def reset(self):
        self.select.fill(0)
        self.now = 0
        self.finished = 0
        self.select[0] = 1

    def sel(self, i):
        if self.select[i]:
            self.finished = 1
            return -1
        self.select[i] = 1

        if np.sum(self.select) == 1:
            a = np.where(self.select)[0][0]
            self.finished = 1
            return self.arr[self.now][i] + self.arr[self.now][a] + self.arr[a][0]
        self.now = i
        return self.arr[self.now][i]


def load_arr(i):
    return np.load('./dataset/{}.npy'.format(l[i]))
