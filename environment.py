import numpy as np


l = ['a280', 'ali535', 'att48', 'att532', 'bayg29', 'bays29', 'berlin52', 'bier127', 'brazil58']
inf = -1e9

class Env:
    def __init__(self, arr):
        self.selected = None
        self.arr = None
        self.now = 0
        self.finished = 0
        self.n = 0
        self.setup(arr)

    def setup(self, arr):
        self.arr = np.asarray(-arr)
        self.n = self.arr.shape[0]
        self.selected = [0 for x in range(self.arr.shape[0])]
        self.now = 0
        self.finished = 0
        self.selected[0] = 1

    def reset(self):
        self.setup(arr)

    def sel(self, i):
        if self.selected[i]:
            self.finished = 1
            return inf
        self.selected[i] = 1

        if np.sum(self.selected) == 1:
            a = np.where(self.selected)[0][0]
            self.finished = 1
            return self.arr[self.now][i] + self.arr[self.now][a] + self.arr[a][0]
        self.now = i
        return self.arr[self.now][i]


def load_arr(i):
    return np.load('./dataset/{}.npy'.format(l[i]))


def create_env(i):
    return Env(load_arr(i))


if __name__ == "__main__":
    arr = load_arr(0)
    env = Env(arr)
    print(env.arr)
