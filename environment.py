import numpy as np


l = ['a280', 'ali535', 'att48', 'att532', 'bayg29', 'bays29', 'berlin52', 'bier127', 'brazil58']
inf = 1e18


class Env:
    def __init__(self, arr):
        self.selected = None
        self.arr = np.asarray(arr)
        self.now = 0
        self.finished = 0
        self.n = 0
        self.setup()

    def setup(self):
        self.n = self.arr.shape[0]
        self.selected = [0 for x in range(self.n*2)]
        self.now = 0
        self.finished = 0
        self.selected[0] = 1

    def reset(self):
        self.setup()

    def sel(self, i):
        if self.selected[i]:
            self.finished = 1
            return inf
        self.selected[i] = 1

        if np.sum(self.selected) == self.n + 1:
            a = np.where(self.selected)[0][0]
            self.finished = 1
            return self.arr[self.now][i] + self.arr[self.now][a] + self.arr[a][0]
        v = self.arr[self.now][i]
        self.selected[self.now + self.n] = 0
        self.now = i
        self.selected[self.now + self.n] = 1
        return v

    def state(self):
        return self.selected

def load_arr(i):
    return np.load('./dataset/{}.npy'.format(l[i]))


def create_env(i):
    return Env(load_arr(i))


if __name__ == "__main__":
    arr = load_arr(5)
    env = Env(arr)
    print(env.n)
