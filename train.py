import numpy as np
from model import Model
from environment import Env, create_env, inf


res = []


def run(env: Env):
    episode = 1000
    gamma = 0.9
    epsilon = 1
    n = env.n
    model = Model(n)

    for e in range(1, episode+1):
        score = 0
        s1s = []
        s2s = []
        rs = []
        while not env.finished:
            s1 = env.state()
            s1s.append(s1)
            if epsilon >= np.random.rand():
                tmp = np.array([x*inf for x in s1[:n]])
                a1 = np.argmin(np.random.rand(n)+tmp)

            else:
                tmp = np.array([(not x) * inf for x in s1[:n]])
                p = model(np.array([s1, ]))[1][0]
                a1 = np.argmax(p + tmp)

            r = env.sel(a1)
            s2 = env.state()
            s2s.append(s2)
            rs.append(r)
            score += r
        res.append(score)
        s1 = np.array(s1s)
        s2 = np.array(s2s)
        r = np.array(rs)
        model.train(s1, s2, r)
        #print(score)
        epsilon *= 0.991
        env.reset()
        if e % 100 == 0:
            print(f"ep {e}: {score}")


if __name__ == '__main__':
    run(create_env(0))
    import matplotlib.pyplot as plt
    plt.plot(list(range(0, len(res))), res)
    plt.show()
