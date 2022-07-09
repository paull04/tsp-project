import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import init_model
from environment import Env, create_env


res = []


def run(env: Env):
    episode = 1000
    gamma = 0.9
    epsilon = 1
    n = env.n
    model = init_model(n)
    opt = Adam()

    for e in range(1, episode+1):
        score = 0
        while not env.finished:
            s1 = env.state()
            if epsilon >= np.random.rand():
                a1 = np.random.randint(0, n)

            else:
                p = model(np.array([s1, ]))
                a1 = np.argmax(p[0])
            r = env.sel(a1)
            print(r)

            s2 = env.state()
            params = model.trainable_variables
            with tf.GradientTape() as t:
                t.watch(params)
                action = tf.one_hot(a1, n)
                q = tf.reduce_sum(model(np.array([s1,])) * action)
                Qt = r + gamma * np.max(model(np.array([s2, ]))[0])
                loss = tf.reduce_min(tf.square(q-Qt))
            grad = t.gradient(loss, params)
            opt.apply_gradients(zip(grad, params))
            score += r
        res.append(score)
        print(score)
        epsilon *= 0.999
        env.reset()


if __name__ == '__main__':
    run(create_env(0))
    import matplotlib.pyplot as plt
    plt.plot(list(range(0, len(res))), res)
