from collections import defaultdict
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model as kModel
from tensorflow.keras.layers import Dense, InputLayer, ReLU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


def init_model(n):
    model = Sequential()
    model.add(InputLayer([2 * n, ]))
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(n))
    return model


class Model(kModel):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n1 = Dense(256, activation='elu')
        self.n2 = Dense(128, activation='elu')
        self.v = Dense(1)
        self.p = Dense(n, activation="softmax")
        self.opt = Adam(learning_rate=0.0001)
        self.loss = MeanSquaredError()
        self.gamma = 0.6

    def __call__(self, x):
        x = self.n1(x)
        x = self.n2(x)
        return self.v(x), self.p(x)

    def train(self, s1, s2, r):
        params = self.trainable_variables

        with tf.GradientTape() as t:
            t.watch(params)
            v1, p = self(s1)
            v2, p2 = self(s2)
            tar = r + self.gamma * v2
            loss1 = self.loss(tar, v1)
            loss2 = -tf.reduce_mean(tf.math.log(p), axis=1)*(tar - v1)
            loss = (loss1 + loss2)
        grad = t.gradient(loss, params)
        self.opt.apply_gradients(zip(grad, params))


# SARSA: Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t+1 + gamma * maxQ(S_t+1, A_t+1) - Q(S_t, A_t))
if __name__=='__main__':
    m = init_model(10)
    print(m.summary())
    print(m.predict(np.array([[0, 1] * 10, ])))