from collections import defaultdict
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, ReLU
from tensorflow.keras.utils import plot_model


def init_model(n):
    model = Sequential()
    model.add(InputLayer([2 * n, ]))
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dense(n))
    model.add(ReLU())
    return model


# SARSA: Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t+1 + gamma * maxQ(S_t+1, A_t+1) - Q(S_t, A_t))
if __name__=='__main__':
    m = init_model(10)
    print(m.summary())
    print(m.predict(np.array([[0, 1] * 10, ])))