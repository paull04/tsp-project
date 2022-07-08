from collections import defaultdict
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import plot_model


def init_model(n):
    model = Sequential()
    model.add(Dense(256, input_shape = (n*2, )))
    model.add(Activation('relu'))
    return model


# SARSA: Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t+1 + gamma * maxQ(S_t+1, A_t+1) - Q(S_t, A_t))
