from numpy.random import rand
from model import init
from environment import Env

def run(env: Env):
    episode = 1000
    gamma = 0.9
    epsilon = 1
    n = 0
    model = init()

    for e in range(1, episode+1):
        while not env.finished:



