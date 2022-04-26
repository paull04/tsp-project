import numpy as np
import csv
import tsplib95 as t


def load(file):
    a = t.load('C:/Users/casu1/PycharmProjects/Yo my study/12/{0}.tsp/{0}.tsp'.format(file))
    nodes = list(a.get_nodes())
    l = len(nodes)
    arr = np.empty([l, l])
    for i in range(l):
        for j in range(l):
            arr[i][j] = a.get_weight(nodes[i], nodes[j])
    return arr


def save(arr: np.ndarray, name):
    np.save('./dataset/' + name, arr)


def sol():
    with open('solution.txt', 'r') as f:
        s = f.read()
        a = [x.split(' : ') for x in s.split('\n')]
        d = {}
        for x in a:
            d[x[0]] = float(x[1])

        with open('solution.csv', 'w') as f:
            writer = csv.writer(f)
            for e1, e2 in d.items():
                if e1:
                    writer.writerow([e1, e2])


def preprocess():
    sol()
    with open('solution.csv', mode='r') as f:
        reader = csv.reader(f)
        d = {x[0]: x[1] for x in reader if x}
        for x in d.keys():
            arr = load(x)
            save(arr, x)


if __name__ == '__main__':
    preprocess()
