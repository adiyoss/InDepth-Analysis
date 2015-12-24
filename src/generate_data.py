import numpy as np

__author__ = 'yossiadi'


def generate_data(num_samples=1000, size_x=2000, size_y=2):
    X = np.zeros((num_samples, size_x))
    X[0:num_samples / 2, ] = 1
    X[num_samples / 2:num_samples, ] = -1

    Y = np.ones((num_samples, size_y))

    Y[0:num_samples / 2, 0] = 0
    Y[0:num_samples / 2, 1] = 1
    Y[num_samples / 2:num_samples, 0] = 1
    Y[num_samples / 2:num_samples, 1] = 0

    return X, Y


def load_data(train_path, x_size=2000, y_size=2):
    fid = open(train_path)
    lines = fid.readlines()
    fid.close()

    x_train = np.zeros((len(lines), x_size))
    y_train = np.zeros((len(lines), y_size))

    for i in range(len(lines)):
        values = lines[i].split()
        target = float(values[0])
        y_train[i][int(target)] = 1.0
        x_train[i] = [float(x) for x in values[1:x_size+1]]
    return x_train, y_train
