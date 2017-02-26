import numpy as np

from src.enum_tests import Tests

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
        x_train[i] = [float(x) for x in values[1:x_size + 1]]
    return x_train, y_train


def load_repr_data(train_path, x_size=2000, y_size=1000):
    fid = open(train_path)
    lines = fid.readlines()
    fid.close()

    x_train = np.zeros((len(lines), x_size))
    y_train = np.zeros((len(lines), y_size))

    for i in range(len(lines)):
        values = lines[i].split()
        y_train[i] = [float(x) for x in values[0:y_size]]
        x_train[i] = [float(x) for x in values[y_size:x_size + y_size + 1]]

    return x_train, y_train


def build_data(word_rep_path, train_idx_path, train_rep_path, test_idx_path, test_rep_path, val_idx_path, val_rep_path,
               rep_dim_w, rep_dim_s, task=1):
    word_rep = np.load(word_rep_path)
    train_rep = np.load(train_rep_path)
    test_rep = np.load(test_rep_path)
    val_rep = np.load(val_rep_path)

    if (task is Tests.FIRST_WORD) or (task is Tests.LAST_WORD) or (task is Tests.RANDOM_WORD):
        y_size = 2
        x_size = rep_dim_s + rep_dim_w
        train_size = 300000
        test_size = 50000
        val_size = 50000

        x_train, y_train = populate_vector(train_idx_path, word_rep, train_rep, train_size, x_size, y_size, rep_dim_s,
                                           rep_dim_w)
        x_test, y_test = populate_vector(test_idx_path, word_rep, test_rep, test_size, x_size, y_size, rep_dim_s,
                                         rep_dim_w)
        x_val, y_val = populate_vector(val_idx_path, word_rep, val_rep, val_size, x_size, y_size, rep_dim_s, rep_dim_w)
    elif task is Tests.SENTENCE_LENGTH:
        y_size = 8
        x_size = rep_dim_s
        train_size = 150000
        test_size = 25000
        val_size = 25000

        x_train, y_train = populate_vector(train_idx_path, word_rep, train_rep, train_size, x_size, y_size, rep_dim_s,
                                           rep_dim_w)
        x_test, y_test = populate_vector(test_idx_path, word_rep, test_rep, test_size, x_size, y_size, rep_dim_s,
                                         rep_dim_w)
        x_val, y_val = populate_vector(val_idx_path, word_rep, val_rep, val_size, x_size, y_size, rep_dim_s, rep_dim_w)
    elif task is Tests.WORD_ORDER:
        y_size = 2
        x_size = rep_dim_s + rep_dim_w
        train_size = 300000
        test_size = 50000
        val_size = 50000

        x_train, y_train = populate_vector(train_idx_path, word_rep, train_rep, train_size, x_size, y_size, rep_dim_s,
                                           rep_dim_w)
        x_test, y_test = populate_vector(test_idx_path, word_rep, test_rep, test_size, x_size, y_size, rep_dim_s,
                                         rep_dim_w)
        x_val, y_val = populate_vector(val_idx_path, word_rep, val_rep, val_size, x_size, y_size, rep_dim_s, rep_dim_w)
    elif task is Tests.NEXT_WORD_PREDICTION:
        y_size = 1000
        x_size = rep_dim_s + rep_dim_w
        train_size = 150000
        test_size = 25000
        val_size = 25000

        x_train, y_train = populate_vector_representations(train_idx_path, word_rep, train_rep, train_size, x_size,
                                                           y_size, rep_dim_s, rep_dim_w)
        x_test, y_test = populate_vector_representations(test_idx_path, word_rep, test_rep, test_size, x_size, y_size,
                                                         rep_dim_s, rep_dim_w)
        x_val, y_val = populate_vector_representations(val_idx_path, word_rep, val_rep, val_size, x_size, y_size,
                                                       rep_dim_s, rep_dim_w)
    else:
        return None
    return x_train, y_train, x_test, y_test, x_val, y_val


def populate_vector(train_idx_path, word_rep, sen_rep, data_size, example_size, target_size, rep_size_s, rep_size_w):
    x = np.zeros((data_size, example_size))
    y = np.zeros((data_size, target_size))
    with open(train_idx_path) as fid:
        lines = fid.readlines()
        for i in range(len(lines)):
            vals = lines[i].split()
            target = int(vals[0])
            sen_target = int(vals[1])
            y[i][target] = 1
            x[i][0:rep_size_s] = sen_rep[sen_target]
            if len(vals) > 2:
                word_target = int(vals[2])
                w_size = rep_size_w
                if len(vals) > 3:
                    w_size = rep_size_w / 2
                x[i][rep_size_s:rep_size_s + w_size] = word_rep[word_target]
            if len(vals) > 3:
                word_target = int(vals[3])
                w_size = rep_size_w / 2
                x[i][rep_size_s + w_size:rep_size_s + rep_size_w] = word_rep[word_target]
    return x, y


def populate_vector_representations(train_idx_path, word_rep, sen_rep, data_size, example_size, target_size, rep_size_s,
                                    rep_size_w):
    x = np.zeros((data_size, example_size))
    y = np.zeros((data_size, target_size))
    with open(train_idx_path) as fid:
        lines = fid.readlines()
        for i in range(len(lines)):
            vals = lines[i].split()
            target = int(vals[0])
            sen_target = int(vals[1])
            y[i] = word_rep[target]
            x[i][0:rep_size_s] = sen_rep[sen_target]
            word_target = int(vals[2])
            x[i][rep_size_s:rep_size_s + rep_size_w] = word_rep[word_target]
    return x, y


def representation2bin(path, output_path, data_size=25000, rep_size=100):
    sen_rep = np.zeros((data_size, rep_size))
    with open(path) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            vals = lines[i].split()
            sen_rep[i] = [float(x) for x in vals[0:rep_size]]
    f.close()
    np.save(output_path, sen_rep, True)
