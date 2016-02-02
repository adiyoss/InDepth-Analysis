import matplotlib
import numpy as np
import operator
from matplotlib import pyplot as plt


def check_norms(repr_path, labels):
    reprs = np.load(repr_path)
    norms = dict()
    plot_norm = list()
    for i, r in enumerate(reprs):
        n = np.linalg.norm(r)
        l = labels[i]
        if l not in norms:
            norms[l] = list()
        norms[l].append(n)
        plot_norm.append(n)

    # # plot_norm = list()
    # for n in norms:
    #     avg = np.average(norms[n])
    #     # plot_norm.append(avg)
    #     print "Length = %d, average norm = %.3f" % (n, avg)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}
    matplotlib.rc('font', **font)

    plt.scatter(labels, plot_norm)
    # plt.bar(range(len(plot_norm)), plot_norm, align='center', color='b', width=0.8)
    # plt.ylim(0, 1)
    plt.xlabel("Lengths")
    plt.ylabel("Norm")
    # plt.xticks(range(len(plot_norm)), [int(l)+5 for l in range(len(plot_norm))], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def get_lengths(path):
    with open(path) as fid:
        lines = fid.readlines()
        labels = np.zeros(len(lines))
        for i, line in enumerate(lines):
            vals = line.split()
            labels[i] = len(vals)
    fid.close()

    return labels


def accuracy_vs_sen_len(random_acc_path, labels):
    lens = dict()
    rand_acc = list()

    with open(random_acc_path) as fid:
        lines = fid.readlines()
        for line in lines:
            rand_acc.append(float(line[:-1]))
    fid.close()

    for i, val in enumerate(rand_acc):
        l = labels[(i % 25000)]
        if l not in lens:
            lens[l] = list()
        lens[l].append(val)

    acc_plot = list()
    for l in lens:
        acc_plot.append((1 - np.average(lens[l])))
        print "Length = %d, accuracy = %.3f" % (l, (1 - np.average(lens[l])))

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}
    matplotlib.rc('font', **font)

    plt.bar(range(len(acc_plot)), acc_plot, align='center', color='b', width=0.8)
    plt.xlabel("Sentence length")
    plt.ylabel("Random word accuracy")
    plt.ylim(0.6, 1)
    plt.xticks(range(len(lens)), [int(l) for l in lens], fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()


def accuracy_vs_word_location(random_acc_path, random_path, idx_path, lens):
    rand_acc = list()
    with open(random_acc_path) as fid:
        lines = fid.readlines()
        for line in lines:
            rand_acc.append(float(line[:-1]))
    fid.close()

    data_idx = list()
    with open(idx_path) as fid:
        lines = fid.readlines()
        for line in lines:
            vals = line.split()
            row = list()
            for val in vals:
                row.append(val)
            data_idx.append(row)
    fid.close()

    rand_words = list()
    with open(random_path) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            if i >= 25000:
                break
            vals = line.split()
            for j, w in enumerate(data_idx[i]):
                if int(vals[2]) == (int(w) - 1):
                    rand_words.append(j)
                    break
    fid.close()

    d = list()
    d.append(list())
    d.append(list())
    d.append(list())
    d.append(list())
    d.append(list())

    for i, item in enumerate(rand_words):
        if item < (1 * lens[i]) / 5:
            d[0].append(rand_acc[i])
        elif (1 * lens[i]) / 5 <= item < (2 * lens[i]) / 5:
            d[1].append(rand_acc[i])
        elif (2 * lens[i]) / 5 <= item < (3 * lens[i]) / 5:
            d[2].append(rand_acc[i])
        elif (3 * lens[i]) / 5 <= item < (4 * lens[i]) / 5:
            d[3].append(rand_acc[i])
        else:
            d[4].append(rand_acc[i])

    fig = list()
    fig.append(1 - np.average(d[0]))
    fig.append(1 - np.average(d[1]))
    fig.append(1 - np.average(d[2]))
    fig.append(1 - np.average(d[3]))
    fig.append(1 - np.average(d[4]))

    print([np.sum(l) for l in d])
    print(fig)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}
    matplotlib.rc('font', **font)

    x_ticks = ["1st Part", "2nd Part", "3rd Part", "4th Part", "5th Part"]
    plt.bar(range(len(x_ticks)), fig, align='center', color='b', width=0.8)
    plt.ylim(0.5, 0.9)
    plt.xlabel("Part of sentence")
    plt.ylabel("Random word accuracy")
    plt.xticks(range(len(x_ticks)), [l for l in x_ticks], fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def sen_rep_plot(sen_rep, lens):
    data_idx = dict()
    repr_size = len(sen_rep[0])

    for i, l in enumerate(lens):
        data_idx[i] = int(l)

    var = np.argsort(np.var(sen_rep, axis=0))

    sorted_x = sorted(data_idx.items(), key=operator.itemgetter(1))
    plot_arr = np.zeros((len(sorted_x), repr_size))

    for i, k in enumerate(sorted_x):
        for j in range(len(var)):
            plot_arr[i][j] = np.abs(sen_rep[k[0]][var[j]])

    im = plt.imshow(plot_arr, interpolation='nearest', aspect='auto')
    plt.colorbar(im)
    plt.show()

idx_path = "/Users/yossiadi/Projects/representation_analysis/data/orig/test.txt"
repr_path = "/Users/yossiadi/Projects/representation_analysis/data/w2v_1000_win5/test.rep.npy"
# rand_path = "../docs/rand_acc/enc_dec/random_100_acc_enc_dec.txt"
# rand_path_idx = "../data/idx/random_word/test.txt"

labels = get_lengths(idx_path)
check_norms(repr_path, labels)

# sen_rep_plot(np.load(repr_path), labels)

# accuracy_vs_word_location(rand_path, rand_path_idx, idx_path, labels)

# accuracy_vs_sen_len(rand_path, labels)

# check_norms(repr_path, labels)
