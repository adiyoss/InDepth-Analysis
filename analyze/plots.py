import matplotlib
import numpy as np
import operator
from matplotlib import pyplot as plt
from scipy import stats


def check_norms(repr_path, labels):
    reprs = np.load(repr_path)
    norms = dict()
    # plot_norm = list()
    for i, r in enumerate(reprs):
        n = np.linalg.norm(r)
        l = labels[i]
        if l not in norms:
            norms[l] = list()
        norms[l].append(n)
        # plot_norm.append(n)

    plot_norm = list()
    e = list()
    for n in norms:
        avg = np.average(norms[n])
        std = np.std(norms[n])
        plot_norm.append(avg)
        e.append(std)
        print "Length = %d, average norm = %.3f" % (n, avg)
        print "Length = %d, std norm = %.3f" % (n, std)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}
    matplotlib.rc('font', **font)

    # np.savetxt('w2v_300.txt', plot_norm)
    # plt.errorbar(range(len(plot_norm)), plot_norm, e, linestyle='None', marker='^')
    plt.scatter(range(len(plot_norm)), plot_norm)
    # plt.bar(range(len(plot_norm)), plot_norm, align='center', color='b', width=0.8)
    plt.ylim(0.3, 0.6)
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
        if l > 32:
            l = 32
        if l not in lens:
            lens[l] = list()
        lens[l].append(val)

    c = 0
    acc_plot = list()
    for l in lens:
        acc_plot.append((1 - np.average(lens[l])))
        c += len(lens[l])
        print "Length = %d, accuracy = %.3f, num = %.3f" % (l, (1 - np.average(lens[l])), len(lens[l]))
    print 'Total: ', c

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}
    matplotlib.rc('font', **font)

    np.savetxt('random_acc_skip.txt', np.asarray(acc_plot))
    y_point = np.arange(len(acc_plot))
    m, b = np.polyfit(np.asarray(acc_plot), np.asarray(y_point), 1)
    x = np.random.rand(1000)
    y = list()
    for i in x:
        y.append(i * m + b)
    print(m)

    plt.subplot(211)
    plt.bar(range(len(acc_plot)), acc_plot, align='center', color='b', width=0.8)
    plt.xlabel("Sentence length")
    plt.ylabel("Random word accuracy")
    plt.ylim(0.5, 1)
    plt.xticks(range(len(lens)), [int(l) for l in lens], fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(212)
    plt.plot(x, y)

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


def central_limit_check(path, lengths, w_p):
    data_tes = np.load(path)
    words = np.load(w_p)
    mue = np.mean(words, axis=0)
    sigma = np.std(words, axis=0)

    count = 0.0
    for x, l in zip(data_tes, lengths):
        v = x - mue * np.sqrt(l)
        c_t = 0
        for i, fit in enumerate(v):
            if fit > (3 * sigma[i]) or fit < (-3 * sigma[i]):
                c_t += 1
        if c_t > len(v)*0.03:
            count += 1
    print(count/len(data_tes))*100

    # c = 0.0
    # for i in p:
    #     if i > 0.0001:
    #         c += 1
    # print(c)

    # x_hat = np.mean(data, axis=0)
    # mue = np.mean(data)
    # std = np.std(data, axis=0)
    # sqrt = np.sqrt(len(data))
    # data_n = (x_hat - mue) / sqrt
    # count = 0.0
    #
    # for i, r in enumerate(data_n):
    #     if r > x_hat[i] + 2 * std[i] or r < x_hat[i] - 2 * std[i]:
    #         count += 1
    # print(count/len(data_n))


idx_path = "/Users/yossiadi/Projects/representation_analysis/data/orig/test.txt"
repr_path = "/Users/yossiadi/Projects/representation_analysis/data/w2v_300_win5/test.rep.npy"
words_p = "/Users/yossiadi/Projects/representation_analysis/data/w2v_300_win5/word_repr.npy"
# rand_path = "../docs/rand_acc/skip/random_skip.txt"
# rand_path_idx = "../data/idx/random_word/test.txt"

labels = get_lengths(idx_path)
central_limit_check(repr_path, labels, words_p)

# labels = get_lengths(idx_path)
# check_norms(repr_path, labels)
#
# # sen_rep_plot(np.load(repr_path), labels)
#
# # accuracy_vs_word_location(rand_path, rand_path_idx, idx_path, labels)
#
# # accuracy_vs_sen_len(rand_path, labels)
#
# # check_norms(repr_path, labels)
