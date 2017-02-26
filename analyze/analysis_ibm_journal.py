import numpy as np
from matplotlib import pyplot as plt
import operator


def acc_w_freq(words_data, path_w2v):
    w2v_acc = np.loadtxt(path_w2v)
    words = list()
    with open(words_data) as f:
        for l in f.readlines():
            vals = l.split()
            words.append(vals[2])

    word_acc = dict()
    freq = dict()
    for i in range(len(words)):
        if words[i] not in word_acc:
            word_acc[words[i]] = list()
            freq[words[i]] = 0
        word_acc[words[i]].append(w2v_acc[i])
        freq[words[i]] += 1
    freq_x = sorted(freq.items(), key=operator.itemgetter(1))

    data = list()
    types = list()
    prev_val = 1
    i = 0
    while i < len(freq_x):
        num = 0
        n_items = 0
        cond = True
        while i < len(freq_x) and cond:
            cond = False
            num += len(word_acc[freq_x[i][0]]) - np.sum(word_acc[freq_x[i][0]])
            n_items += len(word_acc[freq_x[i][0]])
            i += 1
            if i < len(freq_x) and (freq_x[i][1] == prev_val or n_items < 1000):
                if freq_x[i][1] != prev_val:
                    prev_val = freq_x[i][1]
                cond = True
        types.append(prev_val)
        prev_val = freq_x[i][1] if (i < len(freq_x)) else 1
        data.append(num / n_items)
    return data, types


def save_files_and_plot_figure():
    words_data = '/Users/yossiadi/Projects/representation_analysis/data/idx/random_word/test.txt'
    path_d = ['/Users/yossiadi/Projects/representation_analysis/docs/rand_acc/enc_dec/random_1000_acc_enc_dec.txt',
              '/Users/yossiadi/Projects/representation_analysis/docs/rand_acc/enc_dec/random_750_acc_enc_dec.txt',
              '/Users/yossiadi/Projects/representation_analysis/docs/rand_acc/w2v/random_300_acc_w2v.txt']
    data = list()
    for p in path_d:
        d, t = acc_w_freq(words_data, p)
        data.append(d)

    types = ['1', '2', '3', '4', '5', '6', '7-8', '9-14', '15-63', '64-172', '173+']
    w = 0.2
    plt.xticks(np.arange(len(data[0])), types, fontsize=14)
    plt.ylim(0.2, 1.01)
    plt.bar(np.arange(len(data[0])) + w, data[0], width=w, color='#9900cc', align='center', label='ED-1000')
    plt.bar(np.arange(len(data[0])), data[1], width=w, color='#009999', align='center', label='ED-750')
    plt.bar(np.arange(len(data[0])) - w, data[2], width=w, color='#993333', align='center', label='w2v-300')
    plt.xlabel('Word Frequency')
    plt.ylabel('Content Prediction Accuracy')
    plt.legend(loc=2, borderaxespad=0., prop={'size': 14})
    plt.show()

    # np.savetxt('/Users/yossiadi/Desktop/x.values.txt', types[0])
    # np.savetxt('/Users/yossiadi/Desktop/ed.1000.txt', data[0])
    # np.savetxt('/Users/yossiadi/Desktop/ed.750.txt', data[1])
    # np.savetxt('/Users/yossiadi/Desktop/w2v.300.txt', data[2])


def analyze_generalization_db(path='/Users/yossiadi/Projects/representation_analysis/data/random_k/w2v_300/test.txt'):
    data = np.loadtxt(path)
    words_pos = dict()
    words_neg = dict()
    for item in data:
        # 0, 1, 10, 50, 100, 500
        pos_wrd = item[0]
        neg_wrd = item[5]
        if pos_wrd not in words_pos:
            words_pos[pos_wrd] = 0
            if pos_wrd not in words_neg:
                words_neg[pos_wrd] = 0
        if neg_wrd not in words_neg:
            words_neg[neg_wrd] = 0
            if neg_wrd not in words_pos:
                words_pos[neg_wrd] = 0
        words_pos[pos_wrd] += 1
        words_neg[neg_wrd] += 1

    c = 0
    t = 0
    for k in words_neg:
        if (words_neg[k] < words_pos[k] and words_neg[k] == 0) or (
                    words_pos[k] < words_neg[k] and words_pos[k] == 0):
            print 'n_pos = %s, n_neg = %s, k = %s' % (words_pos[k], words_neg[k], k)
            t += words_neg[k] + words_pos[k]
            c += 1
    print '\ntotal = %s' % c
    print t


def word_dist_vs_order_acc(path_acc='/Users/yossiadi/Projects/representation_analysis/docs/ibm_journal/word_dist_vs_order_acc/1000.enc.dec.order.txt'):
    path_order = '/Users/yossiadi/Projects/representation_analysis/data/idx/order/test.txt.order.txt'
    distances = list()
    INFINITY = 1000
    bins = [[1, 3], [4, 4], [5, 5], [6, 6], [7, 8], [9, 10], [11, 13], [14, 17], [18, INFINITY]]

    with open(path_order) as f:
        for line in f.readlines():
            vals = line.split()
            distances.append(np.abs(int(vals[0]) - int(vals[1])))
    acc_data = np.loadtxt(path_acc)

    x = np.zeros(len(bins))
    s = np.zeros(len(bins))
    for i, d in enumerate(acc_data):
        for j, b in enumerate(bins):
            if b[0] <= int(distances[i]) <= b[1]:
                x[j] += float(d)
                s[j] += 1
                break

    plot_data = list()
    for i, w in enumerate(s):
        plot_data.append(1 - float(x[i]) / w)
    np.savetxt('1000.enc.dec.order.txt', plot_data)
    plt.plot(plot_data)
    plt.show()


word_dist_vs_order_acc()
# save_files_and_plot_figure()
# analyze_generalization_db()


