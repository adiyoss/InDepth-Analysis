import numpy as np
from matplotlib import pyplot as plt
import pylab as P


# computes precision@k for the multi class case
def precision_at_k(y, y_hat, k=5):
    count = 0.0
    for item in y:
        t = np.argmax(item)
        for i in range(k):
            if t == y_hat[i]:
                count += 1.0
                break
    return float(count) / len(y)


def plot_accuracy_vs_distance(y, y_hat, order_file_path):
    INFINITY = 1000
    bins = [[1, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 14],
            [15, 16], [17, 18], [19, 20], [21, 22], [23, INFINITY]]

    fid = open(order_file_path)
    lines = fid.readlines()
    fid.close()

    acc = dict()
    durations = list()
    for i in range(len(y)):
        cur_acc = y_hat[i] == np.argmax(y[i])
        order = lines[i].split()
        duration = np.abs(int(order[1]) - int(order[0]))
        if cur_acc:
            durations.append(duration)

        # if duration in acc:
        #     acc[duration].append(cur_acc)
        # else:
        #     acc[duration] = list()
        #     acc[duration].append(cur_acc)
        for b in bins:
            if b[0] <= duration <= b[1]:
                if b[0] in acc:
                    acc[b[0]].append(cur_acc)
                else:
                    acc[b[0]] = list()
                    acc[b[0]].append(cur_acc)
    plot_acc = dict()
    for val in acc:
        plot_acc[val] = (float(np.sum(acc[val])) / len(acc[val]))
        print("B: %s, P: %s" % (val, len(acc[val])))

    # plt.title("Accuracy vs. Word Distance")
    # plt.ylabel('Accuracy')
    # plt.xlabel('Word Distance')

    np.save("w2v_500", plot_acc)
    # plt.bar(range(len(plot_acc)), plot_acc.values(), align='center')
    # plt.xticks(range(len(plot_acc)), plot_acc.keys())
    # plt.show()


def plot_accuracy_vs_word_position(path_indices, path_test):
    miss_class_idx = np.loadtxt(path_indices)
    test_data = dict()
    with open(path_test) as fid:
        lines = fid.readlines()
        for i in range(len(lines)):
            vals = lines[i].split()
            test_data[i] = [vals]
    fid.close()

    miss_class_data = dict()
    for id in range(len(miss_class_idx)):
        miss_class_data[id] = test_data[int(miss_class_idx[id])]


def read_dictionary(path):
    d_i2w = dict()
    d_w2i = dict()
    with open(path) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            vals = line.split()
            d_i2w[i] = vals[0]
            d_w2i[vals[0]] = i
    fid.close()
    return d_i2w, d_w2i


dict_skip_i2w = None
dict_skip_w2i = None
dict_enc_dec_i2w = None
dict_enc_dec_w2i = None


def enc_dec_2_skip_thoughts_dict(id, dict_enc_dec_path="data/orig/dictionary.txt",
                                 dict_skip_path="data/skip_thoughts/raw/dictionary.txt"):
    global dict_skip_i2w
    global dict_skip_w2i
    global dict_enc_dec_i2w
    global dict_enc_dec_w2i

    # read skip-thoughts dictionary
    if dict_skip_i2w is None or dict_skip_w2i is None:
        dict_skip_i2w, dict_skip_w2i = read_dictionary(dict_skip_path)
    # read enc-dec dictionary
    if dict_enc_dec_i2w is None or dict_enc_dec_w2i is None:
        dict_enc_dec_i2w, dict_enc_dec_w2i = read_dictionary(dict_enc_dec_path)

    word = dict_enc_dec_i2w[id]
    if word in dict_skip_w2i:
        new_id = dict_skip_w2i[word]
    elif word.lower() in dict_skip_w2i:
        new_id = dict_skip_w2i[word.lower()]
    else:
        new_id = dict_skip_w2i['unknown']
    return new_id


def check_intersection(dict_size=50001):
    count = 0
    for i in range(dict_size):
        id = enc_dec_2_skip_thoughts_dict(i)
        if id == 1877:
            count += 1
    print "Total number: %d" % count
    print "Percentage from dictionary: %.2f" % (float(count) / dict_size)
