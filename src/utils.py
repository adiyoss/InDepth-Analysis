import numpy as np
from matplotlib import pyplot as plt


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
    bins = [[2, 4], [5, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, INFINITY]]

    fid = open(order_file_path)
    lines = fid.readlines()
    fid.close()

    acc = dict()
    for i in range(len(y)):
        cur_acc = y_hat[i] == y[i][0]
        order = lines[i].split()
        duration = np.abs(int(order[1]) - int(order[0]))
        for b in bins:
            if b[0] <= duration <= b[1]:
                if b[0] in acc:
                    acc[b[0]].append(cur_acc)
                else:
                    acc[b[0]] = list()
                    acc[b[0]].append(cur_acc)

    plot_acc = list()
    for val in acc:
        plot_acc.append(float(np.sum(acc[val])) / len(acc[val]))
        print("B: %s, P: %s" % (val, len(acc[val])))

    plt.title("Accuracy vs. Word Distance")
    plt.ylabel('Accuracy')
    plt.xlabel('Word Distance')
    plt.plot(plot_acc)
    plt.show()
