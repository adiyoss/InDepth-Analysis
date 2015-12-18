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
    fid = open(order_file_path)
    lines = fid.readlines()
    fid.close()

    acc = dict()
    for i in range(len(y)):
        cur_acc = y_hat[i] == y[i][0]
        order = lines[i].split()
        duration = np.abs(int(order[1]) - int(order[0]))
        if duration in acc:
            acc[duration].append(cur_acc)
        else:
            acc[duration] = list()
            acc[duration].append(cur_acc)

    plot_acc = list()
    for val in acc:
        plot_acc.append(float(np.sum(acc[val])) / len(acc[val]))

    plt.title("Accuracy vs. Word Distance")
    plt.ylabel('Accuracy')
    plt.xlabel('Word Distance')
    plt.plot(plot_acc)
    plt.show()
