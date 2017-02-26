import numpy as np
from scipy import stats
import os


def calc_significance(path_1, path_2):
    x1 = np.loadtxt(path_1)
    x2 = np.loadtxt(path_2)

    for i in range(len(x1)):
        if x1[i] == 0.0:
            x1[i] = 1.0
        elif x1[i] == 1.0:
            x1[i] = 0.0

        if x2[i] == 0.0:
            x2[i] = 1.0
        elif x2[i] == 1.0:
            x2[i] = 0.0
    t, p = stats.wilcoxon(x1, x2)
    # t, p = stats.ttest_rel(x1, x2)
    # t, p = stats.ttest_ind(x1, x2, equal_var=False)
    return t, p


dirs = ['content/', 'len/', 'order/']
types = ['content', 'len', 'order']
dims = [100, 300, 500, 750, 1000]
ed = 'ed'
w2v = 'w2v'
for j, dir in enumerate(dirs):
    print('\n=======' + str(types[j]) + '=======')
    for i in range(len(dims)):
        ed_item = str(dims[i]) + '.' + ed + "." + str(types[j]) + '.txt'
        w2v_item = str(dims[i]) + '.' + w2v + "." + str(types[j]) + '.txt'
        t, p = calc_significance(dir + ed_item, dir + w2v_item)
        print('\n=======' + str(dims[i]) + '=======')
        print("T-test value: ", t)
        print("p-value: ", p)
        print("\n")

for j, dir in enumerate(dirs):
    print('\n=======' + str(types[j]) + '=======')
    for i in range(len(dims) - 1):
        ed_item_1 = str(dims[i]) + '.' + ed + "." + str(types[j]) + '.txt'
        ed_item_2 = str(dims[i + 1]) + '.' + ed + "." + str(types[j]) + '.txt'
        t, p = calc_significance(dir + ed_item_1, dir + ed_item_2)
        print('\n=======' + str(dims[i]) + '-' + str(dims[i + 1]) + '=======')
        print('======= ED =======')
        print("T-test value: ", t)
        print("p-value: ", p)
        print("\n")

        w2v_item_1 = str(dims[i]) + '.' + w2v + "." + str(types[j]) + '.txt'
        w2v_item_2 = str(dims[i + 1]) + '.' + w2v + "." + str(types[j]) + '.txt'
        t, p = calc_significance(dir + w2v_item_1, dir + w2v_item_2)
        print('======= CBOW =======')
        print("T-test value: ", t)
        print("p-value: ", p)
        print("\n")