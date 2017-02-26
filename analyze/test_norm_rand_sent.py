import random

import numpy as np
from matplotlib import pyplot as plt

num_of_sent = 50000
sent_len = 300
sent = np.zeros((num_of_sent, sent_len))
lengths = np.zeros(num_of_sent)

for i in xrange(num_of_sent):
    l = np.random.randint(5, 40)
    for j in xrange(l):
        sent[i] += (2 * np.random.normal(0, 1, sent_len)) - 1
    sent[i] /= l
    lengths[i] = l

norms = dict()
plot_norm = list()
for i, r in enumerate(sent):
    n = np.linalg.norm(r)
    l = lengths[i]
    if l not in norms:
        norms[l] = list()
    norms[l].append(n)

for n in norms:
    avg = np.average(norms[n])
    plot_norm.append(avg)
    print "Length = %d, average norm = %.3f" % (n, avg)

# plt.bar(range(len(plot_norm)), plot_norm, align='center', color='b', width=0.8)
plt.scatter(range(len(plot_norm)), plot_norm)
# plt.ylim(8.5, 9)
plt.xlabel("Lengths")
plt.ylabel("Norm")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

