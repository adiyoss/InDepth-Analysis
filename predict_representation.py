from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import src.model as m
import src.generate_data as gd
import src.utils as utils
from src.enum_tests import Tests
from scipy import spatial

__author__ = 'yossiadi'

np.random.seed(1237)  # for reproducibility

# # params
# batch_size = 100
# output_dim = 1000
# input_dim = 2000
# # test_path = "data/processed/order/test.txt"
# model_path = "models/encoder_decoder/repr.pred.model.net"
# data_dir = "data/idx/order_repr/"
# rep_dir = "data/orig/"
# word_rep_path = rep_dir + "word_repr.npy"
# task = Tests.NEXT_WORD_PREDICTION
#
# x_train, y_train, x_test, y_test, x_val, y_val = gd.build_data(rep_dir + "word_repr.npy",
#                                                                data_dir + "train.txt",
#                                                                rep_dir + "/train.rep.npy",
#                                                                data_dir + "test.txt", rep_dir + "test.rep.npy",
#                                                                data_dir + "/val.txt",
#                                                                rep_dir + "val.rep.npy", task=task)
# # build the model
# model = m.build_model(input_dim=input_dim, output_dim=output_dim)
# model.load_weights(model_path)
# # get predictions
# y_hat = model.predict(x_test, batch_size=batch_size)
#
# # load word representations
# word_rep = np.load(word_rep_path)
# tree = spatial.KDTree(word_rep)
# y_hat_class = np.zeros(len(y_hat))
# i = 0
# for v in y_hat:
#     y_hat_class[i] = tree.query(v)[1]
#     i += 1
#     print(i)
# print("Accuracy on the test set: %s" % (float(np.sum(y_hat == y_hat_class)) / len(y_hat)))
#
# # l = y_hat != np.argmax(y_test, axis=1)
# # y = [i for i, x in enumerate(l) if x]
# # np.savetxt("1.txt", y)

from matplotlib import pyplot as plt
import matplotlib

font = {'family': 'normal',
        'weight': 'bold',
        'size': 20}
matplotlib.rc('font', **font)

# BAR PLOT
s2s = np.load("s2s.npy")
w2v = np.load("w2v.npy")
w2v_300 = np.load("w2v_300.npy")

x = dict()
y = dict()
z = dict()

s2s_x = str(s2s).split(",")
w2v_x = str(w2v).split(",")
w2v_300_x = str(w2v_300).split(",")

for i in range(len(s2s_x)):
    if ":" in s2s_x[i]:
        if "}" not in s2s_x[i]:
            x[i + 1] = float(s2s_x[i].split(":")[1])
            y[i + 1] = float(w2v_x[i].split(":")[1])
            z[i + 1] = float(w2v_300_x[i].split(":")[1])
        else:
            x[i + 1] = float(s2s_x[i].split(":")[1][:-1])
            y[i + 1] = float(w2v_x[i].split(":")[1][:-1])
            z[i + 1] = float(w2v_300_x[i].split(":")[1][:-1])


t_enc = np.zeros(17)
t_w2v = np.zeros(17)
i = 0
for item in x:
    t_enc[i] = x[item]
    t_w2v[i] = z[item]
    i += 1

np.save("t_enc", t_enc)
np.save("t_w2v", t_w2v)

bins = ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '15', '17', '19', '21', '23']
loc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

plt.subplot(121)
enc_dec = plt.bar(range(len(x)), x.values(), align='center', color='b', width=1, label="Encoder Decoder")
plt.xticks(range(len(x)), x.keys())
plt.xlabel("Word Distance")
plt.ylabel("Accuracy")
plt.ylim(0.6, 1)
plt.xticks(loc, bins, fontsize=15)
plt.legend(handles=[enc_dec], prop={'size': 18})

plt.subplot(122)
bow = plt.bar(range(len(z)), z.values(), align='center', color='g', width=1, label="BOW (average w2v-size 300)")
plt.xticks(range(len(x)), x.keys())
plt.xlabel("Word Distance")
plt.ylabel("Accuracy")
plt.ylim(0.6, 1)
plt.xticks(loc, bins, fontsize=15)
plt.legend(handles=[bow], prop={'size': 18})

plt.show()

# plt.bar(range(len(x)), x.values(), align='center', width=1)
# plt.bar(range(len(z)), z.values(), align='center', color='red', width=1)
# plt.xticks(range(len(x)), x.keys())
# plt.xlabel("Word Distance")
# plt.ylabel("Accuracy")
# plt.ylim(0.6, 1)
# plt.show()
