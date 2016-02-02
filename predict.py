from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from sklearn.metrics import confusion_matrix

import src.model as m
import src.generate_data as gd
import src.utils as utils
from src.enum_tests import Tests

__author__ = 'yossiadi'

np.random.seed(1237)  # for reproducibility

# params
batch_size = 100
output_dim = 8
input_dim = 1*500
# test_path = "data/processed/order/test.txt"
model_path = "models/enc_dec_500/sen.len.enc.dec.500.model.net"
data_dir = "data/idx/sen_len/"
rep_dir = "data/enc_dec_500/"
# order_path = "data/idx/order/test.txt.order.txt"
task = Tests.SENTENCE_LENGTH


x_train, y_train, x_test, y_test, x_val, y_val = gd.build_data(rep_dir + "word_repr.npy",
                                                               data_dir + "train.txt",
                                                               rep_dir + "train.rep.npy",
                                                               data_dir + "test.txt", rep_dir + "test.rep.npy",
                                                               data_dir + "val.txt",
                                                               rep_dir + "val.rep.npy", task=task)
# build the model
model = m.build_model(input_dim=input_dim, output_dim=output_dim)
model.load_weights(model_path)
# get predictions
y_hat = model.predict_classes(x_test, batch_size=batch_size)

# utils.plot_accuracy_vs_distance(y_test, y_hat, order_path)

print("")
# print("Test set: %s" % test_path)
print("Accuracy on the test set: %s" % (float(np.sum(y_hat == np.argmax(y_test, axis=1))) / len(y_hat)))
print(confusion_matrix(np.argmax(y_test, axis=1), y_hat))

# l = y_hat != np.argmax(y_test, axis=1)
# # y = [i for i, x in enumerate(l) if x]
# np.savetxt("random_300_acc_w2v.txt", l)
