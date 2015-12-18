from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import src.model as m
import src.generate_data as gd
import src.utils as utils

__author__ = 'yossiadi'

np.random.seed(1237)  # for reproducibility

# params
batch_size = 16
nb_epoch = 10
test_path = "data/processed/small_data/complex_order_words/test_s.txt"
model_path = "models/complex.big.big.word.order.model.net"
order_path = "data/processed/small_data/complex_order_words/order.txt"

# loading the data
x_test, y_test = gd.load_data(test_path)

# build the model
model = m.build_model()
model.load_weights(model_path)
# get predictions
y_hat = model.predict_classes(x_test, batch_size=batch_size)

utils.plot_accuracy_vs_distance(y_test, y_hat, order_path)

print("")
print("Test set: %s" % test_path)
print("Accuracy on the test set: %s" % (float(np.sum(y_hat == y_test[:, 1]))/len(y_hat)))

