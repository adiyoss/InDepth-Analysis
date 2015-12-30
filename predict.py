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
test_path = "data/processed/sen_len/val.txt"
model_path = "models/sen.len.model.net"
# order_path = "data/processed/order/order.txt"

# loading the data
x_test, y_test = gd.load_data(test_path, x_size=1000, y_size=6)

# build the model
model = m.build_model(input_dim=1000, output_dim=6)
model.load_weights(model_path)
# get predictions
y_hat = model.predict_classes(x_test, batch_size=batch_size)

# utils.plot_accuracy_vs_distance(y_test, y_hat, order_path)

print("")
print("Test set: %s" % test_path)
print("Accuracy on the test set: %s" % (float(np.sum(y_hat == np.argmax(y_test, axis=1)))/len(y_hat)))