from __future__ import absolute_import
from __future__ import print_function
from keras import callbacks
import numpy as np
import src.model as m
import src.generate_data as gd

np.random.seed(1237)  # for reproducibility

__author__ = 'yossiadi'

batch_size = 100
nb_epoch = 100
save_dir = "models/random.word.model.net"
early_stopping_patience = 5
output_dim = 2
input_dim = 2000
hidden_size = 2000

# loading the data
x_train, y_train = gd.load_data("data/processed/random_word/train.txt", x_size=input_dim, y_size=output_dim)
x_test, y_test = gd.load_data("data/processed/random_word/test.txt", x_size=input_dim, y_size=output_dim)
x_val, y_val = gd.load_data("data/processed/random_word/val.txt", x_size=input_dim, y_size=output_dim)

print("\n=============================")
print("Train data shape: ", x_train.shape)
print("Test data shape: ", x_test.shape)
print("")
precent = np.sum(y_train[:, 0])/float(len(y_train))
print("Train labels balance: 0: %.2f, 1: %.2f" % (precent, (1-precent)))
precent = np.sum(y_test[:, 0])/float(len(y_test))
print("Test labels balance: 0: %.2f, 1: %.2f" % (precent, (1-precent)))
print("=============================\n")

check_pointer = callbacks.ModelCheckpoint(filepath=save_dir, verbose=1, save_best_only=True)
early_stop = callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1)

model = m.build_model(input_dim=input_dim, output_dim=output_dim)
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
          validation_data=(x_val, y_val), shuffle=True, callbacks=[check_pointer, early_stop])

print("\n============ TEST =============\n")
score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
