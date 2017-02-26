from __future__ import absolute_import
from __future__ import print_function
from keras import callbacks
import numpy as np
import src.model as m
import src.generate_data_old as gd
from src.enum_tests import Tests
from sklearn.metrics import confusion_matrix


batch_size = 100
nb_epoch = 500
save_dir = "models/order.margin.w2v.500.5.model.net"

early_stopping_patience = 5
output_dim = 2
input_dim = 3*500
data_dir = "data/idx/order/"
rep_dir = "data/w2v_500_win5/"
task = Tests.WORD_ORDER

# loading the data
x_train, y_train, x_test, y_test, x_val, y_val = gd.build_data(rep_dir + "word_repr.npy",
                                                               data_dir + "train.txt",
                                                               rep_dir + "train.rep.npy",
                                                               data_dir + "test.txt", rep_dir + "test.rep.npy",
                                                               data_dir + "val.txt",
                                                               rep_dir + "val.rep.npy", task=task)
print("\n=============================")
print("Train data shape: ", x_train.shape)
print("Test data shape: ", x_test.shape)
print("Validation data shape: ", x_val.shape)
print("")
precent = np.sum(y_train[:, 0]) / float(len(y_train))
print("Train labels balance: 0: %.2f, 1: %.2f" % (precent, (1 - precent)))
precent = np.sum(y_test[:, 0]) / float(len(y_test))
print("Test labels balance: 0: %.2f, 1: %.2f" % (precent, (1 - precent)))
precent = np.sum(y_val[:, 0]) / float(len(y_val))
print("Validation labels balance: 0: %.2f, 1: %.2f" % (precent, (1 - precent)))
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

y_hat = model.predict_classes(x_test, batch_size)
print(confusion_matrix(np.argmax(y_test, axis=1), y_hat))