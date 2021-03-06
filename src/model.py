from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad, Adam
from theano.compile import mode

__author__ = 'yossiadi'

theano_mode = mode.FAST_RUN


def build_model(input_dim=2000, output_dim=2, drop_out=0.9):
    # CLASSIFICATION
    h_layer_size = input_dim
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=h_layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(input_dim=h_layer_size, output_dim=output_dim))
    model.add(Activation('softmax'))

    # optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    optimizer = Adagrad()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

    # # REGRESSION
    # h_layer_size = input_dim
    # model = Sequential()
    # model.add(Dense(input_dim=input_dim, output_dim=h_layer_size))
    # model.add(Activation('relu'))
    # model.add(Dropout(drop_out))
    # model.add(Dense(input_dim=h_layer_size, output_dim=output_dim))
    #
    # optimizer = Adagrad()
    # model.compile(loss='mse', optimizer=optimizer, theano_mode=theano_mode)
    #
    # return model
