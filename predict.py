from __future__ import absolute_import
from __future__ import print_function

import argparse

import numpy as np
from sklearn.metrics import confusion_matrix

import src.model as m
import src.generate_data as gd
from src import utils
from src.enum_tests import Tests

__author__ = 'yossiadi'


def predict(_model_path, test_dir, _rep_dir, input_size, sent_rep_size, type, output_size=2, _batch_size=100,
            seed=1237):
    np.random.seed(seed)  # for reproducibility

    # params
    batch_size = _batch_size
    output_dim = output_size
    rep_dim = sent_rep_size  # sentence_rep_size
    input_dim = input_size
    rep_dim_w = input_dim - rep_dim
    # test_path = "data/processed/order/test.txt"
    model_path = _model_path
    data_dir = test_dir
    rep_dir = _rep_dir
    order_path = "data/idx/order/test.txt.order.txt"
    task = type
    x_train, y_train, x_test, y_test, x_val, y_val = gd.build_data(rep_dir + "word_repr.npy",
                                                                   data_dir + "train.txt",
                                                                   rep_dir + "train.rep.npy",
                                                                   data_dir + "test.txt", rep_dir + "test.rep.npy",
                                                                   data_dir + "val.txt",
                                                                   rep_dir + "val.rep.npy", rep_dim_w, rep_dim,
                                                                   task=task)
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
    # y = [i for i, x in enumerate(l) if x]
    # np.savetxt('1000.ed.content.txt', l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Analyzing decoder prediction vs. random word test.")
    parser.add_argument("model_path", help="The path to save the models")
    parser.add_argument("tests_dir", help="The path tests dir")
    parser.add_argument("rep_dir", help="The path to the representations dir")
    parser.add_argument("rep_dim_sent", help="The sentence representation size")
    parser.add_argument("rep_dim_word", help="The word representation size")
    parser.add_argument("type", help="the type of test: FIRST_WORD = 1, LAST_WORD = 2, RANDOM_WORD = 3, "
                                     "SENTENCE_LENGTH = 4, WORD_ORDER = 5")
    args = parser.parse_args()

    _type = int(args.type)
    input_size = 0
    output_size = 2
    if (_type is 1) or (_type is 2) or (_type is 3):
        input_size = int(args.rep_dim_sent) + int(args.rep_dim_word)
    elif _type is 5:
        input_size = int(args.rep_dim_sent) + 2 * int(args.rep_dim_word)
    elif _type is 4:
        input_size = int(args.rep_dim_sent)
        output_size = 8
    else:
        print("No such test type")

    if input_size != 0:
        predict(args.model_path, args.tests_dir, args.rep_dir, input_size, int(args.rep_dim_sent), int(args.type),
                output_size=output_size)

        # "models/enc_dec_750/random.word.enc.dec.750.model.net"
        # "data/idx/random_word/"
        # "data/enc_dec_750/"
        # Tests.RANDOM_WORD
