import argparse
import numpy as np
import operator
from matplotlib import pyplot as plt


def run_stats(idx_path, dict_size):
    fid = open(idx_path)
    x_train = fid.readlines()
    fid.close()

    first_word = dict()
    last_word = dict()
    sen_lens = dict()

    for sen in x_train:
        s = sen.split()
        if s[0] not in first_word:
            first_word[s[0]] = 1
        else:
            first_word[s[0]] += 1
        last = s[len(s) - 1]
        if last == '4':
            last = s[len(s) - 2]
        if last not in last_word:
            last_word[last] = 1
        else:
            last_word[last] += 1

        s_len = len(s)
        INFINITY = 1000
        # bins = [[2, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, INFINITY]] # 6 bins
        bins = [[5, 8], [9, 12], [13, 16], [17, 20], [21, 25], [26, 29], [30, 33], [34, INFINITY]]  # 8 bins
        # bins = [[5, 8], [9, 12], [13, 16], [17, 20], [21, 25], [26, INFINITY]]
        # bins = [[5, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
        #         [35, 37], [38, INFINITY]]  # 12 bins
        for i, b in enumerate(bins):
            if b[0] <= s_len <= b[1]:
                if i in sen_lens:
                    sen_lens[i] += 1
                else:
                    sen_lens[i] = 1

    print "\nNumber of different first word: %d, First word stats: %.3f" % (
        len(first_word), (float(len(first_word)) / dict_size))
    print "Max first word: %s" % (sorted(first_word.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
    print "Number of different first word: %d, Last word stats: %.3f" % (
        len(last_word), (float(len(last_word)) / dict_size))
    print "Max last word: %s" % (sorted(last_word.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
    print "Sentence lengths: ", sen_lens

    first_word_distribution = list()
    last_word_distribution = list()
    for i in first_word:
        first_word_distribution.append(float(first_word[i]) / len(x_train))
    for i in last_word:
        last_word_distribution.append(float(last_word[i]) / len(x_train))

        # plt.title("First Word Distribution")
        # plt.plot(first_word_distribution)
        # plt.show()
        #
        # plt.title("Last Word Distribution")
        # plt.plot(last_word_distribution)
        # plt.show()


def stat_mutual_words(train_path, test_path):
    train = set()
    test = set()

    fid = open(train_path, 'r')
    train_lines = fid.readlines()
    fid.close()

    fid = open(test_path, 'r')
    test_lines = fid.readlines()
    fid.close()

    # get all the unique words in the train set
    for item in train_lines:
        vals = item.split()
        for val in vals:
            train.add(val)

    # get all the unique words in the test set
    for item in test_lines:
        vals = item.split()
        for val in vals:
            test.add(val)

    inter = set.intersection(train, test)
    print("Shared words between train and test: %d" % (len(inter)))
    print("Percentage from test: %.2f" % (float(len(inter)) / len(test)))
    print("Percentage from train: %.2f" % (float(len(inter)) / len(train)))


def stat_mutual_sentences(train_path, test_path):
    train = set()
    test = set()

    fid = open(train_path, 'r')
    train_lines = fid.readlines()
    fid.close()

    fid = open(test_path, 'r')
    test_lines = fid.readlines()
    fid.close()

    # get all the unique words in the train set
    for item in train_lines:
        train.add(item[:-1])

    # get all the unique words in the test set
    for item in test_lines:
        test.add(item[:-1])

    inter = set.intersection(train, test)
    print("Shared words between train and test: %d" % (len(inter)))
    print("Percentage from test: %.5f" % (float(len(inter)) / len(test)))
    print("Percentage from train: %.5f" % (float(len(inter)) / len(train)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Get statistics for the words location in the sentence.")
    parser.add_argument("in_filename", help="The path to the train/test/val file, it should be in index format not"
                                            " exact words")
    parser.add_argument("--dict_size", help="The path to the dictionary", default=50000)
    args = parser.parse_args()
    run_stats(args.in_filename, args.dict_size)

    # stat_mutual_words("../../data/representation/orig/test.txt", "../../data/representation/orig/val.txt")
    # stat_mutual_sentences("../../data/representation/orig/test.txt", "../../data/representation/orig/val.txt")
