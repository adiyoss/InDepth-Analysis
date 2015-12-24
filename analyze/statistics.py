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
        bins = [[2, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, INFINITY]]
        for i, b in enumerate(bins):
            if b[0] <= s_len <= b[1]:
                if i in sen_lens:
                    sen_lens[i] += 1
                else:
                    sen_lens[i] = 1

    print "\nNumber of different first word: %d, First word stats: %.3f" % (len(first_word), (float(len(first_word))/dict_size))
    print "Max first word: %s" % (sorted(first_word.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
    print "Number of different first word: %d, Last word stats: %.3f" % (len(last_word), (float(len(last_word))/dict_size))
    print "Max last word: %s" % (sorted(last_word.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
    print "Sentence lengths: ", sen_lens

    first_word_distribution = list()
    last_word_distribution = list()
    for i in first_word:
        first_word_distribution.append(float(first_word[i])/len(x_train))
    for i in last_word:
        last_word_distribution.append(float(last_word[i])/len(x_train))

    # plt.title("First Word Distribution")
    # plt.plot(first_word_distribution)
    # plt.show()
    #
    # plt.title("Last Word Distribution")
    # plt.plot(last_word_distribution)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Get statistics for the words location in the sentence.")
    parser.add_argument("in_filename", help="The path to the train/test/val file, it should be in index format not"
                                            " exact words")
    parser.add_argument("--dict_size", help="The path to the dictionary", default=50000)
    args = parser.parse_args()

    run_stats(args.in_filename, args.dict_size)
