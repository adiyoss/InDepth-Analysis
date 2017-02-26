import argparse
from src.utils import enc_dec_2_skip_thoughts_dict
import os
import numpy as np


def main(tests_path, output_dir):
    tests = ['first_word/', 'last_word/', 'random_word/', 'order/', 'sen_len/']
    files = ['train.txt', 'test.txt', 'val.txt']

    for t in tests:
        print '\n===> create data for %s' % t
        for f in files:
            path = tests_path + t + f
            with open(path) as fid:
                print '===> convert %s' % f
                if not os.path.exists(output_dir + t):
                    os.mkdir(output_dir + t)
                out_fid = open(output_dir + t + f, 'w')
                lines = fid.readlines()
                for line in lines:
                    vals = line.split()
                    if len(vals) == 3:
                        new_id = enc_dec_2_skip_thoughts_dict(id=int(vals[2]),
                                                              dict_enc_dec_path="../data/orig/dictionary.txt",
                                                              dict_skip_path="../data/skip_thoughts/raw/dictionary.txt")
                        out_fid.write(str(vals[0]) + ' ' + str(vals[1]) + ' ' + str(new_id) + '\n')
                    elif len(vals) == 4:
                        new_id_1 = enc_dec_2_skip_thoughts_dict(id=int(vals[2]),
                                                                dict_enc_dec_path="../data/orig/dictionary.txt",
                                                                dict_skip_path="../data/skip_thoughts/raw/dictionary.txt")
                        new_id_2 = enc_dec_2_skip_thoughts_dict(id=int(vals[3]),
                                                                dict_enc_dec_path="../data/orig/dictionary.txt",
                                                                dict_skip_path="../data/skip_thoughts/raw/dictionary.txt")
                        out_fid.write(
                                str(vals[0]) + ' ' + str(vals[1]) + ' ' + str(new_id_1) + ' ' + str(new_id_2) + '\n')
                    else:
                        out_fid.write(str(vals[0]) + ' ' + str(vals[1]) + '\n')
                out_fid.close()


def merge_word_repr(word_rep_1_path, word_rep_2_path, output_path):
    word_rep_1 = np.load(word_rep_1_path)
    word_rep_2 = np.load(word_rep_2_path)
    if len(word_rep_1) != len(word_rep_2):
        print 'Error: table size mismatch'
        return None

    # concatenate the representations
    output_repr = np.zeros((len(word_rep_1), 2 * len(word_rep_1[0][0])))
    for i in range(len(word_rep_1)):
        print(i)
        if len(word_rep_2[i]) == 1:
            output_repr[i] = np.concatenate((word_rep_1[i][0], word_rep_2[i][0]))
        else:
            output_repr[i] = np.concatenate((word_rep_1[i], word_rep_2[i]))

    # saving the representations
    np.save(output_path, output_repr)


def limit_dict(enc_dec_dict_path, skip_dict_path, skip_repr_path, output_path):
    # read the encoder decoder dictionary
    enc_dec_dict = dict()
    with open(enc_dec_dict_path) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            enc_dec_dict[i] = line[:-1]
    fid.close()

    # read the skip thoughts dictionary
    skip_dict = dict()
    with open(skip_dict_path) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            skip_dict[line[:-1].lower()] = i
    fid.close()

    # read the skip thoughts words representation
    skip_repr = np.load(skip_repr_path)

    output_dict = np.zeros((len(enc_dec_dict), len(skip_repr[0])))
    # creates the new dictionary
    count = 0
    for i in range(len(enc_dec_dict)):
        try:
            output_dict[i] = skip_repr[skip_dict[enc_dec_dict[i].lower()]]
        except:
            try:
                output_dict[i] = skip_repr[skip_dict[enc_dec_dict[i]]]
            except:
                count += 1
                continue
    # save the new dict
    np.save(output_path, output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Convert the tests for the skip-thoughts model")
    parser.add_argument("tests_path", help="The path for the tests dir")
    parser.add_argument("output_dir", help="The output path, should be dir")

    args = parser.parse_args()

    # main(args.tests_path, args.output_dir)

    # merge_word_repr("../data/skip_thoughts/repr/utable.npy", "../data/skip_thoughts/repr/btable.npy",
    #                 "../data/skip_thoughts/repr/word_repr.npy")

    # count = 0
    # for i in range(50001):
    #     id = enc_dec_2_skip_thoughts_dict(i, dict_enc_dec_path="../data/orig/dictionary.txt",
    #                                       dict_skip_path="../data/skip_thoughts/raw/dictionary.txt")
    #     if id == 585031:
    #         count += 1
    # print(count)

    # limit_dict("../data/orig/dictionary.txt", "../data/skip_thoughts/raw/dictionary.txt",
    #            "../data/skip_thoughts/repr/word_repr.npy", "../data/skip_thoughts/repr/s_word_repr.npy")
