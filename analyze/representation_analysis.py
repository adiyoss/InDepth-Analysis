import argparse
import os

import reader as r
import create_db as db
import numpy as np

__author__ = 'yossiad'

np.random.seed(1521)  # for reproducibility

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Create DB for the representation analysis, i.e. remove long/short sentences, remove "
                        "sentences with known words")
    parser.add_argument("in_filename", help="The path to the train/test/val file, it should be in index format not"
                                            " exact words")
    parser.add_argument("repr", help="The path to the train/test/val representation file")
    parser.add_argument("out_filename", help="The output path should be dir")
    parser.add_argument("file_name", help="the file name to be created for each test")
    parser.add_argument("--words_repr", help="The path to the words representation file",
                        default="../data/orig/word_repr.txt")
    parser.add_argument("--dictionary", help="The path to the dictionary",
                        default="../data/orig/dictionary.txt")
    args = parser.parse_args()

    dictionary = r.read_dictionary(args.dictionary, args.words_repr)
    print "Dictionary size is: ", len(dictionary)

    sent = r.read_files(args.in_filename)
    print "Number of original sentences is: ", len(sent)

    # # =========== FIRST WORD =========== #
    # print("\nCreate first word db ...")
    # first_word_path = args.out_filename+"first_word/"
    # first_word_filename = first_word_path+args.file_name
    # if not os.path.exists(first_word_path):
    #     os.mkdir(args.out_filename+"first_word")
    # db.create_first_word_db(first_word_filename, sent, args.repr, dictionary)
    # print("Done.")
    # # ================================== #
    #
    # # ============ LAST WORD =========== #
    # print("\nCreate last word db ...")
    # last_word_path = args.out_filename+"last_word/"
    # last_word_filename = last_word_path+args.file_name
    # if not os.path.exists(last_word_path):
    #     os.mkdir(last_word_path)
    # db.create_last_word_db(last_word_filename, sent, args.repr, dictionary)
    # print("Done.")
    # # ================================== #
    #
    # # =========== RANDOM WORD ========== #
    # print("\nCreate random word db ...")
    # random_word_path = args.out_filename+"random_word/"
    # random_word_filename = random_word_path+args.file_name
    # if not os.path.exists(random_word_path):
    #     os.mkdir(random_word_path)
    # db.create_random_word_db(random_word_filename, sent, args.repr, dictionary)
    # print("Done.")
    # # ================================== #
    #
    # # ========= SENTENCE LENGTH ======== #
    # print("\nCreate sentence length db ...")
    # sen_len_path = args.out_filename+"sen_len/"
    # sen_len_filename = sen_len_path+args.file_name
    # if not os.path.exists(sen_len_path):
    #     os.mkdir(sen_len_path)
    # db.create_sentence_length_db(sen_len_filename, sent, args.repr, dictionary)
    # print("Done.")
    # # ================================== #
    #
    # # =========== ORDER WORDS ========== #
    # print("\nCreate word order db ...")
    # order_path = args.out_filename+"order/"
    # order_filename = order_path+args.file_name
    # if not os.path.exists(order_path):
    #     os.mkdir(order_path)
    # db.create_following_words_db(order_filename, sent, args.repr, dictionary)
    # print("Done.")
    # # ================================== #

    # # ======= ORDER WORDS CHECKER ====== #
    # print("\nCreate checker for word order db ...")
    # order_path_check = args.out_filename+"order_checker/"
    # order_check_filename = order_path_check+args.file_name
    # if not os.path.exists(order_path_check):
    #     os.mkdir(order_path_check)
    # db.create_checker_word_order_db(order_check_filename, sent, args.repr, dictionary)
    # print("Done.")
    # # ================================== #
