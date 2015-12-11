import argparse
import gc
import reader as r
import filters as f
import create_db as db

__author__ = 'yossiad'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create DB for the representation analysis, i.e. remove long/short sentences, remove "
                    "sentences with known words")
    parser.add_argument("in_filename", help="The path to the train/test/val file, it should be in index format not"
                                            " exact words")
    parser.add_argument("repr", help="The path to the train/test/val representation file")
    parser.add_argument("out_filename", help="The output path")
    parser.add_argument("--words_repr", help="The path to the words representation file",
                        default="../../data/representation/orig/word_repr.txt")
    parser.add_argument("--dictionary", help="The path to the dictionary",
                        default="../../data/representation/orig/dictionary")
    args = parser.parse_args()

    dictionary = r.read_dictionary(args.dictionary, args.words_repr)
    print "Dictionary size is: ", len(dictionary)

    orig_sent = r.read_files(args.in_filename)
    print "Number of original sentences is: ", len(orig_sent)

    f1_sent = f.remove_long_short_sentences(orig_sent)
    print "Number of sentences filtered by length is: ", (len(orig_sent) - len(f1_sent))
    gc.collect()

    f2_sent = f.remove_unknown(f1_sent)
    print "Number of sentences filtered because containing unknown word is: ", (len(f1_sent) - len(f2_sent))
    print ""
    print "Total number of sentences left: ", len(f2_sent)
    gc.collect()

    # db.create_first_word_db(args.out_filename, f2_sent, args.repr, dictionary)
    # db.create_last_word_db(args.out_filename, f2_sent, args.repr, dictionary)
    # db.create_following_words_db(args.out_filename, f2_sent, args.repr, dictionary)
    # db.create_order_words_db(args.out_filename, f2_sent, args.repr, dictionary)
    db.create_random_word_db(args.out_filename, f2_sent, args.repr, dictionary)