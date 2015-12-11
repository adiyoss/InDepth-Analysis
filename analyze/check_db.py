import argparse
import gc
import reader as r
import filters as f
import create_db as c

__author__ = 'yossiad'


def check(in_filename, sen_repr_path, db_path, word_repr, dictionary):
    # read the data set
    db = r.read_files_float(db_path)
    # read dictionary
    d = r.read_dictionary(dictionary, word_repr)

    # read the original sentences indices and filter them
    orig_sent = r.read_files(in_filename)
    f1_sent = f.remove_long_short_sentences(orig_sent)
    gc.collect()
    f2_sent = f.remove_unknown(f1_sent)
    gc.collect()

    # read the representations
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    word_test_flag = True
    sentence_test_flag = True
    log_word = ""
    log_sen = ""

    # testing
    for i in range(len(f2_sent)):
        # target_word = 0  # first word test
        target_word = len(f2_sent[i][1]) - 1  # last word

        sen_from_db = c.vector2string(db[i * 2][1][1:1001])
        w_from_db = c.vector2string(db[i * 2][1][1001:2001])

        w_target = c.vector2string(d[f2_sent[i][1][target_word] - 1][1])
        sen_target = c.vector2string([float(x) for x in sen_repr[f2_sent[i][0]][1].split()])

        if w_from_db != w_target:
            log_word += "From DB: " + w_from_db + "\n"
            log_word += "Target: " + w_target + "\n\n"
            word_test_flag = False

        if sen_from_db != sen_target:
            log_sen += "From DB: " + sen_from_db + "\n"
            log_sen += "Target: " + sen_target + "\n\n"
            sentence_test_flag = False

    # test summary
    if sentence_test_flag and word_test_flag:
        print "Test pass!"
    elif not sentence_test_flag and word_test_flag:
        print "Word test pass, sentence test failed."
        print log_sen
    elif sentence_test_flag and not word_test_flag:
        print "Sentence test pass, word test failed."
        print log_word
    else:
        print "Both sentence and word tests failed."
        print "SENTENCE:"
        print log_sen
        print "WORD:"
        print log_word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create DB for the representation analysis, i.e. remove long/short sentences, remove "
                    "sentences with known words")
    parser.add_argument("in_filename", help="The path to the train/test/val file, it should be in index format not"
                                            " exact words")
    parser.add_argument("repr", help="The path to the train/test/val representation file")
    parser.add_argument("db", help="The path to the train/test/val db file")
    parser.add_argument("--words_repr", help="The path to the words representation file",
                        default="../../data/representation/orig/word_repr.txt")
    parser.add_argument("--dictionary", help="The path to the dictionary",
                        default="../../data/representation/orig/dictionary")
    args = parser.parse_args()

    check(args.in_filename, args.repr, args.db, args.words_repr, args.dictionary)