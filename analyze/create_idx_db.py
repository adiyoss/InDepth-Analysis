import numpy as np
import sys

__author__ = 'yossiad'
premute = np.random.permutation(50001)


def create_first_word_db(output_path, sen_idx, dict_size=50001):
    """
    Create db for first word analysis
    We consider this test as binary classification
    Here the input is the sentence representation concatenate with the representation of the some word
    For positive examples we use the representation of the first word in the sentence
    For negative examples we use the representation of the some random word which could be from another location in the
    sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param dict_size: the size of the dictionary
    """
    first_words_cache = list()
    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target_wrd = sen_idx[i][1][0]
        if target_wrd not in first_words_cache:
            first_words_cache.append(target_wrd)

        target_wrd -= 1

        # positive example
        fid.write("1 ")
        fid.write(str(sen_idx[i][0]))
        fid.write(" ")
        fid.write(str(target_wrd))
        fid.write("\n")

    for i in range(len(sen_idx)):
        # negative example
        idx = np.random.randint(len(first_words_cache))
        target_wrd = sen_idx[i][1][0]
        rnd = first_words_cache[idx]
        while rnd == target_wrd:
            idx = np.random.randint(len(first_words_cache))
            rnd = first_words_cache[idx]
        rnd -= 1

        fid.write("0 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(rnd))
        fid.write("\n")
    fid.close()


def create_last_word_db(output_path, sen_idx, dict_size=50001):
    """
    Create db for last word analysis
    We consider this test as binary classification
    Here the input is the sentence representation concatenate with the representation of the some word
    For positive examples we use the representation of the last word in the sentence
    For negative examples we use the representation of the some random word which could be from another location in the
    sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param dict_size: the dictionary size
    """
    last_words_cache = list()
    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 1]
        if target_wrd == 4:  # avoid dot token
            target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 2]

        if target_wrd not in last_words_cache:
            last_words_cache.append(target_wrd)
        target_wrd -= 1

        # positive example
        fid.write("1 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd))
        fid.write("\n")

    for i in range(len(sen_idx)):
        # negative example
        idx = np.random.randint(len(last_words_cache))
        rnd = last_words_cache[idx]
        target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 1]
        if target_wrd == 4:  # avoid dot token
            target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 2]
        while rnd == target_wrd:
            idx = np.random.randint(len(last_words_cache))
            rnd = last_words_cache[idx]
        rnd -= 1

        fid.write("0 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(rnd))
        fid.write("\n")
    fid.close()


def create_last_word_multi_class_db(output_path, sen_idx):
    """
    Create db for last word analysis
    We consider this test as multi-class classification, predict the word index
    Here the input is the sentence representation
    The desired output is the index of the last word in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    """

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 1] - 1
        if target_wrd == 3:  # avoid dot token
            target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 2] - 1
        # positive example
        fid.write(str(target_wrd) + " ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write("\n")
    fid.close()


def create_random_word_db(output_path, sen_idx, dict_size=50001):
    """
    Create db for existing of random word in the sentence
    We consider this test as binary classification
    Here the input is the sentence representation concatenate with the representation of the some word
    For positive examples we use the representation of the some random word from the sentence
    For negative examples we use the representation of the some random word which does not appear in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param dict_size: the size of the dictionary
    """

    random_words_cache = list()
    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        idx = np.random.randint(len(sen_idx[i][1]))
        target_wrd_pos = sen_idx[i][1][idx]

        if len(random_words_cache) == 0:
            random_words_cache.append(target_wrd_pos)
        else:
            count = 0  # give up after 20 tries
            num_of_tries = 20
            while (target_wrd_pos in random_words_cache) and (count < num_of_tries):
                idx = np.random.randint(len(sen_idx[i][1]))
                target_wrd_pos = sen_idx[i][1][idx]
                count += 1
            if target_wrd_pos not in random_words_cache:
                random_words_cache.append(target_wrd_pos)
        # convert from torch to python
        target_wrd_pos -= 1

        # positive example
        fid.write("1 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd_pos))
        fid.write("\n")

    for i in range(len(sen_idx)):
        idx = np.random.randint(len(random_words_cache))
        target_wrd_neg = random_words_cache[idx]
        while target_wrd_neg in sen_idx[i][1]:
            idx = np.random.randint(len(random_words_cache))
            target_wrd_neg = random_words_cache[idx]
        target_wrd_neg -= 1

        # negative example
        fid.write("0 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd_neg))
        fid.write("\n")
    fid.close()


def create_random_word_db_hard(output_path, sen_idx, word_repr, dict_size=50001):
    close_vecs = get_close_vecs(word_repr)
    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        idx = np.random.randint(len(sen_idx[i][1]))
        target_wrd_pos = sen_idx[i][1][idx]
        target_wrd_pos -= 1  # convert from torch to python

        # positive example
        fid.write("1 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd_pos))
        fid.write("\n")

        target_wrd_neg = close_vecs[target_wrd_pos + 1] - 1  # convert from torch to python
        # negative example
        fid.write("0 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd_neg))
        fid.write("\n")
    fid.close()


def get_close_vecs(word_repr):
    from scipy import spatial
    tree_rep = spatial.KDTree(word_repr)
    close_vecs = dict()
    for i, r in enumerate(word_repr):
        c = tree_rep.query(r)
        close_vecs[i] = c[1]
    return close_vecs


def create_following_words_db(output_path, sen_idx):
    """
    Create db for words order, the first word should appear in the somewhere in the first half of the sentence
    the following word should appear somewhere later in the sentence (the second half of the sentence)
    Here the input is the sentence representation concatenate with the representations of two words from the sentence
    For positive examples we concatenate the words representations in the original order they appear in the sentence
    For negative examples we concatenate the words representations opposite to the direction they appear in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    """
    word_order_file = open(output_path + ".order.txt", 'w')
    fid = open(output_path, 'w')

    # checker path
    check_path = output_path + ".check"
    fid_c = open(check_path, 'w')
    sen_repr_size = len(sen_idx)

    for i in range(len(sen_idx)):
        idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) / 2)
        idx_2 = np.random.randint(low=len(sen_idx[i][1]) / 2 + 1, high=len(sen_idx[i][1]))

        # idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) - 1)
        # idx_2 = np.random.randint(low=idx_1 + 1, high=len(sen_idx[i][1]))
        target_wrd_1 = sen_idx[i][1][idx_1] - 1
        target_wrd_2 = sen_idx[i][1][idx_2] - 1

        sen_id = np.random.randint(sen_repr_size)
        while sen_id == sen_idx[i][0]:
            sen_id = np.random.randint(sen_repr_size)

        # positive example
        fid.write("1 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd_1))
        fid.write(" ")
        fid.write(str(target_wrd_2))
        fid.write("\n")
        word_order_file.write(str(idx_1) + " " + str(idx_2) + "\n")

        # negative example
        fid.write("0 ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write(" ")
        fid.write(str(target_wrd_2))
        fid.write(" ")
        fid.write(str(target_wrd_1))
        fid.write("\n")
        word_order_file.write(str(idx_2) + " " + str(idx_1) + "\n")
        # ===================================== #

        # ============== CHECKER ============== #
        # positive example
        fid_c.write("1 ")
        fid_c.write(str(sen_id))
        fid_c.write(" ")
        fid_c.write(str(target_wrd_1))
        fid_c.write(" ")
        fid_c.write(str(target_wrd_2))
        fid_c.write("\n")

        # negative example
        fid_c.write("0 ")
        fid_c.write(str(sen_id))
        fid_c.write(" ")
        fid_c.write(str(target_wrd_2))
        fid_c.write(" ")
        fid_c.write(str(target_wrd_1))
        fid_c.write("\n")
        # ===================================== #
    fid.close()
    fid_c.close()
    word_order_file.close()


def create_sentence_length_db(output_path, sen_idx):
    """
    Create db for sentence length
    Here the input is the sentence representation by itself
    The goal is to predict the sentence length (the number of words in it)
    We consider this task a multi-class classification
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    """
    INFINITY = 1000
    # bins = [[5, 8], [9, 12], [13, 16], [17, 20], [21, 25], [26, INFINITY]]  # 6 bins
    bins = [[5, 8], [9, 12], [13, 16], [17, 20], [21, 25], [26, 29], [30, 33], [34, INFINITY]]  # 8 bins
    # bins = [[5, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
    #         [35, 37], [38, INFINITY]]  # 12 bins

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target = 0
        sen_len = len(sen_idx[i][1])
        for b in range(len(bins)):
            if int(bins[b][0]) <= int(sen_len) <= int(bins[b][1]):
                target = b
        fid.write(str(target) + " ")
        fid.write(str(str(sen_idx[i][0])))
        fid.write("\n")
    fid.close()


def create_next_word_prediction(output_path, sen_idx):
    """
    Create db for the next word prediction
    Here the input is sentence representation concatenated with the representation of a single word from the sentence
    Our goal is to predict the id of next word in the sentence, we consider this test as multi-class classification
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    """
    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) - 1)
        while sen_idx[i][1].count(idx_1) == 1:
            idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) - 1)

        target_wrd_1 = sen_idx[i][1][idx_1] - 1
        target = sen_idx[i][1][idx_1 + 1] - 1

        # positive example
        fid.write(str(target) + " ")
        fid.write(str(sen_idx[i][0]))
        fid.write(" ")
        fid.write(str(target_wrd_1))
        fid.write("\n")
    fid.close()


def random_test_k_closest(output_path, sen_idx, words):
    from scipy import spatial
    tree = spatial.KDTree(words, leafsize=1000)
    count = 0
    avg_distance = 0
    cache = dict()
    for s in range(len(sen_idx)):
        # sys.stdout.write('\r' + str(s))
        tmp_words = list()
        for w in sen_idx[s][1]:
            if w > 500:
                tmp_words.append(w)
        if len(tmp_words) > 0:
            rand_word_id = tmp_words[np.random.randint(low=0, high=len(tmp_words))] - 1  # minus 1 for torch
        else:
            rand_word_id = sen_idx[s][1][
                               np.random.randint(low=0, high=len(sen_idx[s][1]))] - 1  # minus 1 for torch
            count += 1
        if rand_word_id in cache:
            ids = cache[rand_word_id]
        else:
            ids = tree.query(words[rand_word_id], 501)
            n_ids = tree.query(words[ids[1][1]], 501)
            avg_distance += ids[0][1]
            print "==================="
            print "%.3f, %.3f, %.3f, %.3f, %.3f" % (ids[0][1], ids[0][10], ids[0][50], ids[0][100], ids[0][500])
            print "%.3f, %.3f, %.3f, %.3f, %.3f" % (n_ids[0][1], n_ids[0][10], n_ids[0][50], n_ids[0][100], n_ids[0][500])
            print "==================="
            cache[rand_word_id] = ids
    print(avg_distance / len(sen_idx))

    # fid_out = open(output_path + 'train.txt', 'w')
    # count = 0
    # cache = dict()
    # for s in range(len(sen_idx)):
    #     sys.stdout.write('\r' + str(s))
    #     tmp_words = list()
    #     for w in sen_idx[s][1]:
    #         if w > 500:
    #             tmp_words.append(w)
    #     if len(tmp_words) > 0:
    #         rand_word_id = tmp_words[np.random.randint(low=0, high=len(tmp_words))] - 1  # minus 1 for torch
    #     else:
    #         rand_word_id = sen_idx[s][1][
    #                            np.random.randint(low=0, high=len(sen_idx[s][1]))] - 1  # minus 1 for torch
    #         count += 1
    #     if rand_word_id in cache:
    #         ids = cache[rand_word_id]
    #     else:
    #         ids = tree.query(words[rand_word_id], 501)[1]
    #         cache[rand_word_id] = ids
    #     fid_out.write(
    #         str(s) + ' ' + str(ids[0]) + ' ' + str(ids[1]) + ' ' + str(ids[10]) + ' ' + str(ids[50]) + ' ' + str(
    #                 ids[100]) + ' ' + str(ids[500]) + '\n')
    # fid_out.close()
    print '\nNumber of under 500: %d' % count

    # x_ranges = [0, 1, 10, 50, 100]
    # y_ranges = [10, 50, 100, 500]
    # for x in x_ranges:
    #     for y in y_ranges:
    #         print('======' + str(x) + '_' + str(y) + '======')
    #         if y <= x:
    #             continue
    #         fid_out = open(output_path + str(x) + '_' + str(y) + '.txt', 'w')
    #         count = 0
    #         for s in range(len(sen_idx)):
    #             sys.stdout.write('\r' + str(s))
    #             tmp_words = list()
    #             for w in sen_idx[s][1]:
    #                 if w > 500:
    #                     tmp_words.append(w)
    #             if len(tmp_words) > 0:
    #                 rand_word_id = tmp_words[np.random.randint(low=0, high=len(tmp_words))] - 1  # minus 1 for torch
    #             else:
    #                 rand_word_id = sen_idx[s][1][
    #                                    np.random.randint(low=0, high=len(sen_idx[s][1]))] - 1  # minus 1 for torch
    #                 count += 1
    #             if x == 0:
    #                 w_x = rand_word_id
    #             else:
    #                 w_x = tree.query(words[rand_word_id], (x + 1))[1][x]
    #             w_y = tree.query(words[rand_word_id], y + 1)[1][y]
    #             fid_out.write('1 ' + str(s) + ' ' + str(w_x) + '\n')
    #             fid_out.write('0 ' + str(s) + ' ' + str(w_y) + '\n')
    #         fid_out.close()
    #         print 'Number of under 500: %d' % count
