import numpy as np

__author__ = 'yossiad'


def create_first_word_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for first word analysis
    We consider this test as binary classification
    Here the input is the sentence representation concatenate with the representation of the some word
    For positive examples we use the representation of the first word in the sentence
    For negative examples we use the representation of the some random word which could be from another location in the
    sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target_wrd = sen_idx[i][1][0] - 1
        # positive example
        fid.write("1 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd][1]))
        fid.write("\n")

        # negative example
        rnd = np.random.randint(len(words))
        while rnd == target_wrd:
            rnd = np.random.randint(len(words))
        fid.write("0 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[rnd][1]))
        fid.write("\n")
    fid.close()


def create_last_word_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for last word analysis
    We consider this test as binary classification
    Here the input is the sentence representation concatenate with the representation of the some word
    For positive examples we use the representation of the last word in the sentence
    For negative examples we use the representation of the some random word which could be from another location in the
    sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 1] - 1
        if target_wrd == 3:  # avoid dot token
            target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 2] - 1
        # positive example
        fid.write("1 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd][1]))
        fid.write("\n")

        # negative example
        rnd = np.random.randint(len(words))
        while rnd == target_wrd:
            rnd = np.random.randint(len(words))
        fid.write("0 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[rnd][1]))
        fid.write("\n")
    fid.close()


def create_last_word_multi_class_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for last word analysis
    We consider this test as multi-class classification, predict the word index
    Here the input is the sentence representation
    The desired output is the index of the last word in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 1] - 1
        if target_wrd == 3:  # avoid dot token
            target_wrd = sen_idx[i][1][len(sen_idx[i][1]) - 2] - 1
        # positive example
        fid.write(str(target_wrd) + " ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write("\n")
    fid.close()


def create_random_word_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for existing of random word in the sentence
    We consider this test as binary classification
    Here the input is the sentence representation concatenate with the representation of the some word
    For positive examples we use the representation of the some random word from the sentence
    For negative examples we use the representation of the some random word which does not appear in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        # for j in range(len(sen_idx[i][1])):

        idx_1 = np.random.randint(len(sen_idx[i][1]))
        target_wrd_pos = sen_idx[i][1][idx_1]

        target_wrd_2 = target_wrd_pos
        while target_wrd_2 in sen_idx[i][1]:
            target_wrd_2 = np.random.randint(len(words))

        # positive example
        fid.write("1 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_pos - 1][1]))
        fid.write("\n")

        # negative example
        fid.write("0 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_2 - 1][1]))
        fid.write("\n")
    fid.close()


def create_following_words_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for words order, the first word should appear in the somewhere in the first half of the sentence
    the following word should appear somewhere later in the sentence (the second half of the sentence)
    Here the input is the sentence representation concatenate with the representations of two words from the sentence
    For positive examples we concatenate the words representations in the original order they appear in the sentence
    For negative examples we concatenate the words representations opposite to the direction they appear in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    word_order_file = open(output_path + ".order.txt", 'w')
    fid = open(output_path, 'w')

    # checker path
    check_path = output_path + ".check"
    fid_c = open(check_path, 'w')
    sen_repr_size = len(sen_repr)

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
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_1][1]))
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_2][1]))
        fid.write("\n")
        word_order_file.write(str(idx_1) + " " + str(idx_2) + "\n")

        # negative example
        fid.write("0 ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_2][1]))
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_1][1]))
        fid.write("\n")
        word_order_file.write(str(idx_2) + " " + str(idx_1) + "\n")
        # ===================================== #

        # ============== CHECKER ============== #
        # positive example
        fid_c.write("1 ")
        fid_c.write(sen_repr[sen_id][1][:-1])
        fid_c.write(" ")
        fid_c.write(vector2string(words[target_wrd_1][1]))
        fid_c.write(" ")
        fid_c.write(vector2string(words[target_wrd_2][1]))
        fid_c.write("\n")

        # negative example
        fid_c.write("0 ")
        fid_c.write(sen_repr[sen_id][1][:-1])
        fid_c.write(" ")
        fid_c.write(vector2string(words[target_wrd_2][1]))
        fid_c.write(" ")
        fid_c.write(vector2string(words[target_wrd_1][1]))
        fid_c.write("\n")
        # ===================================== #
    fid.close()
    fid_c.close()
    word_order_file.close()


def create_checker_word_order_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for words order validation
    Here the input is the sentence representation concatenate with the representations of two words from the sentence
    The examples generation are identical to the above function, but instead of using the original sentence representation
    we use the representation of random sentence
    The goal of this test is to check if the words order has some relation to the sentence representation or is it only
    words order statistics
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    word_order_file = open("order.txt", 'w')
    fid = open(output_path, 'w')
    sen_repr_size = len(sen_repr)
    for i in range(len(sen_idx)):
        idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) / 2)
        idx_2 = np.random.randint(low=len(sen_idx[i][1]) / 2 + 1, high=len(sen_idx[i][1]))
        target_wrd_1 = sen_idx[i][1][idx_1] - 1
        target_wrd_2 = sen_idx[i][1][idx_2] - 1

        sen_id = np.random.randint(sen_repr_size)
        while sen_id == sen_idx[i][0]:
            sen_id = np.random.randint(sen_repr_size)

        # positive example
        fid.write("1 ")
        fid.write(sen_repr[sen_id][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_1][1]))
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_2][1]))
        fid.write("\n")
        word_order_file.write(str(idx_1) + " " + str(idx_2) + "\n")

        # negative example
        fid.write("0 ")
        fid.write(sen_repr[sen_id][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_2][1]))
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_1][1]))
        fid.write("\n")
        word_order_file.write(str(idx_2) + " " + str(idx_1) + "\n")
    fid.close()
    word_order_file.close()


def create_order_words_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for any order words, like the above function but generating all the possible words order
    Use this function when you have limited amount of data
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    word_order_file = open("order.txt", 'w')
    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        for j in range(len(sen_idx[i][1])):
            idx_1 = j
            for k in range(j + 1, len(sen_idx[i][1])):
                idx_2 = k
                target_wrd_1 = sen_idx[i][1][idx_1] - 1
                target_wrd_2 = sen_idx[i][1][idx_2] - 1

                # write the words order for the graph plot
                word_order_file.write(str(target_wrd_1) + " " + str(target_wrd_2))

                # positive example
                fid.write("1 ")
                fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
                fid.write(" ")
                fid.write(vector2string(words[target_wrd_2][1]))
                fid.write(" ")
                fid.write(vector2string(words[target_wrd_1][1]))
                fid.write("\n")

                # write the words order for the graph plot
                word_order_file.write(str(target_wrd_1) + " " + str(target_wrd_2))

                # negative example
                fid.write("0 ")
                fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
                fid.write(" ")
                fid.write(vector2string(words[target_wrd_1][1]))
                fid.write(" ")
                fid.write(vector2string(words[target_wrd_2][1]))
                fid.write("\n")
    fid.close()
    word_order_file.close()


def create_sentence_length_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for sentence length
    Here the input is the sentence representation by itself
    The goal is to predict the sentence length (the number of words in it)
    We consider this task a multi-class classification
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    INFINITY = 1000
    # bins = [[5, 8], [9, 12], [13, 16], [17, 20], [21, 25], [26, INFINITY]]  # 6 bins
    # bins = [[5, 8], [9, 12], [13, 16], [17, 20], [21, 25], [26, 29], [30, 33], [34, INFINITY]]  # 8 bins
    bins = [[5, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
            [35, 37], [38, INFINITY]]  # 12 bins
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        target = 0
        sen_len = len(sen_idx[i][1])
        for b in range(len(bins)):
            if int(bins[b][0]) <= int(sen_len) <= int(bins[b][1]):
                target = b
        fid.write(str(target) + " ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write("\n")
    fid.close()


def create_next_word_prediction_multi(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for the next word prediction
    Here the input is sentence representation concatenated with the representation of a single word from the sentence
    Our goal is to predict the id of next word in the sentence, we consider this test as multi-class classification
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) - 1)

        target_wrd_1 = sen_idx[i][1][idx_1] - 1
        target = sen_idx[i][1][idx_1 + 1] - 1

        # positive example
        fid.write(str(target) + " ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_1][1]))
        fid.write("\n")
    fid.close()


def create_next_word_prediction_repr(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for the next word prediction
    Here the input is sentence representation concatenated with the representation of a single word from the sentence
    Our goal is to predict the representation of the next word in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    sen_repr = list()
    fid = open(sen_repr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        sen_repr.append([i, lines[i]])

    fid = open(output_path, 'w')
    for i in range(len(sen_idx)):
        idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) - 1)

        target_wrd_1 = sen_idx[i][1][idx_1] - 1
        target = sen_idx[i][1][idx_1 + 1] - 1

        fid.write(vector2string(words[target][1]) + " ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
        fid.write(" ")
        fid.write(vector2string(words[target_wrd_1][1]))
        fid.write("\n")
    fid.close()


def vector2string(vec):
    """
    Helper function, converts vector to one string separated by spaces
    :param vec:
    :return:
    """
    s = ""
    for item in vec:
        s += str(item) + " "
    return s
