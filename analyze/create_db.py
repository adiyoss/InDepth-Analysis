import numpy as np

__author__ = 'yossiad'


def create_first_word_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for first word analysis
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
    Create db for existing of random word in the sentence, i.e. positive example will contain:
    sentence representation, representation of random word from the sentence, negative example will contain:
    sentence representation, representation of random word that did not appear in the sentence
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

    for i in range(len(sen_idx)):
        for j in range(len(sen_idx[i][1])):
            idx_1 = j
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
    Create db for the two last words, i.e. any positive example contains the sentence representation,
    the last word and the one before it. In negative examples we flipped
    the words order
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
        idx_1 = np.random.randint(low=0, high=len(sen_idx[i][1]) / 2)
        idx_2 = np.random.randint(low=len(sen_idx[i][1]) / 2 + 1, high=len(sen_idx[i][1]))
        target_wrd_1 = sen_idx[i][1][idx_1] - 1
        target_wrd_2 = sen_idx[i][1][idx_2] - 1

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
    fid.close()
    word_order_file.close()


def create_order_words_db(output_path, sen_idx, sen_repr_path, words):
    """
    Create db for any order words, i.e. any positive example contains the sentence representation,
    some random word and another random word that appears later in the sentence. In negative examples we flipped
    the words order
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
    Create db for existing of random word in the sentence, i.e. positive example will contain:
    sentence representation, representation of random word from the sentence, negative example will contain:
    sentence representation, representation of random word that did not appear in the sentence
    :param output_path: the path to save the data
    :param sen_idx: a mapping between sentence id to its indices
    :param sen_repr_path: a mapping between sentence id to its representation
    :param words: a mapping between word id to its representation
    """
    INFINITY = 1000
    bins = [[2, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, INFINITY]]
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
                print "\n"
                print sen_len
                print bins[b]
                print "\n"
                target = b
        fid.write(str(target) + " ")
        fid.write(sen_repr[(sen_idx[i][0])][1][:-1])
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
