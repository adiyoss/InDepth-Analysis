__author__ = 'yossiad'


def read_files_float(path):
    """
    read all the lines in path
    :param path:  the file path
    :returns a list containing the file content
    """
    sen = list()
    fid = file(path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        numbers = [float(x) for x in lines[i][:-1].split()]
        sen.append([i, numbers])
    return sen


def read_files(path):
    """
    read all the lines in path
    :param path:  the file path
    :returns a list containing the file content
    """
    sen = list()
    fid = file(path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        numbers = [int(x) for x in lines[i][:-1].split()]
        sen.append([i, numbers])
    return sen


def read_dictionary(dict_path, wr_path):
    """
    read all the lines in path
    :param dict_path:  the path to the dictionary
    :param wr_path:  the path to the word representation
    :returns a list the words and their representations
    """
    raw_words = dict()
    fid = file(dict_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines)):
        raw_words[i] = lines[i][:-1]

    words = list()
    fid = open(wr_path)
    lines = fid.readlines()
    fid.close()
    for i in range(len(lines) - 1):
        repr = [float(x) for x in lines[i][:-1].split()]
        words.append([raw_words[i], repr])
    return words

