__author__ = 'yossiad'


def remove_long_short_sentences(sentences, s_th=3, l_th=25):
    """
    remove too short and too long sentences
    :param sentences: a list containing the orig sentences in formant of: [index, sentence indices as list]
    :param s_th: the threshold for the short sentences, i.e. sentences with length below this number will be removed
    :param l_th: the threshold for the long sentences, i.e. sentences with length above this number will be removed
    :return: a new list of the filtered sentences
    """
    filter_sent = list()
    for i in range(len(sentences)):
        s_len = len(sentences[i][1])
        if s_th <= s_len <= l_th:
            filter_sent.append(sentences[i])
    return filter_sent


def remove_unknown(sentences, unknown_idx=1):
    """
    remove sentences with unknown word in them
    :param sentences: a list containing the orig sentences in formant of: [index, sentence indices as list]
    :param unknown_idx: the index number of unknown word
    :return: a new list of the filtered sentences
    """
    filter_sent = list()
    for i in range(len(sentences)):
        if unknown_idx not in sentences[i][1]:
            filter_sent.append(sentences[i])
    return filter_sent
