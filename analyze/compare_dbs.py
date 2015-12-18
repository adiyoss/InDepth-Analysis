__author__ = 'yossiad'


def wer(r, h):
    """
        Calculation of WER with Levenshtein distance.
        Works only for iterables up to 254 elements (uint8).
        O(nm) time ans space complexity.

        wer("who is there".split(), "is there".split())
        1
        wer("who is there".split(), "".split())
        3
        wer("".split(), "who is there".split())
        3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def check_file_diff(path_1, path_2):
    # read the first file
    fid = open(path_1)
    lines_1 = fid.readlines()
    fid.close()

    # read the second file
    fid = open(path_2)
    lines_2 = fid.readlines()
    fid.close()

    for i in range(len(lines_1)):
        if lines_1[i].strip() != lines_2[i].strip():
            print ""
            print "Orig: ", lines_1[i]
            print "Cons: ", lines_2[i]


def filter_high_wer_sentences(input_path_orig, input_path_decode, output_path):
    # read the first file
    fid = open(input_path_orig)
    lines_orig = fid.readlines()
    fid.close()

    # read the second file
    fid = open(input_path_decode)
    lines_decode = fid.readlines()
    fid.close()

    if len(lines_orig) != len(lines_decode):
        print "The number of sentences do not match."

    f_sen = list()
    for i in range(min(len(input_path_orig), len(input_path_decode))):
        wer_cost = wer(lines_orig[i].split(), lines_decode[i].split())
        if wer_cost <= 2:
            f_sen.append([i, lines_orig[i]])
    fid = open(output_path, 'w')
    for sen in f_sen:
        fid.write(sen[0]+" ")
        fid.write(sen[1]+"\n")
    fid.close()

orig_test = "../../data/representation/orig/test.txt"
construct_test = "../../data/representation/orig/test2.txt"
pred_test = "../../data/representation/orig/pred.txt"
output = "../../data/representation/orig/filt.txt"