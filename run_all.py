import argparse
from train import *


def main(model_path, tests_dir, rep_dir, rep_dim):
    test_type = [Tests.FIRST_WORD, Tests.LAST_WORD, Tests.RANDOM_WORD, Tests.WORD_ORDER, Tests.SENTENCE_LENGTH,
                 Tests.WORD_DISTANCE]
    tests = ['first_word/', 'last_word/', 'random_word/', 'order/', 'sen_len/', 'word_distance/']
    tests_input = [2, 2, 2, 3, 1, 3]
    tests_outputs = [2, 2, 2, 2, 8, 9]
    tests_model_names = ['first.word.' + rep_dim + '.model.net', 'last.word.' + rep_dim + '.model.net',
                         'random.word.' + rep_dim + '.model.net', 'order.' + rep_dim + '.model.net',
                         'sen.len.' + rep_dim + '.model.net', 'wrd.dst.' + rep_dim + '.model.net']
    names = ['FIRST WORD', 'LAST WORD', 'RANDOM WORD', 'WORDS ORDER', 'SENTENCE LENGTH', 'WORDS ORDER DISTANCE']

    for i, t in enumerate(tests):
        print('\n\n============================= %s =============================' % names[i])
        if i != 2:
            continue
        m_path = model_path + tests_model_names[i]
        test_dir = tests_dir + t
        train(m_path, test_dir, rep_dir, int(rep_dim), int(rep_dim) * tests_input[i], tests_outputs[i], test_type[i])
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Analyzing decoder prediction vs. random word test.")
    parser.add_argument("model_path", help="The path to save the models")
    parser.add_argument("tests_dir", help="The path tests dir")
    parser.add_argument("rep_dir", help="The path to the representations dir")
    parser.add_argument("rep_dim", help="The representation size")
    args = parser.parse_args()

    main(args.model_path, args.tests_dir, args.rep_dir, args.rep_dim)
