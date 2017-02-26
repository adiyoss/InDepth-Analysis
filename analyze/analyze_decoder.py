import argparse


def main(decoder_predictions, test_predictions, rand_test):
    # parse the decoder predictions
    decoder_pred = dict()
    with open(decoder_predictions) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            vals = line.split()
            tmp = list()
            for val in vals:
                tmp.append(int(val))
            decoder_pred[i] = tmp
    fid.close()

    # parse the accuracy vals
    test_pred = list()
    with open(test_predictions) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            vals = line.split()
            test_pred.append(int(float(vals[0])))
    fid.close()

    # parse the test data
    test_data = list()
    with open(rand_test) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            vals = line.split()
            test_data.append(int(vals[2]))
    fid.close()

    decoder_test_pred = list()
    for s in range(len(test_data)/2):
        if (test_data[s]+1) in decoder_pred[s]:
            decoder_test_pred.append(0)
        else:
            decoder_test_pred.append(1)

    print("Random test error: %.2f" % (float(sum(test_pred[0:25000]))/25000))
    print("Decoder random test error: %.2f" % (float(sum(decoder_test_pred))/len(decoder_test_pred)))

    # check if the decoder could predict the word when the test did not
    decoder_success = list()
    for i in range(len(test_pred)/2):
        if test_pred[i] == 1 and (test_data[i]+1) in decoder_pred[i]:
            decoder_success.append(i)

    print("\nNumber of decoder success when test fails: %d" % (len(decoder_success)))
    print("Percentage: %.2f" % (100*float(len(decoder_success))/25000))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Analyzing decoder prediction vs. random word test.")
    parser.add_argument("decoder_predictions", help="The path to the file with the decoder predictions")
    parser.add_argument("test_predictions", help="The path to the accuracy at the random word test")
    parser.add_argument("rand_test", help="The path to the random word test data")
    args = parser.parse_args()

    main(args.decoder_predictions, args.test_predictions, args.rand_test)
