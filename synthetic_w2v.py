import numpy as np

dict_size = 50001
num_of_examples = 50000
embed_size = 100
max_sen_len = 50

word_rep = np.load("data/w2v_100_win5/word_repr.npy")
prob = np.zeros(dict_size)

for i in range(1, len(word_rep)):
    prob[i] = (1/float(i+1))
prob /= np.sum(prob)
x = np.arange(dict_size)

data_set = np.zeros((num_of_examples, embed_size))
label = list()
for i in range(num_of_examples):
    sen_tmp = np.zeros(embed_size)
    label_tmp = ""
    s_l = np.random.randint(low=5, high=max_sen_len)
    for j in range(s_l):
        w_idx = np.random.choice(x, p=prob)
        tmp = word_rep[w_idx]
        sen_tmp += tmp
        label_tmp += (str(w_idx)+" ")
    sen_tmp /= s_l
    data_set[i] = sen_tmp
    label.append(label_tmp)

np.save("test_rep.txt", data_set)
with open("test.txt", 'w') as fid:
    for item in label:
        fid.write(item+"\n")
fid.close()

