
test_idx_path = "../data/idx/order/test.txt"
order_idx_path = "../data/idx/order/test.txt.order.txt"
old_data_x = dict()
new_data_y = dict()

with open(test_idx_path) as fid:
    lines = fid.readlines()
    for i in xrange(0, len(lines), 2):
        vals = lines[i].split()
        old_data_x[i] = (vals[2]+" "+vals[3])
    fid.close()

with open(order_idx_path) as fid:
    lines = fid.readlines()
    for i in xrange(0, len(lines), 2):
        vals = lines[i].split()
        new_data_y[i] = (int(vals[1])-int(vals[0]))
    fid.close()

bins = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 15], [16, 19], [20, 70]]  # 9 bins

# output_path = "../data/idx/word_distance/val.txt"
# fid = open(output_path, 'w')
# stat = dict()
# c = 0
# for i in new_data_y:
#     for b, bin in enumerate(bins):
#         if bin[0] <= new_data_y[i] <= bin[1]:
#             if bin[0] in stat:
#                 stat[bin[0]] += 1
#             else:
#                 stat[bin[0]] = 1
#             vals = old_data_x[i].split()
#             fid.write(str(b) + ' ' + str(c) + ' ' + vals[0] + ' ' + vals[1] + '\n')
#             break
#     c += 1
# fid.close()
#
# for i in stat:
#     print stat[i]
