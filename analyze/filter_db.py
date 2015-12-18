in_path = "../data/orig/index_trainBatch32_filter.txt"
out_path = "../data/orig/index_trainBatch32_filter_aggressive.txt"

lower_bound = 5
upper_bound = 70

f_out = open(out_path, 'w')
with open(in_path) as f:
    for line in f:
        vals = line.split()
        if lower_bound <= len(vals) <= upper_bound:
            count = 0
            for i in vals:
                if int(i) == 1:
                    count += 1
                    break
            if count < 1:
                for i in vals:
                    f_out.write(str(i) + ' ')
                f_out.write('\n')
f.close()
f_out.close()
