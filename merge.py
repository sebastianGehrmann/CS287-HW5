import sys
import collections

default = lambda: collections.defaultdict(collections.Counter)
tag_freq = collections.defaultdict(default)

def main(args):
    in_files = args[1:-1]
    out_file = args[-1]
    sum_preds = collections.defaultdict(int)
    header = None
    n = len(in_files)
    for file_name in in_files:
        with open(file_name, "r") as f:
            header = f.readline().strip()
            for line in f:
                res = line.split(',')
                prevrow = 0
                if len(res) >= 2:
                    row, tagline = res
                    row = int(row)
                    for i in xrange(prevrow + 1, row):
                        tag_freq[i]
                    prevrow = row

                    tags = tagline.split(' ')
                    prevind = 0
                    for tag in tags:
                        inds = tag.split('-')
                        label = inds[0]
                        for ind in inds[1:]:
                            ind = int(ind)
                            for i in xrange(prevind + 1, ind):
                                tag_freq[row][i][""] += 1
                            tag_freq[row][ind][label] += 1
                            prevind = ind
    with open(out_file, "w") as f:
        print >>f, header
        for row in sorted(tag_freq.keys()):
            d = tag_freq[row]
            prevind = None
            prevlabel = None
            first = True
            f.write("%d," % row)
            for ind in sorted(d.keys()):
                label = d[ind].most_common(1)[0][0]
                if label:
                    if prevlabel != label:
                        if not first:
                            f.write(' ')
                        f.write(label)
                        first = False
                    f.write("-%d" % ind)
                prevlabel = label
            f.write('\n')

if __name__ == "__main__":
    main(sys.argv)
