#!/usr/bin/env python

import operator

word_stat = '../model/word_stat3.txt'

score = []
with open(word_stat) as inf:
    for line in inf:
        line = line.strip().split(',')
        score.append([line[0], float(line[1] if line[0] else 0)/(float(line[2] if line[2] != 'None' else 0) + float(line[3] if line[3] != 'None' else 0) + 0.002)])

score = sorted(score, key=operator.itemgetter(1), reverse=True)
for item in score:
    print '%s,%s' % (item[0], item[1])
