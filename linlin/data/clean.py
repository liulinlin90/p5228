#!/usr/bin/env python

import re
import string
import pandas as pd

data_path = '/Users/linlin/Desktop/nus/cs5228/project/data/train.csv'
out_path = '/Users/linlin/Desktop/nus/cs5228/project/data/processed/train.csv'
emo_pos = [':-)', ':)', '(:', '(-:', '<3', ':-D', ': D', ' XD', ';-)',
           ';)', ';-D', '; D']
emo_neg = [':-(', ':(', ': (', ':((', 'T_T', ' t_t', ' ><']


def process(line):
    for e in emo_pos:
        line = line.replace(e, ' myemopos ')
    for e in emo_neg:
        line = line.replace(e, ' myemoneg ')
    line = line.strip()
    line = line.lower()
    line = line.replace("'", "")
    line = line.replace("?", " myquestionmark ")
    line = line.replace("!", " myexclamationmark ")
    line = re.sub('@([a-zA-Z0-9]|[_])*', ' ', line)
    line = re.sub('\\s?(f|ht)(tp)(s?)(://)([^\\.]*)[\\.|/](\\S*)', ' ', line)
    line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line)
    line = re.sub('[0-9]+', ' ', line)
    line = re.sub('([a-z])\\1+', '\\1\\1', line)
    line = line.split()
    return ' '.join(line)

def run(inpath, outpath):
    data = pd.read_csv(inpath, sep=',')
    data['tweet'] = data['tweet'].apply(process)
    data.to_csv(outpath, index=False)

if __name__ == '__main__':
    run(data_path, out_path)
