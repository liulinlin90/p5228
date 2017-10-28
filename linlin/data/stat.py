#!/usr/bin/env python

import operator

import pandas as pd

data_path = '../../../data/processed/train.csv'

def get_avg_freq(wdict):
    llen = len(wdict)
    for wd in wdict.keys():
        wdict[wd] = wdict[wd] / float(llen)
    return wdict

def get_phrase(tweet, p_len):
    result = []
    tweet_len = len(tweet)
    for i in range(tweet_len):
        if (i + p_len) > tweet_len:
            continue
        result.append('_'.join(tweet[i: i+p_len]))
    return result

def run(inpath, p_len=1):
    data = pd.read_csv(inpath, sep=',')[:750000]
    pos_count = {}
    neg_count = {}
    for i in range(len(data)):
        pos = data.iloc[i]['positive']
        tweet = str(data.iloc[i]['tweet']).strip().split()
        doc_freq = {}
        if pos == 1:
            for wd in get_phrase(tweet, p_len):
                if wd not in doc_freq:
                    doc_freq[wd] = 1
                elif doc_freq[wd] < 10:
                    doc_freq[wd] += 1
                else:
                    continue
                if wd not in pos_count:
                    pos_count[wd] = 1
                else:
                    pos_count[wd] += 1
        else:
            for wd in get_phrase(tweet, p_len):
                if wd not in doc_freq:
                    doc_freq[wd] = 1
                elif doc_freq[wd] < 10:
                    doc_freq[wd] += 1
                else:
                    continue
                if wd not in neg_count:
                    neg_count[wd] = 1
                else:
                    neg_count[wd] += 1
    pos_count = get_avg_freq(pos_count)
    neg_count = get_avg_freq(neg_count)

    allwd = set( item for item in (pos_count.keys() + neg_count.keys()))
    diff = {}
    for wd in allwd:
        diff[wd] = abs(pos_count.get(wd, 0) - neg_count.get(wd, 0))
    s_diff = sorted(diff.items(), key=operator.itemgetter(1), reverse=True)
    for item in s_diff:
        print ','.join(map(str, [item[0], item[1], pos_count.get(item[0]), neg_count.get(item[0])]))


if __name__ == '__main__':
    run(data_path, 3)
