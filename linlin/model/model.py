#!/usr/bin/env python

import pandas as pd
import numpy as np

data_path = '../../../data/processed/train.csv'
word_stat = '../../../data/processed/word_stat.txt'
word_stat2 = '../../../data/processed/word_stat2.txt'
word_stat3 = '../../../data/processed/word_stat3.txt'

def get_words(inpath):
    result = []
    with open(inpath, 'r') as inf:
        for line in inf:
            line = line.split(',')
            result.append(line[0].strip())
    return result

def get_phrase(tweet, p_len):
    result = []
    tweet_len = len(tweet)
    for i in range(tweet_len):
        if (i + p_len) > tweet_len:
            continue
        result.append('_'.join(tweet[i: i+p_len]))
    return result

def build_design_matrix_phrase(data, word_stat_file, p_len, num_word=1000):
    words = get_words(word_stat_file)[:num_word]
    result = pd.DataFrame(np.zeros([len(data), len(words)]), columns=words)
    for i in range(len(data)):
        tweet = str(data.iloc[i]).strip().split()
        tweet = get_phrase(tweet, p_len)
        freq = {}
        for wd in tweet:
            if wd not in freq:
                freq[wd] = 1
            else:
                freq[wd] += 1
        for wd in freq:
            result.iloc[i][wd] = freq[wd]
    return result

def build_design_matrix(data, num_phrase1=0, num_phrase2=0, num_phrase3=0):
    m_list = []
    if num_phrase1 > 0:
        m_list.append(build_design_matrix_phrase(data, word_stat, 1, num_phrase1))
    if num_phrase2 > 0:
        m_list.append(build_design_matrix_phrase(data, word_stat2, 2, num_phrase2))
    if num_phrase3 > 0:
        m_list.append(build_design_matrix_phrase(data, word_stat3, 3, num_phrase3))
    return pd.concat(m_list, axis=1)

def run(inpath):
    data = pd.read_csv(inpath, sep=',')
    train = data[:75]
    test = data[750000:]
    print build_design_matrix(train['tweet'], 100,20,2)


if __name__ == '__main__':
    run(data_path)
