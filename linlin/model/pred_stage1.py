#!/usr/bin/env python
import os

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train_path = '../../../data/processed/train.csv'
test_path = '../../../data/processed/test.csv'
out_dir = '../../../data/stage1/'
word_stat = './word_stat.txt'
word_stat2 = './word_stat2.txt'
word_stat3 = './word_stat3.txt'
train_range = (0, 800000)

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
        result.append('_'.join(tweet[i: i + p_len]))
    return result


def build_design_matrix_phrase(data, word_stat_file, p_len, word_range=(0,1000)):
    words = get_words(word_stat_file)[word_range[0]:word_range[1]]
    result = pd.DataFrame(np.zeros([len(data), len(words)]), columns=words)
    for i in range(len(data)):
        tweet = str(data.iloc[i]).strip().split()
        tweet = get_phrase(tweet, p_len)
        freq = {}
        for wd in tweet:
            if wd not in words:
                continue
            if wd not in freq:
                freq[wd] = 1
            else:
                freq[wd] += 1
        for wd in freq:
            result.iloc[i][wd] = freq[wd]
    return result


def build_design_matrix(data, phrase1_range, phrase2_range, phrase3_range):
    m_list = []
    if phrase1_range is not None:
        m_list.append(build_design_matrix_phrase(data, word_stat, 1, phrase1_range))
    if phrase2_range is not None:
        m_list.append(build_design_matrix_phrase(data, word_stat2, 2, phrase2_range))
    if phrase3_range is not None:
        m_list.append(build_design_matrix_phrase(data, word_stat3, 3, phrase3_range))
    return pd.concat(m_list, axis=1)


def model_nb(train_x, train_y):
    model = GaussianNB()
    scores = cross_val_score(model, train_x, train_y, cv=3)
    return model, scores


def model_svc(train_x, train_y):
    model = SVC()
    scores = cross_val_score(model, train_x, train_y, cv=3)
    return model, scores

def model_rf(train_x, train_y):
    model = RandomForestClassifier(max_depth=10, random_state=0)
    scores = cross_val_score(model, train_x, train_y, cv=3)
    return model, scores

def get_accuracy(pred, true_value):
    return 1 - np.sum(np.absolute(np.array(pred) - true_value)) / float(len(pred))


def gen_tuples(max_val, tp_len):
    result = []
    for i in range(tp_len, max_val + tp_len, tp_len):
        result.append((i-tp_len, i))
    return result


def output_pred_result(x, model, p1num, p2num, p3num, ftype, wmod):
    filename = '{ftype}_{train_start}-{train_end}_{p1start}-{p1end}_{p2start}-{p2end}_{p3start}-{p3end}.csv'
    fparam = {'ftype' : ftype,
              'train_start': train_range[0],
              'train_end': train_range[1],
              'p1start': p1num[0] if p1num else 0,
              'p1end': p1num[1] if p1num else 0,
              'p2start': p2num[0] if p2num else 0,
              'p2end': p2num[1] if p2num else 0,
              'p3start': p3num[0] if p3num else 0,
              'p3end': p3num[1] if p3num else 0,
              }
    pred_file = os.path.join(out_dir, filename.format(**fparam))
    print 'writing prediction file %s' % pred_file
    with open(pred_file, wmod) as outf:
        for p in model.predict(x):
            print >> outf, str(p)


def run_model(train, val, test, train_full, p1num, p2num, p3num):
    train_x = build_design_matrix(train['tweet'], p1num, p2num, p3num)
    train_y = train['positive']
    model = GaussianNB()
    print 'training model ...'
    model.fit(train_x, train_y)
    val_x = build_design_matrix(val['tweet'], p1num, p2num, p3num)
    val_y = val['positive']
    pred = model.predict(val_x)
    print get_accuracy(pred, val_y), p1num, p2num, p3num
    del train_x
    del train_y
    del val_x
    del val_y
    for drange in gen_tuples(200000, 10000):
        dstart, dend = drange
        print 'predict range %d to %d ...' % (dstart, dend)
        test_x = build_design_matrix(test[dstart:dend]['tweet'], p1num, p2num, p3num)
        if dstart == 0:
            wmod='w'
        else:
            wmod='a'
        output_pred_result(test_x, model, p1num, p2num, p3num, 'test', wmod)
    for drange in gen_tuples(800000, 10000):
        dstart, dend = drange
        print 'predict range %d to %d ...' % (dstart, dend)
        train_full_x = build_design_matrix(train_full[dstart:dend]['tweet'], p1num, p2num, p3num)
        if dstart == 0:
            wmod='w'
        else:
            wmod='a'
        output_pred_result(train_full_x, model, p1num, p2num, p3num, 'train', wmod)


def run(train_p, test_p):
    data = pd.read_csv(train_p, sep=',')
    train = data[train_range[0]:train_range[1]]
    val = data[-1000:]
    test = pd.read_csv(test_p, sep=',')
    for p1num in gen_tuples(5000, 500):
        p2num, p3num = None, None
        run_model(train, val, test, data, p1num, p2num, p3num)

    for p2num in gen_tuples(1000, 500):
        p1num, p3num = None, None
        run_model(train, val, test, data, p1num, p2num, p3num)

    for p3num in gen_tuples(300, 100):
        p1num, p2num = None, None
        run_model(train, val, test, data, p1num, p2num, p3num)

if __name__ == '__main__':
    run(train_path, test_path)
