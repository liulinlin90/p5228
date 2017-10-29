#!/usr/bin/env python
import os
import glob

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train_path = '../../../data/processed/train.csv'
test_path = '../../../data/processed/test.csv'
in_dir = ['../../../data/stage1_nb/',
        '../../../data/stage1a_nb/',
        '../../../data/stage1_xg/']

out_dir = '../../../data/stage2/'
train_range = (0, 800000)

def get_accuracy(pred, true_value):
    return 1 - np.sum(np.absolute(np.array(pred) - true_value)) / float(len(pred))


def load_stage1_result():
    filename = '{ftype}_{train_start}-{train_end}_*.csv'
    fparam = {'ftype' : 'train',
              'train_start': train_range[0],
              'train_end': train_range[1],
              }
    train = []
    test = []
    all_file = []
    for item in in_dir:
        all_file += glob.glob(os.path.join(item, filename.format(**fparam)))
    for f in all_file:
        result = []
        print 'loading file %s' % f
        with open(f) as inf:
            for line in inf:
                result.append(float(line))
        train.append(result)
        result = []
        print 'loading file %s' % f.replace('train', 'test')
        with open(f.replace('train', 'test')) as inf:
            for line in inf:
                result.append(float(line))
        test.append(result)
    return np.array(train).T,  np.array(test).T

def output_pred_result(x, model, ftype):
    filename = os.path.join(out_dir, '%s.csv' % ftype)
    pred = model.predict(x)
    with open(filename, 'w') as outf:
        print >> outf, 'id,prediction'
        for i in range(len(pred)):
            print >> outf, '%s,%s' % (i, 1 if pred[i] > 0.5 else 0)
    print 'generated ' + filename


def run_model(train, val, test, train_full):
    train_x, test_x = load_stage1_result()
    train_y = train['positive']
    #model = LogisticRegression()
    #model = RandomForestClassifier()
    #model = GaussianNB()
    model = XGBClassifier()
    print 'training model ...'
    model.fit(train_x, train_y)
    val_x = train_x[-1000:]
    val_y = val['positive']
    pred = model.predict(val_x)
    print get_accuracy(pred, val_y)
    output_pred_result(test_x, model,'test')
    output_pred_result(train_x, model,'train')

def run(train_p, test_p):
    data = pd.read_csv(train_p, sep=',')
    train = data[train_range[0]:train_range[1]]
    val = data[-1000:]
    test = pd.read_csv(test_p, sep=',')
    run_model(train, val, test, data)

if __name__ == '__main__':
    run(train_path, test_path)
