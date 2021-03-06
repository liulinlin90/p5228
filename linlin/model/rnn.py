from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
import os,csv
import numpy as np

max_features = 20480
maxlen = 40  # cut texts after this number of words (among top max_features most common words)
batch_size = 64

print('Loading data...')

x_txt = []
y_txt = []
with open('../../../data/processed/train.csv') as f:
    f.next()
    reader = csv.reader(f)  # remove fist line
    for input_row in reader:
        x_txt.append(input_row[2])
        y_txt.append(int(input_row[1]))

x_input = []
with open('../../../data/processed/test.csv') as f:
    f.next()
    reader = csv.reader(f)
    for input_row in reader:
        x_input.append(input_row[1])


tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_txt[0:800000])
x_train = tokenizer.texts_to_sequences(x_txt[0:750000])
y_train = y_txt[0:750000]

x_test = tokenizer.texts_to_sequences(x_txt[750000:800000])
y_test = y_txt[750000:800000]
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.25))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=2,
          validation_data=(x_test, y_test),
          shuffle=True)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

x_input = tokenizer.texts_to_sequences(x_input[0:200000])
x_input = sequence.pad_sequences(x_input, maxlen=maxlen)
output = model.predict(x_input, batch_size=100)
output = (output >= 0.50)
np.savetxt('./test_rnn_output.csv', output.ravel(), fmt='%d')

x_txt = tokenizer.texts_to_sequences(x_txt[0:800000])
x_txt = sequence.pad_sequences(x_txt, maxlen=maxlen)
output = model.predict(x_txt, batch_size=100)
output = (output >= 0.50)
np.savetxt('./train_rnn_output.csv', output.ravel(), fmt='%d')


