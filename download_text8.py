import urllib.request
import shutil
import zipfile
import os
import pickle
from collections import Counter
import numpy as np

url = 'http://mattmahoney.net/dc/text8.zip'
filename = 'text8.zip'
train_size = 99000000

if not os.path.isfile(filename):
    print('Downloading text8 dataset...')

    with urllib.request.urlopen(url) as response, \
        open(filename, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)

rawdata = zipfile.ZipFile(filename).read('text8').decode('utf-8')

train_split = rawdata[:train_size].split()
valid_split = rawdata[train_size:].split()

vocab = Counter()

print('Constructing dictionary...')

for word in train_split:
    vocab[word] += 1

vocab_cut = {k: v for k, v in vocab.items() if v > 10}
vocab_sorted = sorted(vocab_cut.items(), key=lambda x: x[1], reverse=True)
wordmap = {k: id + 1 for id, (k, _) in enumerate(vocab_sorted)}

def save_pickle(split, wordmap, filename):
    data = []

    for word in split:
        try:
            data.append(wordmap[word])

        except KeyError:
            data.append(0)

    data = np.array(data)
    data_cut = data[:data.shape[0] // (128 * 20) * 20 * 128]

    data_next = data_cut.copy()
    data_next[:-1] = data_cut[1:]
    data_next[-1] = data_cut[0]

    input = np.array_split(data_cut.reshape((128, -1)),
                           data_cut.shape[0] / 128 / 20, axis=1)
    label = np.array_split(data_next.reshape((128, -1)),
                           data_next.shape[0] / 128 / 20, axis=1)

    with open(filename, 'wb') as f:
        pickle.dump({'input': input, 'label': label, 'worddic': wordmap}, f)

print('Writing train split...')
save_pickle(train_split, wordmap, 'text8.train.pkl')

print('Writing valid split...')
save_pickle(valid_split, wordmap, 'text8.test.pkl')
