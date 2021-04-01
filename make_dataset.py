import csv
import os
import glob
import re
import string
import numpy as np
from keras_bert import Tokenizer
import codecs
from tqdm import tqdm
import json
import pickle

np.random.seed(0)

config = json.load(open('config.json','r'))
SEQ_LEN = config['SEQ_LEN']
train_units = config['train_units']
test_units = config['test_units']

pretrained_path = '../uncased_L-12_H-768_A-12'
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# 特殊文字除去
def chenge_text(text):
    text = text.translate(str.maketrans( '', '',string.punctuation))
    return text

# BERTの辞書
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# texts:[文章, クラス名]
def load_data(texts):
    tokenizer = Tokenizer(token_dict)
    indices = []
    indices_mask = []
    for text in tqdm(texts):
        ids,masked_ids = tokenizer.encode(text[0],text[1],max_len=SEQ_LEN)
        indices.append(ids)
        indices_mask.append(masked_ids)
    indices = np.array(indices)
    indices_mask = np.array(indices_mask)
    return [indices, indices_mask]

# yahoo topic class
with open('../data/yahootopic/classes.txt','r',encoding='utf-8') as f:
    reader = f.read().splitlines()
    yahoo_class = []
    for i,row in enumerate(reader):
        texts = ' '.join(['this text is about '+r+' .' for r in row.split(' & ')])
        yahoo_class.append(texts)

# yahoo topicのtrain_v0
with open('../data/yahootopic/train_pu_half_v0.txt','r',encoding='utf-8') as f:
    v0 = f.read().splitlines()
train = []
y_train = []
train_rand = []
for i,text in tqdm(enumerate(v0),total=len(v0)):
    text = text.split('\t')
    train.append([text[1],yahoo_class[int(text[0])]])
    y_train.append(1)

    rand_base = list(range(train_units))
    rand_base.remove(int(int(text[0])/2))
    rand = np.random.choice(rand_base)
    train.append([text[1],yahoo_class[rand*2]])
    y_train.append(0)

x_train = load_data(train)
y_train = np.array(y_train)

with open('../dataset/train/x_train.npy','wb') as f:
    pickle.dump(x_train, f, protocol=4)
with open('../dataset/train/y_train.npy','wb') as f:
    pickle.dump(y_train, f, protocol=4)

# testdataのv1
with open('../data/yahootopic/test.txt','r',encoding='utf-8') as f:
    yahoo_test = f.read().splitlines()
yahoo_test_v1 = [y for y in yahoo_test if int(y[0])%2==1]

test = []
y_test = []
for texts in tqdm(yahoo_test_v1,total=len(yahoo_test_v1)):
    text = texts.split('\t')
    for i in range(test_units):
        test.append([text[1],yahoo_class[i*2+1]]) # 要確認
    y_test.append(int(int(text[0])/2))

x_test = load_data(test)
y_test = np.array(y_test)

with open('../dataset/train/x_test.npy','wb') as f:
    pickle.dump(x_test, f, protocol=4)
with open('../dataset/train/y_test.npy','wb') as f:
    pickle.dump(y_test, f, protocol=4)