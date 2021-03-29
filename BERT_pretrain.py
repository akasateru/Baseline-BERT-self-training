import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import csv
import codecs
import json
import pickle
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras_bert.backend import keras
from keras_bert.layers import TokenEmbedding, Extract
from keras_pos_embd import PositionEmbedding
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras_position_wise_feed_forward import FeedForward
from keras_multi_head import MultiHeadAttention
from keras_transformer import gelu
from sklearn import metrics
from keras_layer_normalization import LayerNormalization
import tensorflow as tf

# パラメータの読み込み
config = json.load(open('config.json'))
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LR = config['LR']
SEQ_LEN = config['SEQ_LEN']
BERT_DIM = config['BERT_DIM']

# BERTの読み込み
pretrained_path = '../uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
bert = load_trained_model_from_checkpoint(config_path,checkpoint_path,training=True,seq_len=SEQ_LEN)

# 学習データ読み込み
with open('../dataset/train/x_train.npy','rb') as f:
    x_train = pickle.load(f)
with open('../dataset/train/y_train.npy','rb') as f:
    y_train = pickle.load(f)

# ベクトル化
inputs = bert.inputs[:2]
dense = bert.get_layer('NSP-Dense').output
bert_model = Model(inputs, dense)
bert_cls = bert_model.predict(x_train)
print(bert_cls.shape)
print(y_train.shape)

# 学習
inputs = Input(shape=(768,))
output = Dense(units=1, activation='sigmoid')(inputs)
model = Model(inputs, output)

model.summary()
model.compile(optimizer=Adam(beta_1=0.9,beta_2=0.999),loss='binary_crossentropy',metrics=['acc'])
model.fit(bert_cls,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE)

model.save('BERT_pretraining.h5')