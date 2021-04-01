import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import pickle
import numpy as np
from keras.models import Model, load_model
from keras_bert.layers import TokenEmbedding, Extract
from keras_pos_embd import PositionEmbedding
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward
from keras_transformer import gelu
from keras_layer_normalization import LayerNormalization
from keras_bert import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras.layers import Dense, Input
from sklearn import metrics

try:
    os.remove('result.txt')
    os.remove('result_疑似ラベル抜きテストデータ.txt')
except:
    pass


# パラメータ読み込み
config = json.load(open('config.json'))
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LR = config['LR']
SEQ_LEN = config['SEQ_LEN']
test_units = config['test_units']
threshold = config['threshold']
choice_text_num = config['choice_text_num']
N1 = config['N1']
N2 = config['N2']

# データ読み込み
with open('../dataset/train/x_test.npy','rb') as f:
    x_test = pickle.load(f)
    x_test_moto = x_test
with open('../dataset/train/y_test.npy','rb') as f:
    y_test = pickle.load(f)
    y_test_moto = y_test

# BERT読み込み
pretrained_path = '../uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
bert = load_trained_model_from_checkpoint(config_path,checkpoint_path,training=True,seq_len=SEQ_LEN)
inputs = bert.inputs[:2]
dense = bert.get_layer('NSP-Dense').output
bert_model = Model(inputs, dense)

# 学習済みBERTモデルの読み込み
custom_objects = {'TokenEmbedding': TokenEmbedding,'PositionEmbedding': PositionEmbedding,'MultiHeadAttention': MultiHeadAttention,'FeedForward': FeedForward,'gelu': gelu,'Extract': Extract,'LayerNormalization': LayerNormalization}
model = load_model('BERT_pretraining.h5',custom_objects=custom_objects)

bert_cls_test_moto = bert_model.predict(x_test_moto)
y_pred = model.predict(bert_cls_test_moto)
y_pred_split = np.split(y_pred,len(y_test_moto))
y_pred_list = [np.argmax(x) for x in y_pred_split]
rep2 = metrics.classification_report(y_test_moto,y_pred_list,digits=3)
print(rep2)
with open('result.txt','a') as f:
        f.write(rep2)

for i in range(N1):
    # テストデータをベクトルに変換
    # 入力：[indices, indices_mask]　出力：CLSの768次元
    bert_cls_test = bert_model.predict(x_test)

    # テストデータで予測を行い信頼性の高い文書を選択
    y_pred = model.predict(bert_cls_test)
    y_pred_split = np.split(y_pred,len(y_test))
    y_pred_split = y_pred_split

    confidence = [[np.argmax(y),max(y)] for y in y_pred_split]
    confidence_sort = sorted(confidence,reverse=True,key=lambda x:x[1])[:choice_text_num]
    conf_index = np.argsort(np.squeeze([max(y) for y in y_pred_split]))[::-1][:choice_text_num]

    x_pseudo = [[],[]]
    y_pseudo = []
    for j in range(choice_text_num):
        if confidence_sort[j][1] >= threshold:
            x_pseudo[0].append(x_test[0][conf_index[j]*test_units+confidence_sort[j][0]])
            x_pseudo[1].append(x_test[1][conf_index[j]*test_units+confidence_sort[j][0]])
            y_pseudo.append(1)
            rand_base = [i for i in range(test_units)]
            rand_base.remove(confidence_sort[j][0])
            rand = np.random.choice(rand_base)
            x_pseudo[0].append(x_test[0][conf_index[j]*test_units+rand])
            x_pseudo[1].append(x_test[1][conf_index[j]*test_units+rand])
            y_pseudo.append(0)
        else:
            conf_index = conf_index[:j]
            break
    x_pseudo = [np.array(x_pseudo[0]),np.array(x_pseudo[1])]
    y_pseudo = np.array(y_pseudo)

    # 疑似ラベル付きデータをベクトル化
    bert_cls_pseudo = bert_model.predict(x_pseudo)

    # 疑似ラベル付きデータでBERTモデルを更新
    model.fit(bert_cls_pseudo,y_pseudo,epochs=EPOCHS,batch_size=BATCH_SIZE)

    model.save('update_model/BERT_update_'+str(i)+'.h5')

    # # テストデータから選択したデータを除外
    conf_index_sort = sorted(conf_index,reverse=True)
    x_test = [x_test[0].tolist(),x_test[1].tolist()]
    y_test = y_test.tolist()
    for j in conf_index_sort:
        for _ in range(test_units):
            x_test[0].pop(j*test_units)
            x_test[1].pop(j*test_units)
        y_test.pop(j)
    x_test = [np.array(x_test[0]),np.array(x_test[1])]
    y_test = np.array(y_test)

    # 疑似ラベルに使用したデータを除外したテストデータでテスト
    bert_cls_test = bert_model.predict(x_test)
    y_pred = model.predict(bert_cls_test)
    y_pred_split = np.split(y_pred,len(y_test))
    y_pred_list = [np.argmax(x) for x in y_pred_split]
    rep1 = metrics.classification_report(y_test,y_pred_list,digits=3)
    with open('result_疑似ラベル抜きテストデータ.txt','a') as f:
        f.write(rep1)

    # もとのテストデータでテスト
    bert_cls_test_moto = bert_model.predict(x_test_moto)
    y_pred = model.predict(bert_cls_test_moto)
    y_pred_split = np.split(y_pred,len(y_test_moto))
    y_pred_list = [np.argmax(x) for x in y_pred_split]
    rep2 = metrics.classification_report(y_test_moto,y_pred_list,digits=3)
    print(rep2)
    with open('result.txt','a') as f:
        f.write(rep2)

