set -e
time python make_dataset.py
time python BERT_pretrain.py
time python BERT+self-training.py