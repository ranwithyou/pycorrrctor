# -*- coding: utf-8 -*-

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))


row_train_path = os.path.join(pwd_path, '../../test_my.txt')

use_segment = True
segment_type = 'char'

output_dir = os.path.join(pwd_path, 'output2')
# 模型输出路径.
train_path = os.path.join(output_dir, 'train.txt')
model_dir = os.path.join(output_dir, 'models')
model_path = os.path.join(model_dir, 'model.pth')
vocab_path = os.path.join(model_dir, 'vocab.txt')

# 网络参数
word_embed_size = 200
hidden_size = 200
n_layers = 2
dropout = 0.5

#训练参数
epochs = 20
batch_size = 64
min_freq = 1
learning_rate = 1e-3

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
