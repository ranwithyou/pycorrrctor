# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


class NegativeSampling(nn.Module):
    def __init__(self,
                 embed_size,
                 counter,
                 n_negatives,
                 power,
                 device,
                 ignore_index):
        super(NegativeSampling, self).__init__()

        self.counter = counter
        self.n_negatives = n_negatives
        self.power = power
        self.device = device

        self.W = nn.Embedding(num_embeddings=len(counter),
                              embedding_dim=embed_size,
                              padding_idx=ignore_index)
        self.W.weight.data.zero_()
        self.logsigmoid = nn.LogSigmoid()
        self.sampler = WalkerAlias(np.power(counter, power))

    def negative_sampling(self, shape):
        if self.n_negatives > 0:
            return torch.tensor(self.sampler.sample(shape=shape), dtype=torch.long, device=self.device)
        else:
            raise NotImplementedError

    def forward(self, sentence, context):
        batch_size, seq_len = sentence.size()
        emb = self.W(sentence)
        pos_loss = self.logsigmoid((emb * context).sum(2))

        neg_samples = self.negative_sampling(shape=(batch_size, seq_len, self.n_negatives))
        neg_emb = self.W(neg_samples)
        neg_loss = self.logsigmoid((-neg_emb * context.unsqueeze(2)).sum(3)).sum(2)
        return -(pos_loss + neg_loss).sum()


class WalkerAlias(object):
    """
    This is from Chainer's implementation.
    You can find the original code at
    https://github.com/chainer/chainer/blob/v4.4.0/chainer/utils/walker_alias.py
    This class is
        Copyright (c) 2015 Preferred Infrastructure, Inc.
        Copyright (c) 2015 Preferred Networks, Inc.
    """

    def __init__(self, probs):
        prob = np.array(probs, np.float32)
        # 归一化
        prob /= np.sum(prob)
        # 词典大小
        threshold = np.ndarray(len(probs), np.float32)
        # 大小为两倍的词典大小
        # 每列中最多只放两个事件的思想
        values = np.ndarray(len(probs) * 2, np.int32)
        il, ir = 0, 0
        pairs = list(zip(prob, range(len(probs))))
        # 按照prob的值从小到大排序
        pairs.sort()
        for prob, i in pairs:
            p = prob * len(probs) # 按照其均值归一化, (除以1/N, 即乘以N).
            # p>1, 说明当前列需要被截断.
            # 回填的思想, 如果当前的概率值大于均值的话, 就将遍历之前的ir到il之间没有填满的坑.
            # 主要是为了构造一个1*N的矩阵.
            while p > 1 and ir < il:
                values[ir * 2 + 1] = i# 为了填充没有满的那一列ir, 并且将索引保存到奇数列.
                p -= 1.0 - threshold[ir]# 本列一共减少了(1-threshold[ir])的概率值.
                ir += 1# ir位置的坑已经被填满,故ir+=1.
            threshold[il] = p# 概率值*词典大小.
            values[il * 2] = i# 保存单词的索引, 偶数列.
            il += 1
        # fill the rest
        for i in range(ir, len(probs)):
            values[i * 2 + 1] = 0

        assert ((values < len(threshold)).all())
        self.threshold = threshold
        self.values = values

    def sample(self, shape):
        ps = np.random.uniform(0, 1, shape)# 从均匀分布中抽取样本.
        pb = ps * len(self.threshold)# 均值归一化, (除以1/N, 即乘以N).
        index = pb.astype(np.int32) #转化为int类型, 可以认为index对应着词典中某个词的索引.
        left_right = (self.threshold[index] < pb - index).astype(np.int32)# 选择是奇数列还是偶数列, 注意, (pb - index) 返回的是0-1之间的数组,
        return self.values[index * 2 + left_right]
