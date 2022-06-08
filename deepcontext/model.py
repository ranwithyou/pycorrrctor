import math
import torch
import torch.nn as nn
from .loss import NegativeSampling
import numpy as np

class Context2vec(nn.Module):
    def __init__(self,
                 vocab_size,
                 counter,
                 word_embed_size,
                 hidden_size,
                 n_layers,
                 use_mlp,
                 dropout,
                 pad_index,
                 device,
                 is_inference):

        super(Context2vec, self).__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_mlp = use_mlp
        self.device = device
        self.is_inference = is_inference
        self.rnn_output_size = hidden_size

        self.drop = nn.Dropout(dropout)
        self.l2r_emb = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=word_embed_size,
                                    padding_idx=pad_index)

        self.l2r_rnn = nn.GRU(input_size=word_embed_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              batch_first=True)
        self.r2l_emb = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=word_embed_size,
                                    padding_idx=pad_index)
        self.r2l_rnn = nn.GRU(input_size=word_embed_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              batch_first=True)
        self.criterion = NegativeSampling(hidden_size,
                                          counter,
                                          ignore_index=pad_index,
                                          n_negatives=10,
                                          power=0.75,
                                          device=device)

        if use_mlp:
            self.MLP = MLP(input_size=hidden_size * 2,
                           mid_size=hidden_size * 2,
                           output_size=hidden_size,
                           dropout=dropout)
        else:
            self.weights = nn.Parameter(torch.zeros(2, hidden_size))
            self.gamma = nn.Parameter(torch.ones(1))

        self.init_weights()

    def init_weights(self):
        std = math.sqrt(1. / self.word_embed_size)
        self.r2l_emb.weight.data.normal_(0, std)
        self.l2r_emb.weight.data.normal_(0, std)

    def forward(self, sentences, target, target_pos=None):
        reversed_sentences = sentences.flip(1)[:, :-1]
        sentences = sentences[:, :-1]
        # sentences = np.array(sentences.numpy())
        # id_to_vocab ={}
        # with open('./output2/models/vocab.txt','r', encoding="UTF_8") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line1 = line.strip('\n').split('\t')
        #         id_to_vocab[int(line1[1])] = str(line1[0])
        # print(id_to_vocab)
        # m = len(sentences)
        # n = len(sentences[0])
        # li =[]
        # li2 = []
        # result = []
        # for i in range(0, m):
        #     for j in range(0, n):
        #         if sentences[i][j]==0 or sentences[i][j]==1 or sentences[i][j]==2 or sentences[i][j]==3:
        #             continue
        #         else:
        #             li.append(id_to_vocab.get(sentences[i][j]))
        #     result.append([(''.join(li))])
        # print(result)

        l2r_emb = self.l2r_emb(sentences)
        r2l_emb = self.r2l_emb(reversed_sentences)
        # print(l2r_emb)

        output_l2r, _ = self.l2r_rnn(l2r_emb)
        output_r2l, _ = self.r2l_rnn(r2l_emb)
        # print(output_l2r)

        output_l2r = output_l2r[:, :-1, :]
        output_r2l = output_r2l[:, :-1, :].flip(1)

        if self.is_inference:
            if self.use_mlp:
                output_l2r = output_l2r[0, target_pos]
                output_r2l = output_r2l[0, target_pos]
                # 根据第0个维度进行拼接, 保持第1和第2维度一致, 此处是为了推断出目标词.
                c_i = self.MLP(torch.cat((output_l2r, output_r2l), dim=0))
            return c_i
        else:
            # on a training phase
            if self.use_mlp:
                # 根据第2个维度进行拼接, 保持第0和第1维度一致. (batch_size, seq_length, hidden_size*2)
                c_i = self.MLP(torch.cat((output_l2r, output_r2l), dim=2))
            else:
                s_task = torch.nn.functional.softmax(self.weights, dim=0)
                c_i = torch.stack((output_l2r, output_r2l), dim=2) * s_task
                c_i = self.gamma * c_i.sum(2)
            # target: (batch_size, seq_len)
            # c_i: (batch_size, seq_len, hidden_size*2)
            loss = self.criterion(target, c_i)
            return loss

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_size))

    def run_inference(self, input_tokens, target, target_pos, k=10):
        context_vector = self.forward(input_tokens, target=None, target_pos=target_pos)
        if target is None:
            topv, topi = ((self.criterion.W.weight * context_vector).sum(dim=1)).data.topk(k)
            return topv, topi
        else:
            context_vector /= torch.norm(context_vector, p=2)
            target_vector = self.criterion.W.weight[target]
            target_vector /= torch.norm(target_vector, p=2)
            similarity = (target_vector * context_vector).sum()
            return similarity.item()

    def norm_embedding_weight(self, embedding_module):
        embedding_module.weight.data /= torch.norm(embedding_module.weight.data, p=2, dim=1, keepdim=True)
        # replace NaN with zero
        embedding_module.weight.data[embedding_module.weight.data != embedding_module.weight.data] = 0


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 mid_size,
                 output_size,
                 n_layers=2,
                 dropout=0.3,
                 activation_function='relu'):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.mid_size = mid_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop = nn.Dropout(dropout)

        self.MLP = nn.ModuleList()
        if n_layers == 1:
            self.MLP.append(nn.Linear(input_size, output_size))
        else:
            self.MLP.append(nn.Linear(input_size, mid_size))
            for _ in range(n_layers - 2):
                self.MLP.append(nn.Linear(mid_size, mid_size))
            self.MLP.append(nn.Linear(mid_size, output_size))

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x
        for i in range(self.n_layers - 1):
            out = self.MLP[i](self.drop(out))
            out = self.activation_function(out)
        return self.MLP[-1](self.drop(out))
