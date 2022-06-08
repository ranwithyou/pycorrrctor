import operator
import os
import time


import torch
from torch import optim

from pycorrector1.deepcontext.model import Context2vec
from pycorrector1.deepcontext.data_reader import read_config, load_word_dict
from pycorrector1.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector1.utils.tokenizer import split_text_by_maxlen
from pycorrector1.corrector import Corrector
from pycorrector1.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference(Corrector):
    def __init__(self, model_dir, vocab_path):
        super(Inference, self).__init__()
        self.name = 'deepcontext_corrector'
        t1 = time.time()
        # device
        logger.debug("device: {}".format(device))
        model, config_dict = self._read_model(model_dir)
        # norm weight
        model.norm_embedding_weight(model.criterion.W)
        self.model = model
        self.model.eval()

        self.unk_token, self.sos_token, self.eos_token, self.pad_token, self.itos, self.stoi = self._get_config_data(
            config_dict, vocab_path)
        self.model_dir = model_dir
        self.vocab_path = vocab_path
        self.mask = "[]"
        logger.debug('Loaded deep context model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    @staticmethod
    def _read_model(model_dir):
        config_file = os.path.join(model_dir, 'config.json')
        config_dict = read_config(config_file)
        model = Context2vec(vocab_size=config_dict['vocab_size'],
                            counter=[1] * config_dict['vocab_size'],
                            word_embed_size=config_dict['word_embed_size'],
                            hidden_size=config_dict['hidden_size'],
                            n_layers=config_dict['n_layers'],
                            use_mlp=config_dict['use_mlp'],
                            dropout=config_dict['dropout'],
                            pad_index=config_dict['pad_index'],
                            device=device,
                            is_inference=True
                            ).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu'))
        optimizer = optim.Adam(model.parameters(), lr=config_dict['learning_rate'])
        optimizer.load_state_dict(torch.load(os.path.join(model_dir, 'model_optimizer.pth'), map_location='cpu'))
        return model, config_dict

    @staticmethod
    def _get_config_data(config_dict, vocab_path):
        # load model
        unk_token = config_dict['unk_token']
        sos_token = config_dict['sos_token']
        eos_token = config_dict['eos_token']
        pad_token = config_dict['pad_token']

        # read vocab
        stoi = load_word_dict(vocab_path)
        itos = {v: k for k, v in stoi.items()}

        return unk_token, sos_token, eos_token, pad_token, itos, stoi

    def predict_mask_token(self, tokens, mask_index, k=10):
        pred_words = []
        tokens[mask_index] = self.unk_token
        tokens = [self.sos_token] + tokens + [self.eos_token]
        indexed_sentence = [self.stoi[token] if token in self.stoi else self.stoi[self.unk_token] for token in tokens]
        input_tokens = torch.tensor(indexed_sentence, dtype=torch.long, device=device).unsqueeze(0)
        topv, topi = self.model.run_inference(input_tokens, target=None, target_pos=mask_index, k=k)
        print(topv, topi)
        for value, key in zip(topv, topi):
            score = value.item()
            word = self.itos[key.item()]
            print(score, word)
            if word in [self.unk_token, self.sos_token, self.eos_token, self.pad_token]:
                continue
            pred_words.append((word, score))
        # print(pred_words)
        return pred_words

    def predict(self, text, **kwargs):
        details = []
        text_new = ''
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        print(blocks)
        for blk, start_idx in blocks:
            blk_new = ''
            for idx, s in enumerate(blk):
                # 处理中文错误
                if is_chinese_string(s):
                    sentence_lst = list(blk_new + blk[idx:])
                    # print(sentence_lst)
                    sentence_lst[idx] = self.mask
                    print(sentence_lst)
                    # 预测，默认取top10
                    predict_words = self.predict_mask_token(sentence_lst, idx, k=10)
                    print(predict_words)
                    top_tokens = []
                    for w, _ in predict_words:
                        top_tokens.append(w)
                    # print(top_tokens)
                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        print(candidates)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append((s, token_str, start_idx + idx, start_idx + idx + 1))
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    from pycorrector1.deepcontext import config

    sents = ["下个星期，我跟我朋唷打算去法国玩儿。",
             "对不气，最近我很忙，所以我不会去妳的。",
             "真麻烦你了。希望你们好好的跳无。"]

    inference = Inference(config.model_dir, config.vocab_path)
    for i in sents:
        r = inference.predict(i)
        print(i, r)
