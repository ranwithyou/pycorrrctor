
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# -----用户目录，存储模型文件-----
USER_DATA_DIR = os.path.expanduser('~/.pycorrector/datasets/')
os.makedirs(USER_DATA_DIR, exist_ok=True)
language_model_path = os.path.join(USER_DATA_DIR, 'zh_giga.no_cna_cmn.prune01244.klm')

# -----词典文件路径-----
# 通用分词词典文件  format: 词语 词频
word_freq_path = os.path.join(pwd_path, 'data/word_freq.txt')
# 中文常用字符集
common_char_path = os.path.join(pwd_path, 'data/common_char_set.txt')
# 同音字
same_pinyin_path = os.path.join(pwd_path, 'data/same_pinyin.txt')
# 形似字
same_stroke_path = os.path.join(pwd_path, 'data/same_stroke.txt')


