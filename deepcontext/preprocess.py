import os

from pycorrector1.utils.tokenizer import segment
import config


def get_data_file(path, use_segment, segment_type):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue
            target = ' '.join(segment(parts[1].strip(), cut_type=segment_type)) if use_segment else parts[1].strip()
            data_list.append(target)
            # target = ' '.join(segment(line, cut_type=segment_type))
            # data_list.append(target)
    return data_list


def save_data(data_list, data_path):
    dirname = os.path.dirname(data_path)
    os.makedirs(dirname, exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for line in data_list:
            f.write(line + '\n')
            count += 1
        print("保存数据的大小为:%d to %s" % (count, data_path))


if __name__ == '__main__':
    # 数据预处理
    data_list = []
    data = get_data_file(config.row_train_path, config.use_segment, config.segment_type)
    data_list.extend(data)
    # 保存数据
    save_data(data_list, config.train_path)
