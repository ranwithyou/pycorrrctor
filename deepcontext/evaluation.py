import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
test_path = os.path.join(pwd_path, '../../test_my.txt')


def eval_by_model(correct_fn, test_path=test_path, verbose=True):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            src = parts[0]
            tgt = parts[1]

            tgt_pred, pred_detail = correct_fn(src)
            if verbose:
                print()
                print('input  :', src)
                print('truth  :', tgt)
                print('predict:', tgt_pred, pred_detail)

            # 负样本
            if src == tgt:
                # 预测也为负
                if tgt == tgt_pred:
                    TN += 1
                    print('right')
                # 预测为正
                else:
                    FP += 1
                    print('wrong')
            # 正样本
            else:
                # 预测也为正
                if tgt == tgt_pred:
                    TP += 1
                    print('right')
                # 预测为负
                else:
                    FN += 1
                    print('wrong')
            total_num += 1
        acc = (TP + TN) / total_num
        precision = TP / (TP + FP) if TP > 0 else 0.0
        recall = TP / (TP + FN) if TP > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(
            f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, s')
        return acc, precision, recall, f1


if __name__ == "__main__":
    from pycorrector1.deepcontext.infer import Inference
    from pycorrector1.deepcontext import config

    model = Inference(config.model_dir, config.vocab_path)
    eval_by_model(model.predict)
