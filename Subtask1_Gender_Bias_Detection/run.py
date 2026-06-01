import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif



if __name__ == '__main__':
    dataset = 'data'
    # model_name = 'simcse-roberta-lstm'  # bert   ERNIE  roberta roberta-chinese roberta_chinese_large
    # model_name = 'bert'
    # model_name = 'ERNIE'
    # model_name = 'roberta-chinese'
    model_name = 'roberta_chinese_large'
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)


    ### 运行脚本
# python run.py
    ### 实验结果
# bert
# No optimization for a long time, auto-stopping...
# Test Loss:   0.5,  Test Acc: 79.31%
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#        False     0.7313    0.9267    0.8175       464
#         True     0.9000    0.6595    0.7612       464

#     accuracy                         0.7931       928
#    macro avg     0.8156    0.7931    0.7893       928
# weighted avg     0.8156    0.7931    0.7893       928

# Confusion Matrix...
# [[430  34]
#  [158 306]]
# Time usage: 0:00:01
# ERNIE
# No optimization for a long time, auto-stopping...
# Test Loss:  0.56,  Test Acc: 72.63%
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#        False     0.6601    0.9332    0.7732       464
#         True     0.8860    0.5194    0.6549       464

#     accuracy                         0.7263       928
#    macro avg     0.7730    0.7263    0.7141       928
# weighted avg     0.7730    0.7263    0.7141       928

# Confusion Matrix...
# [[433  31]
#  [223 241]]
# Time usage: 0:00:01
# roberta-chinese
# Test Loss:  0.52,  Test Acc: 75.22%
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#        False     0.7294    0.8017    0.7639       464
#         True     0.7799    0.7026    0.7392       464

#     accuracy                         0.7522       928
#    macro avg     0.7547    0.7522    0.7515       928
# weighted avg     0.7547    0.7522    0.7515       928

# Confusion Matrix...
# [[372  92]
#  [138 326]]
# Time usage: 0:00:04
# roberta_chinese_large
# No optimization for a long time, auto-stopping...
# Test Loss:  0.47,  Test Acc: 77.69%
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#        False     0.7219    0.9009    0.8015       464
#         True     0.8682    0.6530    0.7454       464

#     accuracy                         0.7769       928
#    macro avg     0.7951    0.7769    0.7735       928
# weighted avg     0.7951    0.7769    0.7735       928

# Confusion Matrix...
# [[418  46]
#  [161 303]]
# Time usage: 0:00:04