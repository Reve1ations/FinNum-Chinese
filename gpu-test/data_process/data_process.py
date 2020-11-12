from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pickle
import yaml
from data_process.utils import *

def data_process(config):
    mode = config['mode']
    assert mode in ('term', 'category')  # 判断任务是否为 term或category ,条件为 false 时触发异常
    base_path = config['base_path']  # 数据集路径
    raw_train_path = os.path.join(base_path, 'raw/train.xml')
    raw_val_path = os.path.join(base_path, 'raw/dev.xml')
    raw_test_path = os.path.join(base_path, 'raw/test.xml')
    lowercase = config['lowercase']

    # utils.py文件，处理数据
    if mode == 'term':
        train_data = parse_sentence_term(raw_train_path, lowercase=lowercase)
        val_data = parse_sentence_term(raw_val_path, lowercase=lowercase)
        test_data = parse_sentence_term(raw_test_path, lowercase=lowercase)
    else:
        train_data = parse_sentence_category(raw_train_path, lowercase=lowercase)
        val_data = parse_sentence_category(raw_val_path, lowercase=lowercase)
        test_data = parse_sentence_category(raw_test_path, lowercase=lowercase)

    # 过滤数据
    remove_list = ['conflict']
    train_data = category_filter(train_data, remove_list)
    val_data = category_filter(val_data, remove_list)
    # 获取word2index,index2word字典
    word2index, index2word = build_vocab(train_data, max_size=config['max_vocab_size'], min_freq=config['min_vocab_freq'])

    # 判断是否存在目录./data/MAMS-ATSA/processed；若无，则创建递归的目录树
    if not os.path.exists(os.path.join(base_path, 'processed')):
        os.makedirs(os.path.join(base_path, 'processed'))
    if mode == 'term':
        save_term_data(train_data, word2index, os.path.join(base_path, 'processed/train.npz'))
        save_term_data(val_data, word2index, os.path.join(base_path, 'processed/dev.npz'))
        save_term_data(test_data, word2index, os.path.join(base_path, 'processed/test.npz'))
    else:
        save_category_data(train_data, word2index, os.path.join(base_path, 'processed/train.npz'))
        save_category_data(val_data, word2index, os.path.join(base_path, 'processed/dev.npz'))
        save_category_data(test_data, word2index, os.path.join(base_path, 'processed/test.npz'))

    with open(os.path.join(base_path, 'processed/word2index.pickle'), 'wb') as handle:
        pickle.dump(word2index, handle)  # 序列化对象，将对象word2index,保存到文件handle中去
    with open(os.path.join(base_path, 'processed/index2word.pickle'), 'wb') as handle:
        pickle.dump(index2word, handle)
    analyze = analyze_term if mode == 'term' else analyze_category
    log = {
        'vocab_size': len(index2word),
        'oov_size': len(word2index) - len(index2word),
        'train_data': analyze(train_data),
        'val_data': analyze(val_data),
        'num_categories': 3
    }
    if not os.path.exists(os.path.join(base_path, 'log')):
        os.makedirs(os.path.join(base_path, 'log'))
    with open(os.path.join(base_path, 'log/log.yml'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)