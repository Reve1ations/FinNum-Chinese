from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
from xml.etree.ElementTree import parse
from data_process.vocab import Vocab
import spacy
import re
import json

# import paddlehub as hub
# model = hub.Module(name='ernie_v2_eng_base')
# vocab_path = model.get_vocab_path()
# print(vocab_path)

url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')

def check(x):
    return len(x) >= 1 and not x.isspace()


def tokenizer(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))


def parse_sentence_term(path, lowercase=False):
    tree = parse(path)  # 将XML文档解析为树（tree）
    sentences = tree.getroot()  # 获取根节点<Element 'sentences' at ...>
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text  # 返回当前节点所包含的所有文本内容
        if lowercase:
            text = text.lower()  # 字符串中所有大写字符为小写
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')  # 获取方面词
            if lowercase:
                term = term.lower()
            polarity = aspectTerm.get('polarity')  # 获取情感词
            start = aspectTerm.get('from')  # 获取位置开始信息
            end = aspectTerm.get('to')  # 获取位置结束信息
            if 'test' in path:
                piece = text + split_char + term + split_char + start + split_char + end
            else:
                piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end
            data.append(piece)
    return data

def parse_sentence_category(path, lowercase=False):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        if lowercase:
            text = text.lower()
        aspectCategories = sentence.find('aspectCategories')
        if aspectCategories is None:
            continue
        for aspectCategory in aspectCategories:
            category = aspectCategory.get('category')  # 获取方面类别
            polarity = aspectCategory.get('polarity')
            if 'test' in path:
                piece = text + split_char + category
            else:
                piece = text + split_char + category + split_char + polarity
            data.append(piece)
    return data

def category_filter(data, remove_list):  # remove_list = ['conflict'] 过滤的情感标签为'conflict'的数据
    remove_set = set(remove_list)  # set是一个不允许内容重复的组合，而且set里的内容位置是随意的，不能用索引列出
    filtered_data = []
    for text in data:
        if not text.split('__split__')[2] in remove_set:  # 如果情感标签不为'conflict'
            filtered_data.append(text)
    return filtered_data


def build_vocab(data, max_size, min_freq):
    # 将字符列表转换为字符与索引相互对应的字典
    # 例如：text = ['北京', '上海', '广州', '深圳']
    # word2index = {'北京': 1, '上海': 2, '广州': 3, '深圳': 4}
    # index2word = {1: '北京', 2: '上海', 3: '广州', 4: '深圳'}
    if max_size == 'None':
        max_size = None
    vocab = Vocab()  # vocab.py文件
    for piece in data:
        text = piece.split('__split__')[0]
        text = tokenizer(text)
        vocab.add_list(text)
    return vocab.get_vocab(max_size=max_size, min_freq=min_freq)


def save_term_data(data, word2index, path):
    dirname = os.path.dirname(path)  # 去掉文件名XXX.npz，返回上一级目录：./data/MAMS-ATSA/processed_ernie
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence = []
    aspect = []
    label = []
    d = {
        'positive': 1,
        'negative': 2,
        'neutral': 0,
        'conflict': 3
    }
    for piece in data:
        if 'test' in path:
            text, term, start, end = piece.split('__split__')
            start, end = int(start), int(end)
            assert text[start: end] == term  # 在条件不满足程序运行的情况下直接返回错误
            sentence.append(text)
            aspect.append(term)
            label.append(0)
        else:
            text, term, polarity, start, end = piece.split('__split__')
            start, end = int(start), int(end)
            assert text[start: end] == term  # 在条件不满足程序运行的情况下直接返回错误
            sentence.append(text)
            aspect.append(term)
            label.append(d[polarity])
    np.savez(path, sentence=sentence, aspect=aspect, label=label)


def save_category_data(data, word2index, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence = []
    aspect = []
    label = []
    d = {
        'positive': 1,
        'negative': 2,
        'neutral': 0,
        'conflict': 3
    }
    # cd = {
    #     'food': 0,
    #     'service': 1,
    #     'staff': 2,
    #     'price': 3,
    #     'ambience': 4,
    #     'menu': 5,
    #     'place': 6,
    #     'miscellaneous': 7
    # }
    for piece in data:
        if 'test' in path:
            text, category = piece.split('__split__')
            sentence.append(text)
            aspect.append(category)
            label.append(0)
        else:
            text, category, polarity = piece.split('__split__')
            sentence.append(text)
            aspect.append(category)
            label.append(d[polarity])
    np.savez(path, sentence=sentence, aspect=aspect, label=label)


def analyze_term(data):
    num = len(data)
    sentence_lens = []
    aspect_lens = []
    log = {'total': num}
    for piece in data:
        text, term, polarity, _, _ = piece.split('__split__')
        sentence_lens.append(len(tokenizer(text)))
        aspect_lens.append(len(tokenizer(term)))
        if not polarity in log:
            log[polarity] = 0
        log[polarity] += 1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    log['aspect_max_len'] = max(aspect_lens)
    log['aspect_avg_len'] = sum(aspect_lens) / len(aspect_lens)
    return log

def analyze_category(data):
    num = len(data)
    sentence_lens = []
    log = {'total': num}
    for piece in data:
        text, category, polarity = piece.split('__split__')
        sentence_lens.append(len(tokenizer(text)))
        if not polarity in log:
            log[polarity] = 0
        log[polarity] += 1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    return log



