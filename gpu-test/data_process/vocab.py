import operator

PAD = '<pad>'
UNK = '<unk>'
ASPECT = '<aspect>'

class Vocab(object):

    def __init__(self):
        self._count_dict = dict()
        self._predefined_list = [PAD, UNK, ASPECT]

    def add(self, word):
        # 词频统计：
        if word in self._count_dict:
            self._count_dict[word] += 1
        else:
            self._count_dict[word] = 1

    def add_list(self, words):
        for word in words:
            self.add(word)

    def get_vocab(self, max_size=None, min_freq=0):
        # 文本分词后：text = {'北京', '是', '中国的', '首都'}
        # word2index = {'<UNK>': 0, '北京': 1, '是': 2, '中国的': 3, '首都': 4}
        sorted_words = sorted(self._count_dict.items(), key=operator.itemgetter(1),
                              reverse=True)  # operator.itemgetter(1) 获取对象的第2个域的值；通过比较self._count_dict.items()的第2个域来进行排序
        word2index = {}
        for word in self._predefined_list:
            word2index[word] = len(word2index)  # word2index[PAD]=0，word2index[UNK]=0，word2index[ASPECT]=0
        for word, freq in sorted_words:
            if word in word2index:
                continue
            if (max_size is not None and len(word2index) >= max_size) or freq < min_freq:
                word2index[word] = word2index[UNK]  # 超出规定字典大小或词频小于规定值的词值为0
            else:
                word2index[word] = len(word2index)  # 构建词汇表的索引，输出形式：{'<UNK>': 0, '北京': 1, ...}
        index2word = {}
        index2word[word2index[UNK]] = UNK   # {0: '<UNK>'}
        for word, index in word2index.items():
            if index == word2index[UNK]:
                continue
            else:
                index2word[index] = word  # 输出形式：{0 ：'<UNK>', 1 ：'北京', ...}
        return word2index, index2word
