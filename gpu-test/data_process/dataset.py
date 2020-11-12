import torch
from torch.utils.data import Dataset
import numpy as np


class ABSADataset(Dataset):

    def __init__(self, data, input_list):
        super(ABSADataset, self).__init__()
        self.data = {}
        for key, value in data.items():    # items() 函数以列表返回可遍历的(键, 值) 元组数组
            self.data[key] = torch.tensor(value).long()
        self.len = self.data['label'].size(0)
        self.input_list = input_list

    def __getitem__(self, index):
        return_value = []
        for input in self.input_list:
            # print("self.data[input]=", len(self.data[input]))
            # print("self.data[input][index]=", self.data[input][index])
            return_value.append(self.data[input][index])
        return_value.append(self.data['label'][index])
        return return_value

    def __len__(self):
        return self.len