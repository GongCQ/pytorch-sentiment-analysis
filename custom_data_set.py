import os
import sys
import random 
from torchtext.data.example import Example as E
from torchtext.data.dataset import Dataset

class Example:
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __lt__(self, other):
        return len(self.text) < len(other.text)

def get_examples(path, tokenizer, max_len, max_num=0, include_neutual=True):
    if max_num <= 0:
        max_num = sys.maxsize
    tokenizer = tokenizer
    pos_path = os.path.join(path, 'pos.txt')
    neg_path = os.path.join(path, 'neg.txt')
    neu_path = os.path.join(path, 'neu.txt')
    tag_path_list = [('0', pos_path), ('2', neg_path)] + ([('1', neu_path)] if include_neutual else [])
    examples = []
    for tag, path in tag_path_list:
        file = open(path, encoding='utf-8')
        tag_count  =0
        for line in file:
            line = line.strip()
            if len(line) > max_len - 2:
                line = line[ : max_len - 2]
            if len(line) > 1:
                encoded = tokenizer.encode(line)[1 : -1] # 因为tokenizer.encode会给头尾加上[CLS]和[SEP]，过会在生成数据集时又会加一次，所以这里先把头尾去掉，以免被加两次
                example = Example(encoded, tag)
                examples.append(example)
                tag_count += 1
                if tag_count >= max_num:
                    break
    
    random.shuffle(examples)
    # [print(e.label) for e in examples]
    return examples

def get_data_set(path, tokenizer, text_field, label_field, max_len, max_num=0, include_neutual=True):
    fields = [('text', text_field), ('label', label_field)]
    examples = get_examples(path, tokenizer, max_len=max_len, max_num=max_num, include_neutual=include_neutual)
    data_set = Dataset(examples=examples, fields=fields)
    return data_set