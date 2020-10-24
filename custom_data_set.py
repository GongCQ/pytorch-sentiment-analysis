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

def get_examples(path, tokenizer, max_num=0):
    if max_num <= 0:
        max_num = sys.maxsize
    tokenizer = tokenizer
    pos_path = os.path.join(path, 'pos.txt')
    neg_path = os.path.join(path, 'neg.txt')
    examples = []
    pos_file = open(pos_path, encoding='utf-8')
    for line in pos_file:
        line = line.strip()
        if len(line) > 500:
            line = line[ : 500]
        if len(line) > 1:
            example = Example(tokenizer.encode(line), 'pos')
            # example.text = tokenizer.encode(line)
            # example.label = 'pos'
            examples.append(example)
            if len(examples) >= max_num:
                break
    neg_file = open(neg_path, encoding='utf-8')
    for line in neg_file:
        line = line.strip()
        if len(line) > 500:
            line = line[ : 500]
        if len(line) > 1:
            example = Example(tokenizer.encode(line), 'neg')
            # example.text = tokenizer.encode(line)
            # example.label = 'neg'
            examples.append(example)
            if len(examples) >= max_num * 2:
                break
    random.shuffle(examples)
    return examples

def get_data_set(path, tokenizer, text_field, label_field, max_num=0):
    fields = [('text', text_field), ('label', label_field)]
    examples = get_examples(path, tokenizer, max_num)
    data_set = Dataset(examples=examples, fields=fields)
    return data_set