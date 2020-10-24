import os
import random 
from torchtext.data.example import Example
from torchtext.data.dataset import Dataset

def get_examples(path, tokenizer):
    tokenizer = tokenizer
    pos_path = os.path.join(path, 'pos.txt')
    neg_path = os.path.join(path, 'neg.txt')
    examples = []
    pos_file = open(pos_path, encoding='utf-8')
    for line in pos_file:
        line = line.strip()
        if len(line) > 1:
            example = Example()
            example.text = tokenizer.encode(line)
            example.label = 'pos'
            examples.append(example)
    neg_file = open(neg_path, encoding='utf-8')
    for line in neg_file:
        line = line.strip()
        if len(line) > 1:
            example = Example()
            example.text = tokenizer.encode(line)
            example.label = 'neg'
            examples.append(example)
    random.shuffle(examples)
    return examples

def get_data_set(path, tokenizer, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]
    examples = get_examples(path, tokenizer)
    data_set = Dataset(examples=examples, fields=fields)
    return data_set