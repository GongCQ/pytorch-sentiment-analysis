import os
import sys
import random 
from torchtext.data.example import Example as E
from torchtext.data.dataset import Dataset
import pytorch_pretrained_bert as torch_bert

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


class PPBTok:
    def __init__(self, vocab_file_path, max_len):
        self.tok = torch_bert.tokenization.BertTokenizer(vocab_file=vocab_file_path, max_len=max_len)
        self.max_len = max_len
        self.vocab = self.tok.vocab

    def tokenize(self, text):
        return self.tok.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            try:
                return self.tok.convert_tokens_to_ids([tokens])[0]
            except Exception as e:
                return self.tok.convert_tokens_to_ids(['[UNK]'])[0]
        else:
            ids = []
            for token in tokens:
                id = self.vocab[token] if token in self.vocab.keys() else self.vocab['[UNK]']
                ids.append(id)
            return ids

    def convert_ids_to_tokens(self, ids):
        return self.tok.convert_ids_to_tokens(ids)

    def encode(self, tokens):
        tokens = self.tokenize(tokens)
        return [self.vocab['[CLS]']] + self.convert_tokens_to_ids(tokens) + [self.vocab['[SEP]']]

    def decode(self, ids):
        return ''.join(self.convert_ids_to_tokens(ids))