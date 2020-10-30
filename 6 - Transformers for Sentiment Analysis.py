# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# # 6 - Transformers for Sentiment Analysis
# 
# In this notebook we will be using the transformer model, first introduced in [this](https://arxiv.org/abs/1706.03762) paper. Specifically, we will be using the BERT (Bidirectional Encoder Representations from Transformers) model from [this](https://arxiv.org/abs/1810.04805) paper. 
# 
# Transformer models are considerably larger than anything else covered in these tutorials. As such we are going to use the [transformers library](https://github.com/huggingface/transformers) to get pre-trained transformers and use them as our embedding layers. We will freeze (not train) the transformer and only train the remainder of the model which learns from the representations produced by the transformer. In this case we will be using a multi-layer bi-directional GRU, however any model can learn from these representations.

# ## Preparing Data
# 
# First, as always, let's set the random seeds for deterministic results.

# %%
import os
import datetime as dt
import torch

import random
import numpy as np
import pytorch_pretrained_bert as torch_bert

print('%s begin run.' % dt.datetime.now())
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# %%
USE_PPB = True # 使用pytorch_pretrained_bert而不是transformers
USE_MASK = True #
BATCH_SIZE = 64 # 128
bert_model_name = 'bert-base-chinese'
bert_model_folder = os.path.join('bert_model', 'pytorch_pretrained_bert', bert_model_name)
data_set_path = os.path.join('data', 'summary')
INCLUDE_NEUTUAL = True
BERT_LR = 0.001
FC_LR = 0.01
train_max_num = 120000
test_max_num = 12000
valid_max_num = 12000
model_save_path = os.path.join('saved_models')
MODEL_STAMP = dt.datetime.now().strftime('%Y%m%d%H%M%S')
print('~~ USE_PPB %s' % USE_PPB)
print('~~ USE_MASK %s' % USE_MASK)
print('~~ BATCH_SIZE %s' % BATCH_SIZE)
print('~~ INCLUDE_NEUTUAL %s' % INCLUDE_NEUTUAL)
print('~~ BERT_LR %s' % BERT_LR)
print('~~ FC_LR %s' % FC_LR)

# The transformer has already been trained with a specific vocabulary, which means we need to train with the exact same vocabulary and also tokenize our data in the same way that the transformer did when it was initially trained.
# 
# Luckily, the transformers library has tokenizers for each of the transformer models provided. In this case we are using the BERT model which ignores casing (i.e. will lower case every word). We get this by loading the pre-trained `bert-base-uncased` tokenizer.


# %%

if USE_PPB:
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

    tokenizer = PPBTok(vocab_file_path=os.path.join(bert_model_folder, 'vocab.txt'), max_len=510)
    bert = torch_bert.BertModel.from_pretrained(bert_model_folder)
    max_input_length = tokenizer.max_len
else:
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(bert_model_folder)
    bert = BertModel.from_pretrained(bert_model_folder)
    max_input_length = tokenizer.max_model_input_sizes[bert_model_name]

aaa = tokenizer.encode('从前有座山')
bbb = tokenizer.encode(['从', '前', '有', '座', '山'])
ccc = tokenizer.decode(aaa)
ddd = tokenizer.decode(bbb)

print('success to load bert tokenizer and model.')


# The `tokenizer` has a `vocab` attribute which contains the actual vocabulary we will be using. We can check how many tokens are in it by checking its length.

# %%
len(tokenizer.vocab)


# Using the tokenizer is as simple as calling `tokenizer.tokenize` on a string. This will tokenize and lower case the data in a way that is consistent with the pre-trained transformer model.

# %%
tokens = tokenizer.tokenize('从前有座山，山上有座庙。')

print(tokens)


# We can numericalize tokens using our vocabulary using `tokenizer.convert_tokens_to_ids`.

# %%
indexes = tokenizer.convert_tokens_to_ids(tokens)

print(indexes)


# The transformer was also trained with special tokens to mark the beginning and end of the sentence, detailed [here](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel). As well as a standard padding and unknown token. We can also get these from the tokenizer.
# 
# **Note**: the tokenizer does have a beginning of sequence and end of sequence attributes (`bos_token` and `eos_token`) but these are not set and should not be used for this transformer.

# %%
init_token = '[CLS]' # tokenizer.cls_token
eos_token = '[SEP]' # tokenizer.sep_token
pad_token = '[PAD]' # tokenizer.pad_token
unk_token = '[UNK]' # tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)


# We can get the indexes of the special tokens by converting them using the vocabulary...

# %%
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)


# ...or by explicitly getting them from the tokenizer.

# %%
# init_token_idx = tokenizer.cls_token_id
# eos_token_idx = tokenizer.sep_token_id
# pad_token_idx = tokenizer.pad_token_id
# unk_token_idx = tokenizer.unk_token_id
#
# print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)


# Another thing we need to handle is that the model was trained on sequences with a defined maximum length - it does not know how to handle sequences longer than it has been trained on. We can get the maximum length of these input sizes by checking the `max_model_input_sizes` for the version of the transformer we want to use. In this case, it is 512 tokens.

# %%

print(max_input_length)


# Previously we have used the `spaCy` tokenizer to tokenize our examples. However we now need to define a function that we will pass to our `TEXT` field that will handle all the tokenization for us. It will also cut down the number of tokens to a maximum length. Note that our maximum length is 2 less than the actual maximum length. This is because we need to append two tokens to each sequence, one to the start and one to the end.

# %%
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


# Now we define our fields. The transformer expects the batch dimension to be first, so we set `batch_first = True`. As we already have the vocabulary for our text, provided by the transformer we set `use_vocab = False` to tell torchtext that we'll be handling the vocabulary side of things. We pass our `tokenize_and_cut` function as the tokenizer. The `preprocessing` argument is a function that takes in the example after it has been tokenized, this is where we will convert the tokens to their indexes. Finally, we define the special tokens - making note that we are defining them to be their index value and not their string value, i.e. `100` instead of `[UNK]` This is because the sequences will already be converted into indexes.
# 
# We define the label field as before.

# %%
from torchtext import data

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)


# We load the data and create the validation splits as before.

# %%
from torchtext import datasets
import custom_data_set
#
# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
#
# train_data, valid_data = train_data.split(random_state = random.seed(SEED))
train_data = custom_data_set.get_data_set(os.path.join(data_set_path, 'train'),
                                          tokenizer=tokenizer, max_len=max_input_length,
                                          text_field=TEXT, label_field=LABEL, max_num=train_max_num, 
                                          include_neutual=INCLUDE_NEUTUAL)
test_data = custom_data_set.get_data_set(os.path.join(data_set_path, 'test'),
                                         tokenizer=tokenizer, max_len=max_input_length,
                                         text_field=TEXT, label_field=LABEL, max_num=test_max_num,
                                         include_neutual=INCLUDE_NEUTUAL)
valid_data = custom_data_set.get_data_set(os.path.join(data_set_path, 'valid'),
                                          tokenizer=tokenizer, max_len=max_input_length,
                                          text_field=TEXT, label_field=LABEL, max_num=valid_max_num, 
                                          include_neutual=INCLUDE_NEUTUAL)


# %%
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


# We can check an example and ensure that the text has already been numericalized.

# %%
print(vars(train_data.examples[6]))


# We can use the `convert_ids_to_tokens` to transform these indexes back into readable tokens.

# %%
tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])

print(tokens)


# Although we've handled the vocabulary for the text, we still need to build the vocabulary for the labels.

# %%
LABEL.build_vocab(train_data)


# %%
print(LABEL.vocab.stoi)


# As before, we create the iterators. Ideally we want to use the largest batch size that we can as I've found this gives the best results for transformers.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)

# ## Build the Model
# 
# Next, we'll load the pre-trained model, making sure to load the same model as we did for the tokenizer.



# Next, we'll define our actual model. 
# 
# Instead of using an embedding layer to get embeddings for our text, we'll be using the pre-trained transformer model. These embeddings will then be fed into a GRU to produce a prediction for the sentiment of the input sentence. We get the embedding dimension size (called the `hidden_size`) from the transformer via its config attribute. The rest of the initialization is standard.
# 
# Within the forward pass, we wrap the transformer in a `no_grad` to ensure no gradients are calculated over this part of the model. The transformer actually returns the embeddings for the whole sequence as well as a *pooled* output. The [documentation](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel) states that the pooled output is "usually not a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence", hence we will not be using it. The rest of the forward pass is the standard implementation of a recurrent model, where we take the hidden state over the final time-step, and pass it through a linear layer to get our predictions.

# %%
import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            if USE_MASK:
                attention_mask = (text!=0).long()
            else:
                attention_mask = torch.ones_like(text)
            if USE_PPB:
                embedded = self.bert(text, attention_mask=attention_mask, output_all_encoded_layers=False)[0]
            else:
                embedded = self.bert(text, attention_mask=attention_mask)[0]
            ddd = 0

        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


# Next, we create an instance of our model using standard hyperparameters.

# %%
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)


# We can check how many parameters the model has. Our standard models have under 5M, but this one has 112M! Luckily, 110M of these parameters are from the transformer and we will not be training those.

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In order to freeze paramers (not train them) we need to set their `requires_grad` attribute to `False`. To do this, we simply loop through all of the `named_parameters` in our model and if they're a part of the `bert` transformer model, we set `requires_grad = False`. 

# %%
for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False


# We can now see that our model has under 3M trainable parameters, making it almost comparable to the `FastText` model. However, the text still has to propagate through the transformer which causes training to take considerably longer.

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# We can double check the names of the trainable parameters, ensuring they make sense. As we can see, they are all the parameters of the GRU (`rnn`) and the linear layer (`out`).

# %%
for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)


# ## Train the Model
# 
# As is standard, we define our optimizer and criterion (loss function).

# %%
import torch.optim as optim

optimizer = optim.Adam(params=[{'params': model.bert.parameters(), 'lr': BERT_LR}, 
                               {'params': model.rnn.parameters(), 'lr': FC_LR}, 
                               {'params': model.out.parameters(), 'lr': FC_LR}], lr=FC_LR)


# %%
criterion = nn.BCEWithLogitsLoss()


# Place the model and criterion onto the GPU (if available)

# %%
model = model.to(device)
criterion = criterion.to(device)


# Next, we'll define functions for: calculating accuracy, performing a training epoch, performing an evaluation epoch and calculating how long a training/evaluation epoch takes.

# %%
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    if not INCLUDE_NEUTUAL:
        rounded_preds = torch.round(torch.sigmoid(preds))
    else:
        rounded_preds = torch.round(2 * torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc


# %%
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    b = 0
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label/(2 if INCLUDE_NEUTUAL else 1))
        
        acc = binary_accuracy(predictions, batch.label)
        acc_value = float(acc)
        
        loss.backward()        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        print('%s  batch %s, loss %s, acc %s' % (dt.datetime.now(), b, loss.item(), acc_value))
        b += 1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# %%
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label/(2 if INCLUDE_NEUTUAL else 1))
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# %%
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Finally, we'll train our model. This takes considerably longer than any of the previous models due to the size of the transformer. Even though we are not training any of the transformer's parameters we still need to pass the data through the model which takes a considerable amount of time on a standard GPU.

# %%
N_EPOCHS = 5

best_valid_loss = float('inf')

print('%s begin train.' % dt.datetime.now())
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    model_file_name = 'model%s_neu=%s_bat=%s_blr=%s_flr=%s_epo=%s_ppb=%s_msk=%s_acc=%s.pkl' % \
                      (MODEL_STAMP, INCLUDE_NEUTUAL, BATCH_SIZE, BERT_LR, FC_LR, epoch, USE_PPB, USE_MASK, round(valid_acc, 4))
    torch.save(model, os.path.join(model_save_path,model_file_name))


# We'll load up the parameters that gave us the best validation loss and try these on the test set - which gives us our best results so far!

# %%
model.load_state_dict(torch.load('tut6-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# ## Inference
# 
# We'll then use the model to test the sentiment of some sequences. We tokenize the input sequence, trim it down to the maximum length, add the special tokens to either side, convert it to a tensor, add a fake batch dimension and then pass it through our model.

# %%
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


# %%
predict_sentiment(model, tokenizer, "This film is terrible")


# %%
predict_sentiment(model, tokenizer, "This film is great")


