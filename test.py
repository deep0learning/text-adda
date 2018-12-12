import torch
import torch.nn as nn
import numpy as np
from params import model_param, param
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, ConvNetClassifier, ConvNetEncoder
from utils import get_data_loader, init_model, init_random_seed, load_pretrained
from preprocess import read_data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.convert_tokens_to_ids(['[CLS]'])
# preprocess data
src_train = read_data('./data/processed/electronics/train.txt')
src_test = read_data('./data/processed/electronics/test.txt')
tgt_train = read_data('./data/processed/kitchen/train.txt')
tgt_test = read_data('./data/processed/kitchen/test.txt')

src_train_sequences = []
src_test_sequences = []
tgt_train_sequences = []
tgt_test_sequences = []

for i in range(len(src_train.review)):
    tokenized_text = tokenizer.tokenize(src_train.review[i])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    src_train_sequences.append(indexed_tokens)

for i in range(len(tgt_test.review)):
    tokenized_text = tokenizer.tokenize(tgt_test.review[i])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tgt_test_sequences.append(indexed_tokens)
    
src_data_loader = get_data_loader(src_train_sequences, src_train.label, maxlen=100)
tgt_data_loader = get_data_loader(tgt_test_sequences, tgt_test.label, maxlen=100)
cmodel = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

src_encoder = init_model(nn.Sequential(*list(cmodel.children())[:-2]))
tgt_encoder = init_model(nn.Sequential(*list(cmodel.children())[:-2]))
src_classifier = cmodel.classifier
model.cuda()
src_encoder.cuda()
tgt_encoder.cuda()
src_classifier.cuda()
sample = sample.cuda()

model.eval()
src_encoder.eval()

_, pooled_output = model(sample[:2])
print(pooled_output)
src_encoder(sample[:2])[1]
src_classifier(src_encoder(sample[:2])[1])
src_classifier(tgt_encoder(sample[:2])[1])

for param in src_encoder.parameters():
     param.requires_grad = False

    src_encoder.train()
    src_classifier.train()

for step, (reviews, labels) in enumerate(tgt_data_loader):
    sample = reviews
    break
    # make labels squeezed
    print(labels)
    
for sequence in sequences:
    seqlen = len(sequence)
    if seqlen < maxlen:
        sequence.extend([0 for _ in range(maxlen-seqlen)])
    else:
        sequence = sequence[:maxlen]
        seq_data.append(sequence)