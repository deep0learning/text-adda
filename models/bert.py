import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from params import model_param as mp

model = BertModel.from_pretrained('bert-base-uncased')


class BERTEncoder(nn.Module):

    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.restored = False
        self.encoder = model

    def forward(self, x):
        _, feat = self.encoder(x)
        return feat


class BERTClassifier(nn.Module):

    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.restored = False
        self.classifier = nn.Sequential(nn.Dropout(mp.dropout),
                                        nn.Linear(mp.c_input_dims, mp.c_hidden_dims),
                                        nn.LeakyReLU(),
                                        nn.Linear(mp.c_hidden_dims, mp.c_output_dims))

    def forward(self, x):
        out = self.classifier(x)
        return out
