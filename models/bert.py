import torch
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
        self.classifier = nn.Sequential(nn.LeakyReLU(),
                                        nn.Dropout(mp.dropout),
                                        nn.Linear(mp.c_input_dims, mp.c_hidden_dims),
                                        nn.Linear(mp.c_hidden_dims, mp.c_output_dims))
        self.apply(self.init_bert_weights)

    def forward(self, x):
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=0.02)
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias