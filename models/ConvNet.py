import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetEncoder(nn.Module):

    def __init__(self, model_param, pretrain_embed=None):
        """Init ConvNet encoder."""
        super(ConvNetEncoder, self).__init__()
        self.restored = False
        self.model_param = model_param

        num_vocab = model_param.num_vocab
        embed_dim = model_param.embed_dim
        Ci = 1
        Co = model_param.kernel_num
        Ks = model_param.kernel_sizes

        self.embed = nn.Embedding(num_vocab, embed_dim)

        if model_param.pretrain and pretrain_embed is not None:
            self.embed.weight.data.copy_(torch.from_numpy(pretrain_embed))

        if not model_param.embed_freeze:
            self.embed.weight.requires_grad = False

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embed_dim)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(model_param.dropout)

    # def conv_and_pool(self, x, conv): # x: (N, Ci, Hi, D)
    #     x = F.relu(conv(x)).squeeze(3)  # (N, Co, Ho)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    def forward(self, x):
        x = self.embed(x)  # (N, Hi, D)
        x = x.unsqueeze(1)  # (N, Ci, Hi, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, Ho)]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(batch_size,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(batch_size,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(batch_size,Co)
        x = torch.cat((x1, x2, x3), 1) # (batch_size,len(Ks)*Co)
        '''
        feat = self.dropout(x)  # (N, len(Ks)*Co)
        return feat


class ConvNetClassifier(nn.Module):

    def __init__(self, model_param):
        """Init ConvNet encoder."""
        super(ConvNetClassifier, self).__init__()
        self.restored = False
        self.fc2 = nn.Linear(len(model_param.kernel_sizes) * model_param.kernel_num, model_param.class_num)

    def forward(self, feat):
        """Forward the ConvNet classifier."""
        out = self.fc2(feat)
        return out
