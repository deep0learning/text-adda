import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from params import param
from gensim.models.keyedvectors import KeyedVectors


def read_data(file_path_dataset):
    return pd.read_csv(file_path_dataset, delimiter='\t')


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(param.model_root):
        os.makedirs(param.model_root)
    torch.save(net.state_dict(),
               os.path.join(param.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(param.model_root,
                                                             filename)))


def load_pretrained(path, reverse_vocab):
    print("loading word2vec...")
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    pretrain_embed = list()

    pretrain_embed.append(np.zeros(300).astype("float32")) # for <PAD> token
    pretrain_embed.append(np.random.uniform(-0.01, 0.01, 300).astype("float32")) # for <UNK> token
    for i in range(len(reverse_vocab)-2):
        word = reverse_vocab[i+2]
        if word in word_vectors.vocab:
            pretrain_embed.append(word_vectors.word_vec(word))
        else:
            pretrain_embed.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

    pretrain_embed = np.array(pretrain_embed)
    print("word2vec successfully loaded")
    return pretrain_embed


def get_data_loader(sequences, labels, maxlen=None):
    # dataset and data loader
    text_dataset = TextDataset(sequences, labels, maxlen)

    text_data_loader = DataLoader(
        dataset=text_dataset,
        batch_size=param.batch_size,
        shuffle=True)

    return text_data_loader


class TextDataset(Dataset):
    def __init__(self, sequences, labels, maxlen):
        if maxlen is None:
            maxlen = max([len(sequence) for sequence in sequences])

        seq_data = list()
        for sequence in sequences:
            sequence.insert(0, 101)
            seqlen = len(sequence)
            if seqlen < maxlen:
                sequence.extend([0] * (maxlen-seqlen))
            else:
                sequence = sequence[:maxlen]
            seq_data.append(sequence)

        self.data = torch.LongTensor(seq_data).cuda()
        self.labels = torch.LongTensor(labels).cuda()
        self.dataset_size = len(self.data)

    def __getitem__(self, index):
        review, label = self.data[index], self.labels[index]
        return review, label

    def __len__(self):
        return self.dataset_size