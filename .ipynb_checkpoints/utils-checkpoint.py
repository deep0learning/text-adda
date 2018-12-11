import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import params
import numpy as np

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_model(net, restore):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net

def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))


def get_data_loader(sequences, labels):
    # dataset and data loader
    text_dataset = textDataset(sequences, labels)

    text_data_loader = DataLoader(
        dataset=text_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return text_data_loader


class textDataset(Dataset):

    def __init__(self, sequences, labels):
        self.data = np.array(sequences)
        self.labels = labels
        self.dataset_size = len(self.data)

    def __getitem__(self, index):
        review, label = self.data[index], self.labels[index]
        review = torch.LongTensor(review)
        label = torch.LongTensor([np.int64(label).item()])
        return review, label

    def __len__(self):
        return self.dataset_size