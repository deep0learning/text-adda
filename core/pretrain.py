"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
from params import param
from utils import save_model


def train_src(args, encoder, classifier, data_loader, data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    # instantiate EarlyStop
    earlystop = EarlyStop(args.patience)

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=param.c_learning_rate,
        betas=(param.beta1, param.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.num_epochs_pre):
        for step, (reviews, labels) in enumerate(data_loader):

            # set train state for Dropout and BN layers
            encoder.train()
            classifier.train()

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(reviews))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step_pre == 0:
                print("Epoch [%.3d/%.3d] Step [%.2d/%.2d]: loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs_pre,
                         step + 1,
                         len(data_loader),
                         loss.item()))

        # eval model on test set
        if (epoch + 1) % args.eval_step_pre == 0:
            # print('Epoch [{}/{}]'.format(epoch + 1, param.num_epochs_pre))
            eval_src(encoder, classifier, data_loader)
            earlystop.update(eval_src(encoder, classifier, data_loader_eval, True))
            print()

        # save model parameters
        if (epoch + 1) % args.save_step_pre == 0:
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

        if earlystop.stop:
            break

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader, out=False):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, labels) in data_loader:

        preds = classifier(encoder(reviews))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.2f" % (loss, acc))

    if out:
        return loss


class EarlyStop:
    def __init__(self, patience):
        self.count = 0
        self.maxAcc = 0
        self.patience = patience
        self.stop = False

    def update(self, acc):
        if acc < self.maxAcc:
            self.count += 1
        else:
            self.count = 0
            self.maxAcc = acc

        if self.count > self.patience:
            self.stop = True
