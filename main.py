"""Main script for ADDA."""

import params
import torch
import model_param
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, ConvNetClassifier, ConvNetEncoder
from utils import get_data_loader, init_model, init_random_seed, load_pretrained
from preprocess import read_data, tokenize, tokensToSequences, reviewsToSequences

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # preprocess data
    src_train = read_data('./data/processed/electronics/train.txt')
    src_test = read_data('./data/processed/electronics/test.txt')
    tgt_train = read_data('./data/processed/kitchen/train.txt')
    tgt_test = read_data('./data/processed/kitchen/test.txt')

    tokens, vocab, reverse_vocab = tokenize(src_train.review)
    src_train_sequences = tokensToSequences(tokens, vocab)
    src_test_sequences = reviewsToSequences(src_test.review, vocab)
    tgt_train_sequences = reviewsToSequences(tgt_train.review, vocab)
    tgt_test_sequences = reviewsToSequences(tgt_test.review, vocab)
    model_param.num_vocab = len(vocab)

    # load dataset
    src_data_loader = get_data_loader(src_train_sequences, src_train.label)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test.label)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train.label)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test.label)

    # load pretrained
    pretrain_embed = load_pretrained('GoogleNews-vectors-negative300.bin', reverse_vocab)

    # load models
    src_encoder = init_model(net=ConvNetEncoder(model_param, pretrain_embed),
                             restore=None)
    src_classifier = init_model(net=ConvNetClassifier(model_param),
                                restore=None)
    tgt_encoder = init_model(net=ConvNetEncoder(model_param, pretrain_embed),
                             restore=None)
    critic = init_model(Discriminator(input_dims=model_param.d_input_dims,
                                      hidden_dims=model_param.d_hidden_dims,
                                      output_dims=model_param.d_output_dims),
                        restore=None)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader, src_data_loader_eval)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
