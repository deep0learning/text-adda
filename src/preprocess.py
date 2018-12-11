import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter
from timeit import default_timer as timer
import os
import pandas as pd


STOP_WORDS = stopwords.words("english")

def read_data(file_path_dataset):
    return pd.read_csv(file_path_dataset, delimiter='\t')


def process(file_path_dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pos_train_path = os.path.join(dir_path, file_path_dataset, 'train/positive.review')
    neg_train_path = os.path.join(dir_path, file_path_dataset, 'train/negative.review')
    pos_test_path = os.path.join(dir_path, file_path_dataset, 'test/positive.review')
    neg_test_path = os.path.join(dir_path, file_path_dataset, 'test/negative.review')

    pos_train_parser = BeautifulSoup(open(pos_train_path, 'r'), 'lxml')
    neg_train_parser = BeautifulSoup(open(neg_train_path, 'r'), 'lxml')
    pos_test_parser = BeautifulSoup(open(pos_test_path, 'r'), 'lxml')
    neg_test_parser = BeautifulSoup(open(neg_test_path, 'r'), 'lxml')

    category = os.path.basename(file_path_dataset)
    proc_path = os.path.join(dir_path, 'data/processed', category)
    if not os.path.exists(proc_path):
        os.mkdir(proc_path)

    proc_train = os.path.join(dir_path, 'data/processed', category, 'train.txt')
    proc_test = os.path.join(dir_path, 'data/processed', category, 'test.txt')

    with open(proc_train, 'w') as train_file:
        train_file.write('review\tlabel')
        i = 0
        for review in pos_train_parser.find_all("text"):
            i += 1
            if i % 3 == 0:
                if len(review.text.rstrip(['\n', '\t'])) == 0:
                    continue
                text = review.text.rstrip(['\n', '\t']) + '\t1'
                train_file.write(text)

        for review in neg_train_parser.find_all("text"):
            i += 1
            if i % 3 == 0:
                if len(review.text.rstrip(['\n', '\t'])) == 0:
                    continue
                text = review.text.rstrip(['\n', '\t']) + '\t0'
                train_file.write(text)

    with open(proc_test, 'w') as test_file:
        test_file.write('review\tlabel')
        i = 0
        for review in pos_test_parser.find_all("text"):
            i += 1
            if i % 3 == 0:
                if len(review.text.rstrip(['\n', '\t'])) == 0:
                    continue
                text = review.text.rstrip(['\n', '\t']) + '\t1'
                test_file.write(text)

        for review in neg_test_parser.find_all("text"):
            i += 1
            if i % 3 == 0:
                if len(review.text.rstrip(['\n', '\t'])) == 0:
                    continue
                text = review.text.rstrip(['\n', '\t']) + '\t0'
                test_file.write(text)


def build_vocab(reviews):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab_idx = 2

    for review in reviews:
        reviewsentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(review)]

        for sentence in reviewsentences:
            cleaned_sentence = preprocess_sentence(sentence)
            word_counter.update(cleaned_sentence)

    max_vocab = len(word_counter) + 2

    for key, value in word_counter.most_common(max_vocab-2):
        vocab[key] = vocab_idx
        vocab_idx += 1

    for key, value in vocab.items():
        reverse_vocab[value] = key

    print("Number of words in vocab %d" % len(vocab))

    return vocab, reverse_vocab


def tokenize(reviews):
    tokens = []

    for review in reviews:
        cleaned_sentences = []
        reviewsentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(review)]

        for sentence in reviewsentences:
            cleaned_sentence = preprocess_sentence(sentence)
            cleaned_sentences += cleaned_sentence

        tokens.append(cleaned_sentences)

    return tokens


def concat_all(review1, review2, review3, review4):
    all_review = list()
    all_review.append(review1)
    all_review.append(review2)
    all_review.append(review3)
    all_review.append(review4)
    all_review = pd.concat(all_review, axis = 0, ignore_index = True)
    return all_review


def preprocess_sentence(sentence):
    cleaned_sentence = []

    for word in sentence:
        word = word.strip().lower()
        if word is None or word in STOP_WORDS or word == '' or len(word) <= 2:
            continue
        else:
            cleaned_sentence.append(word)

    return cleaned_sentence


def tokensToSequences(tokens, vocab, max_length=0):
    sequences = []
    max_len = max_length

    if max_len == 0:
        max_len = max([len(review) for review in tokens])

    for review in tokens:
        if len(review) < max_len:
            temp = review
            for _ in range(len(temp), max_len):
                temp.append('<PAD>')
        else:
            temp = review[:max_len]

        sequence = []
        for word in temp:
            if word in vocab.keys():
                sequence.append(vocab[word])
            else:
                sequence.append(vocab['<UNK>'])
        sequences.append(sequence)

    return sequences

def reviewsToSequences(reviews, vocab, max_length=0):
    tokens = tokenize(reviews)
    sequences = tokensToSequences(tokens, vocab, max_length)
    return sequences