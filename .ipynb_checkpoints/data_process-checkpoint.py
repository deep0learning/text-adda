import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter
from timeit import default_timer as timer

STOP_WORDS = stopwords.words("english")

def extractallsentences(file_path_dataset):
    parser = BeautifulSoup(open(file_path_dataset, 'r'), 'xml')

    number_of_reviews = 0
    word_counter = Counter()
    reviews = []
    vocab = dict()
    reverse_vocab = dict()

    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab_idx = 2

    start = timer()  # Keep track of processing time

    for review in parser.find_all("review_text"):
        cleaned_sentences = []
        reviewsentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(review.text)]

        for sentence in reviewsentences:
            cleaned_sentence = preprocess_sentence(sentence)
            word_counter.update(cleaned_sentence)
            cleaned_sentences += cleaned_sentence

        reviews.append(cleaned_sentences)
        number_of_reviews += 1

    max_vocab = len(word_counter) + 2

    for key, value in word_counter.most_common(max_vocab-2):
        vocab[key] = vocab_idx
        vocab_idx += 1

    for key, value in vocab.items():
        reverse_vocab[value] = key

    end = timer()

    print("Processed reviews from XML using 'tree.findall' in %s time." % (end - start))
    print("Number of reviews processed: %d" % number_of_reviews)
    print("Number of words in vocab %d" % len(vocab))

    return reviews, vocab, reverse_vocab


def preprocess_sentence(sentence):
    cleaned_sentence = []

    for word in sentence:
        word = word.strip().lower()
        if word == None or word in STOP_WORDS or word == '' or len(word) <= 2:
            continue
        else:
            cleaned_sentence.append(word)

    return cleaned_sentence