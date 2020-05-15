import bcolz
import pickle
import torch.nn as nn
import numpy as np
from collections import Counter

class Glove():
    def __init__(self, pad_token='<pad>'):
        self.emb_dim = 300

        vectors = bcolz.open('glove.6B/glove.6B.300d.dat')[:]
        self.words = pickle.load(open('glove.6B/glove.6B.300d_words.pkl', 'rb'))

        self.pad_token = pad_token
        self.words = [pad_token] + self.words
        vectors = np.concatenate([np.zeros((1, self.emb_dim)), vectors])

        self.word2idx = {o:i for i, o in enumerate(self.words)}
        self.idx2words = {i:o for i, o in enumerate(self.words)}

        self.embedding = {w: vectors[self.word2idx[w]] for w in self.words if w != pad_token}

    # TODO: I think this method is very shitty
    def create_custom_embedding(self, sentences):
        # remove words that appear only once (likely typo)
        words = Counter()
        for sentence in sentences:
            for word in sentence.split(" "):
                words.update([word.lower()]) # lower case
        self.words = {k:v for k,v in words.items() if v > 1}
        
        # sort words => most common words first
        self.words = sorted(words, key=words.get, reverse=True)

        # add back pad and unk token
        self.words = [self.unk_token, self.pad_token] + self.words

    # TODO: I think this method is very shitty
    def add_to_embedding(self, tokens):
        for token in tokens:
            self.embedding[token] = np.random.normal(scale=0.6, size=(self.emb_dim, ))
            if token not in self.words:
                self.words.append(token)

        self.word2idx = {o:i for i, o in enumerate(self.words)}
        self.idx2words = {i:o for i, o in enumerate(self.words)}

    @property
    def unk_token(self):
        return '<unk>' # given by GloVe

    @property
    def weights_matrix(self):
        weights_matrix = np.zeros((len(self.idx2words), self.emb_dim))

        for i, word in self.idx2words.items():
            try: 
                weights_matrix[i] = self.embedding[word]
            except KeyError: # found an unknown word
                weights_matrix[i] = self.embedding[self.unk_token]
                self.embedding[word] = weights_matrix[i]

        return weights_matrix


    def sentence_to_indices(self, sentence, pad=True, seq_len=-1):
        # if pad is False, seq_length is ignored
        if pad and seq_len <= 0:
            raise Exception('Invalid seq_len. Must be positive integer.')

        word_list = sentence.split(" ")
        word_list = [self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_token] for word in word_list]

        if pad and seq_len - len(word_list) > 0:
            word_list.extend([self.word2idx[self.pad_token]] * (seq_len - len(word_list)))

        return word_list


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    emb_layer.weight = nn.Parameter(weights_matrix)

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer
 