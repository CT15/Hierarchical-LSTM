import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader


def load_test_train_val(number):
    # number is int from 1 to 5
    base_dir = f'./separated_data/data{number}'
    test = pd.read_csv(f'{base_dir}/test.csv', comment='#')
    train = pd.read_csv(f'{base_dir}/train.csv', comment='#')
    val = pd.read_csv(f'{base_dir}/val.csv', comment='#')
    return test, train, val


def process_data(df, glove, max_words, max_posts):
    # wl => words_lengths; pl => posts_lengths
    text_indices, wl, pl, final_labels = [], [], [], []
    
    for id_ in df.thread_id.unique():
        temp = df[df.thread_id == id_]
        texts, labels = list(temp.posts), list(temp.new_labels)

        text_sub_ind, sub_wl, sub_labels = [], [], []
        
        for text, label in zip(texts, labels):
            word_list = glove.sentence_to_indices(text, seq_len=max_words)
            assert len(word_list) == max_words
            text_sub_ind.append(word_list)
            sub_wl.append(len(text.split()))
            sub_labels.append(label)

        assert len(text_sub_ind) <= max_posts
        assert len(sub_labels) == len(text_sub_ind)
        assert len(sub_wl) == len(text_sub_ind)

        if len(text_sub_ind) < max_posts:
            for i in range(max_posts - len(text_sub_ind)):
                post_padding = [glove.word2idx[glove.pad_token]] * max_posts
                text_sub_ind.append(post_padding)
                sub_labels.append(-1)
                sub_wl.append(1)
        
        assert len(sub_labels) == max_posts

        text_indices.append(text_sub_ind)
        final_labels.append(sub_labels)
        wl.append(sub_wl)
        pl.append(len(temp))

    return text_indices, final_labels, wl, pl


def to_data_loader(indices, labels, wl, pl, batch_size):
    indices, labels = np.array(indices), np.array(labels)
    wl, pl = np.array(wl), np.array(pl)
    data = TensorDataset(torch.from_numpy(indices).type('torch.FloatTensor'),
                         torch.from_numpy(labels),
                         torch.from_numpy(wl), torch.from_numpy(pl))

    return DataLoader(data, shuffle=False, batch_size=batch_size, drop_last=True)


def evaluate(model, data_loader):
    a = []
    b = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.eval()

    for inputs, labels, wl, pl in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs, wl, pl)
        _, predictions, truths = model.loss(predictions, labels, pl)
        
        predictions = predictions.tolist()
        truths = truths.tolist()

        print(f'sanity_check2: {len(predictions)}, {len(truths)}')
        a.append(predictions)
        b.append(truths)

    a = [int(pred) for predlist in a for pred in predlist]
    b = [int(truth) for truthlist in b for truth in truthlist]

    print(f'sanity_check3: {len(a)}, {len(b)}')
    model.train()

    f1 = f1_score(b, a)
    precision = precision_score(b, a)
    recall = recall_score(b, a)
    accuracy = accuracy_score(b, a)
    conf_matrix = confusion_matrix(b, a)

    return f1, precision, recall, accuracy, conf_matrix
