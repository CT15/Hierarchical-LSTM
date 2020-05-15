import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

from HierarchicalModel import HierarchicalModel
from WeightedBCELoss import WeightedBCELoss
from glove import Glove, create_emb_layer

DATA_NUM = 1
MAX_POSTS = 20
MAX_WORDS = 1790
BATCH_SIZE = 20
INTERVENED_RATIO = 0.25
EPOCHS = 1
CLIP = 5
VAL_EVERY = 200
TB_FOLDER = 'hlstm_data1_ep1'

# HELPER FUNCTIONS #

def load_test_train_val(number):
    # number is int from 1 to 5
    base_dir = f'./separated_data/data{number}'
    test = pd.read_csv(f'{base_dir}/test.csv', comment='#')
    train = pd.read_csv(f'{base_dir}/train.csv', comment='#')
    val = pd.read_csv(f'{base_dir}/val.csv', comment='#')
    return test, train, val


def process_data(df, glove):
    # wl => words_lengths; pl => posts_lengths
    text_indices, wl, pl, final_labels = [], [], [], []
    
    for id_ in df.thread_id.unique():
        temp = df[df.thread_id == id_]
        texts, labels = list(temp.posts), list(temp.new_labels)

        text_sub_ind, sub_wl, sub_labels = [], [], []
        
        for text, label in zip(texts, labels):
            word_list = glove.sentence_to_indices(text, seq_len=MAX_WORDS)
            assert len(word_list) == MAX_WORDS
            text_sub_ind.append(word_list)
            sub_wl.append(len(text.split()))
            sub_labels.append(label)

        assert len(text_sub_ind) <= MAX_POSTS
        assert len(sub_labels) == len(text_sub_ind)
        assert len(sub_wl) == len(text_sub_ind)

        if len(text_sub_ind) < MAX_POSTS:
            for i in range(MAX_POSTS - len(text_sub_ind)):
                post_padding = [glove.word2idx[glove.pad_token]] * MAX_WORDS
                text_sub_ind.append(post_padding)
                sub_labels.append(-1)
                sub_wl.append(1)
        
        assert len(sub_labels) == MAX_POSTS

        text_indices.append(text_sub_ind)
        final_labels.append(sub_labels)
        wl.append(sub_wl)
        pl.append(len(temp))

    return text_indices, final_labels, wl, pl


def to_data_loader(indices, labels, wl, pl):
    indices, labels = np.array(indices), np.array(labels)
    wl, pl = np.array(wl), np.array(pl)
    data = TensorDataset(torch.from_numpy(indices).type('torch.FloatTensor'),
                         torch.from_numpy(labels),
                         torch.from_numpy(wl), torch.from_numpy(pl))

    return DataLoader(data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


def evaluate(model, data_loader):
    a = []
    b = []

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

# END #

test, train, val = load_test_train_val(DATA_NUM) # df
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer_v = SummaryWriter(f'runs/{TB_FOLDER}_val')
writer_t = SummaryWriter(f'runs/{TB_FOLDER}_train')

train_texts = list(train.posts)

print('Init GloVe embedding')
glove = Glove()
glove.create_custom_embedding([word for text in train_texts for word in text.split()])

print(len(glove.word2idx))

print('Padding and packing data into data loader')
train_indices, train_labels, train_wl, train_pl = process_data(train, glove)
test_indices, test_labels, test_wl, test_pl = process_data(test, glove)
val_indices, val_labels, val_wl, val_pl = process_data(val, glove)

train_loader = to_data_loader(train_indices, train_labels, train_wl, train_pl)
test_loader = to_data_loader(test_indices, test_labels, test_wl, test_pl)
val_loader = to_data_loader(val_indices, val_labels, val_wl, val_pl)

print('Creating model')
embedding = create_emb_layer(torch.from_numpy(glove.weights_matrix).float().to(device))
criterion = WeightedBCELoss(zero_weight=INTERVENED_RATIO, one_weight=1-INTERVENED_RATIO)
model = HierarchicalModel(input_dim=glove.emb_dim, 
                          hidden_dim1=glove.emb_dim,
                          embedding=embedding,
                          criterion=criterion)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.zero_grad()
model.train()

batch_completed = 0

for epoch in range(1, EPOCHS+1):
    training_loss = 0
    data_completed = 0
    data_multiple = 1

    for inputs, labels, wl, pl in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        predictions = model(inputs, wl, pl)
        loss, _, _ = model.loss(predictions, labels, pl)

        training_loss += loss
        data_completed += len(inputs)
        batch_completed += 1

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), CLIP)
        optimizer.step()

        if data_completed >= data_multiple * VAL_EVERY:
            f1, precision, recall, _, _ = evaluate(model, val_loader)
            print(f'Evaluating now VALIDATE (f1, precision, recall): {f1}, {precision}, {recall}')

            writer_v.add_scalar('f1', f1, batch_completed, f1)
            writer_v.add_scalar('precision', precision, batch_completed)
            writer_v.add_scalar('recall', recall, batch_completed)

            writer_t.add_scalar('training_loss', training_loss / data_completed, batch_completed)

            f1, precision, recall, _, _ = evaluate(model, train_loader)
            writer_t.add_scalar('f1', f1, batch_completed, f1)
            writer_t.add_scalar('precision', precision, batch_completed)
            writer_t.add_scalar('recall', recall, batch_completed)

            data_multiple += 1

f1, precision, recall, accuracy, conf_matrix = evaluate(model, test_loader)
print('TEST')
print(f'F1: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'Conf matrix: {conf_matrix}')
