import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import utils

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

test, train, val = utils.load_test_train_val(DATA_NUM) # df
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer_v = SummaryWriter(f'runs/{TB_FOLDER}_val')
writer_t = SummaryWriter(f'runs/{TB_FOLDER}_train')

train_texts = list(train.posts)

print('Init GloVe embedding')
glove = Glove()
glove.create_custom_embedding([word for text in train_texts for word in text.split()])

print(len(glove.word2idx))

print('Padding and packing data into data loader')
train_indices, train_labels, train_wl, train_pl = utils.process_data(train, glove, MAX_WORDS, MAX_POSTS)
test_indices, test_labels, test_wl, test_pl = utils.process_data(test, glove, MAX_WORDS, MAX_POSTS)
val_indices, val_labels, val_wl, val_pl = utils.process_data(val, glove, MAX_WORDS, MAX_POSTS)

train_loader = utils.to_data_loader(train_indices, train_labels, train_wl, train_pl, BATCH_SIZE)
test_loader = utils.to_data_loader(test_indices, test_labels, test_wl, test_pl, BATCH_SIZE)
val_loader = utils.to_data_loader(val_indices, val_labels, val_wl, val_pl, BATCH_SIZE)

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
            f1, precision, recall, _, _ = utils.evaluate(model, val_loader)
            print(f'Evaluating now VALIDATE (f1, precision, recall): {f1}, {precision}, {recall}')

            writer_v.add_scalar('f1', f1, batch_completed, f1)
            writer_v.add_scalar('precision', precision, batch_completed)
            writer_v.add_scalar('recall', recall, batch_completed)

            writer_t.add_scalar('training_loss', training_loss / data_completed, batch_completed)

            f1, precision, recall, _, _ = utils.evaluate(model, train_loader)
            writer_t.add_scalar('f1', f1, batch_completed, f1)
            writer_t.add_scalar('precision', precision, batch_completed)
            writer_t.add_scalar('recall', recall, batch_completed)

            data_multiple += 1

f1, precision, recall, accuracy, conf_matrix = utils.evaluate(model, test_loader)
print('TEST')
print(f'F1: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'Conf matrix: {conf_matrix}')
