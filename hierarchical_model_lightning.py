import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import f1_score, recall_score, precision_score

from glove import Glove
import utils

class HierarchicalModel(LightningModule):
    
    def __init__(self, input_dim, hidden_dim1, embedding, criterion,
                 hidden_dim2=None, output_dim=1, drop_prob=0.5, mask_val=-1,
                 data_num=1, max_words=1790, max_posts=20, batch_size=20):
        super(HierarchicalModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2 if hidden_dim2 is not None else hidden_dim1

        self.embedding = embedding

        self.lstm1 = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim1,
                             batch_first=True) # (batch, seq, feature)
        
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim1,
                             hidden_size=self.hidden_dim2,
                             batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=self.hidden_dim2, out_features=output_dim)
        self.sigmoid = nn.Sigmoid()

        self.criterion = criterion

        self.mask_val = mask_val

        self.data_num = data_num
        self.max_words = max_words
        self.max_posts = max_posts
        self.batch_size = batch_size

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.accumulated_training_loss = 0

    def forward(self, inputs, words_lengths, posts_lengths):
        batch_size, max_posts, max_words = inputs.size()
        
        inputs = self.embedding(inputs.long())
        
        inputs = inputs.view(batch_size * max_posts, max_words, -1)
        words_lengths = words_lengths.view(-1)

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, words_lengths, batch_first=True, enforce_sorted=False)
        outputs1, _ = self.lstm1(inputs)

        outputs1, _ = nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True, total_length=max_words)

        words_lengths = words_lengths.tolist()
        assert len(words_lengths) == batch_size * max_posts
        outputs1 = outputs1[range(batch_size * max_posts):,[l-1 for l in words_lengths],:]
        assert outputs1.size() == (batch_size * max_posts, self.hidden_dim1)

        outputs1 = outputs1.view(batch_size, max_posts, self.hidden_dim1)

        outputs1 = nn.utils.rnn.pack_padded_sequence(outputs1, posts_lengths, batch_first=True, enforce_sorted=False)
        outputs2, _ = self.lstm2(outputs1)

        outputs2, _ = nn.utils.rnn.pad_packed_sequence(outputs2, batch_first=True, total_length=max_posts)

        outputs = self.dropout(outputs2)
        outputs = self.fc(outputs)
        outputs = self.sigmoid(outputs)

        return outputs

    def prepare_data(self):
        test, train, val = utils.load_test_train_val(self.data_num) # df

        train_texts = list(train.posts)

        glove = Glove()
        glove.create_custom_embedding([word for text in train_texts for word in text.split()])

        self.train_tuple = utils.process_data(train, glove, self.max_words, self.max_posts)
        self.test_tuple = utils.process_data(test, glove, self.max_words, self.max_posts)
        self.val_tuple = utils.process_data(val, glove, self.max_words, self.max_posts)

    def train_dataloader(self):
        t = self.train_tuple
        return utils.to_data_loader(t[0], t[1], t[2], t[3], self.batch_size)

    def val_dataloader(self):
        v = self.val_tuple
        return utils.to_data_loader(v[0], v[1], v[2], v[3], self.batch_size)

    def test_dataloader(self):
        t = self.train_tuple
        return utils.to_data_loader(t[0], t[1], t[2], t[3], self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)

    def training_step(self, batch, batch_idx):
        inputs, labels, wl, pl = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        predictions = self.forward(inputs, wl, pl)

        loss, _, _ = self._loss(predictions, labels, pl)

        # I think batch_idx starts from 0
        training_loss = (self.accumulated_training_loss + loss) / ((batch_idx + 1) * self.batch_size)
        logs = {'train_loss': training_loss}
        
        # 'loss' is required for backward()
        return {'loss': loss, 'log': logs} 

    def validation_step(self, batch, batch_idx):
        inputs, labels, wl, pl = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        predictions = self.forward(inputs, wl, pl)
        loss, predictions, truths = self._loss(predictions, labels, pl)

        return {'val_loss': loss, 'predictions': predictions.tolist(), 'truths': truths.tolist()}

    def validation_epoch_end(self, outputs):
        truths = []
        predictions = []

        for x in outputs:
            truths.append(x['truths'])
            predictions.append(x['predictions'])

        truths = [int(truth) for truthlist in truths for truth in truthlist]
        predictions = [int(pred) for predlist in predictions for pred in predlist]
        
        f1 = f1_score(truths, predictions)
        precision = precision_score(truths, predictions)
        recall = recall_score(truths, predictions)
        logs = {'val_f1': f1, 'val_precision': precision, 'val_recall': recall}
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # 'avg_val_loss' is used to determine the best model
        return {'avg_val_loss': avg_loss, 'log': logs}

    def _loss(self, predictions, truths, posts_lengths):
        assert predictions.size() == truths.size()

        batch_size, seq_len = predictions.size()

        truths = truths.view(-1)
        predictions = predictions.view(-1)

        mask = (truths > self.mask_val).float()
        truths = truths * mask

        # extract out non_masked values
        indices = []
        for i, post_length in enumerate(posts_lengths):
            for j in range(post_length):
                indices.append(i * batch_size + j)

        truths = truths[indices]
        predictions = predictions[indices]

        a = 0
        for length in posts_lengths:
            a += length
        print(f"sanity_check: {len(truths)}, {len(predictions)}, {a}")

        loss = self.criterion.loss(predictions.float(), truths.float())
        
        return loss, torch.round(predictions), truths
