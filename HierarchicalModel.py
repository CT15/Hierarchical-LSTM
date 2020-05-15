import torch
import torch.nn as nn

class HierarchicalModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 embedding, criterion, drop_prob=0.5, mask_val=-1):

        super(HierarchicalModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = embedding

        self.lstm1 = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             batch_first=True) # (batch, seq, feature)
        
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim,
                             hidden_size=self.output_dim,
                             batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=self.output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.criterion = criterion

        self.mask_val = mask_val


    '''
    inputs of shape (batch_size, max_posts, max_words)
    words_lengths of shape (batch_size, max_posts)
    posts_lengths of shape (batch_size)
    '''
    def forward(self, inputs, words_lengths, posts_lengths):
        # batch_size => number of thread in a batch
        # max_posts => maximum no of posts in a thread
        # max_words => maximum no of words in a post
        batch_size, max_posts, max_words = inputs.size()
        
        inputs = self.embedding(inputs.long()) # (batch_size, max_posts, max_words, feature (emb_dim))
        
        # get all the post encoding first
        inputs = inputs.view(batch_size * max_posts, max_words, -1) # (number_of_posts (batch), max_words, feature)
        words_lengths = words_lengths.view(-1)

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, words_lengths, batch_first=True, enforce_sorted=False)
        outputs1, _ = self.lstm1(inputs)

        # outputs1 of shape (no_of_posts (batch), max_words, hidden_dim)
        outputs1, _ = nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True, total_length=max_words)

        # take out all the last hidden_state of each post
        words_lengths = words_lengths.tolist()
        assert len(words_lengths) == batch_size * max_posts
        outputs1 = outputs1[range(batch_size * max_posts):,[l-1 for l in words_lengths],:]
        assert outputs1.size() == (batch_size * max_posts, self.hidden_dim)

        # organise posts back to threads
        outputs1 = outputs1.view(batch_size, max_posts, self.hidden_dim)

        outputs1 = nn.utils.rnn.pack_padded_sequence(outputs1, posts_lengths, batch_first=True, enforce_sorted=False)
        outputs2, _ = self.lstm2(outputs1)

        # outputs2 of shape (batch_size, max_posts, output_dim)
        outputs2, _ = nn.utils.rnn.pad_packed_sequence(outputs2, batch_first=True, total_length=max_posts)

        outputs2 = outputs2.contiguous().view(-1, self.output_dim)
        assert outputs2.size() == (batch_size * max_posts, self.output_dim)

        outputs2 = self.dropout(outputs2)
        outputs2 = self.fc(outputs2)
        outputs2 = self.sigmoid(outputs2)

        outputs = outputs2.view(batch_size, -1)
        assert outputs.size() == (batch_size, max_posts)

        return outputs

    '''
    predictions of shape (batch_size, seq_len)
    truths of shape (batch_size, seq_len)
    posts_lengths of shape (batch_size)
    '''
    def loss(self, predictions, truths, posts_lengths):
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

        loss = self.criterion.loss(predictions.float(), truths.float())
        
        return loss, torch.round(predictions), truths
