import torch
import torch.nn as nn

class HierarchicalModel(nn.Module):
    """
    A class used to represent hierarchical LSTM network. The first LSTM
    layer produces word level encodings. The second LSTM layer produces
    sentence/paragraph/post level encodings.

    The fully connected layer is used at the end to transform the hidden
    states of the second layer LSTM layer to vectors of desired dimension.

    ...

    Attributes
    ----------
    input_dim : int
        the number of expected features in the input
    hidden_dim1 : int
        the number of expected features in the hidden state of the first LSTM layer
    embedding: torch.nn.Embedding
        a lookup table that stores embeddings of a fixed dictionary and size.
    criterion:
        loss function
    hidden_dim2 : int
        the number of expected features in the hidden state of the second LSTM layer
        (default hidden_dim1)
    output_dim : int
        the number of expected features in the output (default 1)
    drop_prob: float
        probability of an element of the hidden state of the second LSTM layer to 
        be zeroed (default 0.5)
    mask_val: int
        the value in the tensors that will be masked out (default -1)
    lstm1: torch.nn.LSTM
        first LSTM layer
    lstm2: torch.nn.LSTM
        second LSTM layer
    dropout: torch.nn.Dropout
        dropout layer with p = drop_prob
    fc: torch.nn.Linear
        linear transformation that transforms hidden states from the second LSTM layer
        to the output vectors
    sigmoid:
        sigmoid function applied to each element of the output tensor

    Methods
    -------
    forward(inputs, words_lengths, posts_lengths)
        Computation performed at every call

    loss(predictions, truths, posts_lengths)
        Returns the loss value of the predictions as compared to the truths
    """

    def __init__(self, input_dim, hidden_dim1, embedding, criterion,
                 hidden_dim2=None, output_dim=1, drop_prob=0.5, mask_val=-1):
        """
        Parameters
        ----------
        input_dim : int
            the number of expected features in the input
        hidden_dim1 : int
            the number of expected features in the hidden state of the first LSTM layer
        embedding: Embedding
            a lookup table that stores embeddings of a fixed dictionary and size.
        criterion:
            loss function
        hidden_dim2 : int
            the number of expected features in the hidden state of the second LSTM layer
            (default hidden_dim1)
        output_dim : int
            the number of expected features in the output (default 1)
        drop_prob: float
            probability of an element of the hidden state of the second LSTM layer to 
            be zeroed (default 0.5)
        mask_val: int
            the value in the tensors that will be masked out (default -1)
        """

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


    def forward(self, inputs, words_lengths, posts_lengths):
        """Computation performed at every call

        Parameters
        ----------
        inputs:
            input tensors
        words_lengths:
            the number of words in each post
        posts_lengths:
            the number of posts in each thread
    
        Input Shape
        -----------
        inputs:
            (batch_size, max_posts, max_words) where
            max_posts = maximum number of posts in a thread and
            max_words = maximum number of words in a post
        words_lengths:
            (batch_size, max_posts) where
            max_posts = maximum number of posts in a thread
        posts_lengths:
            (batch_size)

        Output Shape
        ------------
        outputs:
            (batch_size, max_posts, output_dim) where
            max_posts = maximum number of posts in a thread
        """
       
        batch_size, max_posts, max_words = inputs.size()
        
        inputs = self.embedding(inputs.long()) # (batch_size, max_posts, max_words, feature (emb_dim))
        
        # get all the post encoding first
        inputs = inputs.view(batch_size * max_posts, max_words, -1) # (number_of_posts (batch), max_words, feature)
        words_lengths = words_lengths.view(-1)

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, words_lengths, batch_first=True, enforce_sorted=False)
        outputs1, _ = self.lstm1(inputs)

        # outputs1 of shape (no_of_posts (batch), max_words, hidden_dim1)
        outputs1, _ = nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True, total_length=max_words)

        # take out all the last hidden_state of each post
        words_lengths = words_lengths.tolist()
        assert len(words_lengths) == batch_size * max_posts
        outputs1 = outputs1[range(batch_size * max_posts):,[l-1 for l in words_lengths],:]
        assert outputs1.size() == (batch_size * max_posts, self.hidden_dim1)

        # organise posts back to threads
        outputs1 = outputs1.view(batch_size, max_posts, self.hidden_dim1)

        outputs1 = nn.utils.rnn.pack_padded_sequence(outputs1, posts_lengths, batch_first=True, enforce_sorted=False)
        outputs2, _ = self.lstm2(outputs1)

        # outputs2 of shape (batch_size, max_posts, hidden_dim2)
        outputs2, _ = nn.utils.rnn.pad_packed_sequence(outputs2, batch_first=True, total_length=max_posts)

        #outputs2 = outputs2.contiguous().view(-1, self.hidden_dim2)
        #assert outputs2.size() == (batch_size * max_posts, self.hidden_dim2)

        outputs = self.dropout(outputs2)
        outputs = self.fc(outputs)
        outputs = self.sigmoid(outputs)

        #outputs = outputs2.view(batch_size, max_posts, -1)
        #assert outputs.size() == (batch_size, max_posts)

        return outputs

    
    def loss(self, predictions, truths, posts_lengths):
        """Returns the loss value of the predictions as compared to the truths

        For convenience, this method also returns all the predictions and truths.
        Predictions and truths are useful for various calculations, for example
        F1 score, precision, recall and confusion matrix.

        ...

        Parameters
        ----------
        predictions:
            predicted values
        truths:
            ground truth values
        posts_lengths:
            the number of posts in each thread in the batch

        Input Shape
        -----------
        predictions: (batch_size, seq_len)
        truths: (batch_size, seq_len)
        posts_lengths: (batch_size)

        Output Shape
        ------------
        loss: float
        predictions: (batch_size * seq_len)
        truths: (batch_size * seq_len)

        """

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
