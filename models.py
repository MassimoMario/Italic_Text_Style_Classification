import torch
import torch.nn as nn
from torch.nn import functional as F



class CNNClassifier(nn.Module):
    ''' Class of a CNN Classifier for text, made up of Conv2d layers

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    num_classes : int, number of classes
    num_filters : int, number of filters in the Conv2d layer
    kernel_sizes : list of int, sizes of kernels in Conv2d layers

    Methods
    ----------
    forward(x) : forward pass of the Classifier'''

    def __init__(self, embedding_matrix, num_classes, num_filters, kernel_sizes):
        super(CNNClassifier, self).__init__()
        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in kernel_sizes])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self,x):
        ''' Forward pass function
        
        Input
        ----------
        x : 2D torch tensor tensor, input sentence with shape [Batch size, Sequence length]
        
        Returns
        ----------
        out : 2D torch tensor with probabilities for every class'''

        # Word Embedding
        x = self.embedding(x)
        x = x.unsqueeze(1)

        # Convolution layers and Max pool
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_results]

        # Concatenation of pooled output and Linear layer to num_class dimensions
        cat = torch.cat(pooled, dim = 1)
        out = self.fc(cat).unsqueeze(0)

        return F.softmax(out, dim=-1)
    
    

# ------------------------------------------------------------------------------------------------- #



class RNNClassifier(nn.Module):
    ''' Class of a RNN Classifier for text, made up of a Recursive Neural Network

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, number of dimensions of hidden states
    num_layers : int, number of RNN layers

    Methods
    ----------
    forward(x) : forward pass of the Classifier'''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(RNNClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.RNN(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        ''' Forward pass  function

        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        pred_labels : 2D torch tensor with probabilities for every class'''

        # Word Embedding mbedding input and  RNN pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder(embedded_input)
        
        # Predicted labels from the last hidden state of the RNN
        pred_label = self.fc(hn)

        return pred_label

    


# ------------------------------------------------------------------------------------------------- #



class GRUClassifier(nn.Module):
    ''' Class of a GRU Classifier for text, made up of a Recursive Neural Network with GRU cells

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, number of dimensions of hidden states
    num_layers : int, number of RNN layers

    Methods
    ----------
    forward(x) : forward pass of the Classifier'''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(GRUClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        ''' Forward pass  function
        
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        pred_labels : 2D torch tensor with probabilities for every class'''

        # Word Embedding input and GRU forward pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder(embedded_input)
        
        # Predicted labels from last hidden state of the GRU
        pred_label = self.fc(hn)

        return pred_label
    

# ------------------------------------------------------------------------------------------------- #



class LSTMClassifier(nn.Module):
    ''' Class of a LSTM Classifier for text, made up of a Recursive Neural Network with LSTM cells

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, number of dimensions of hidden states
    num_layers : int, number of RNN layers

    Methods
    ----------
    forward(x) : forward pass of the Classifier'''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(LSTMClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        ''' Forward pass  function
        
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        pred_labels : 2D torch tensor with probabilities for every class'''

        # Word Embedding input and LSTM forward pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, (hn, cn) = self.encoder(embedded_input)
        

        # Predicted labels from last hidden state of the LSTM
        pred_label = self.fc(hn)

        return pred_label
    

# ------------------------------------------------------------------------------------------------- #


class TClassifier(nn.Module):
    ''' Class of a Transformer Classifier for text, made up of a Transformer Encoder

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding

    Methods
    ----------
    forward(x) : forward pass of the Classifier'''

    def __init__(self, embedding_matrix):
        super(TClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, 10, batch_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)
        self.fc = nn.Linear(self.embedding_dim, 3)

    def forward(self, x):
        ''' Forward pass  function
        
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        pred_labels : 2D torch tensor with probabilities for every class'''

        # Word Embedding input and Transformer Encoder forward pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        out = self.encoder(embedded_input)
        
        # Mean over Sequence length dimension
        out = out.mean(1).unsqueeze(1)
        out = out.permute(1,0,2)

        # Predicted labels from last hidden state of the GRU
        pred_label = self.fc(out)

        return pred_label