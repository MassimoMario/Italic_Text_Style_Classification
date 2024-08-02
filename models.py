import torch
import torch.nn as nn
from torch.nn import functional as F



class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes, num_filters, kernel_sizes):
        super(CNNClassifier, self).__init__()
        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in kernel_sizes])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self,x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_results]
        cat = torch.cat(pooled, dim = 1)
        out = self.fc(cat).unsqueeze(0)
        return F.softmax(out, dim=-1)
    
    

# ------------------------------------------------------------------------------------------------- #



class RNNClassifier(nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs with GRU units

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    style_dim : int, dimension of the style space within the latent space
    content_dim : int, dimension of the content space within the latent 
    vocab_size : int, number of unique tokens in the dataset
    sos_token : torch tensor of the 'start of the sequence' token
    num_layers : int, number of RNNs layers

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    reconstruction(x) : inference for reconstruction
    TST(x, new_style) : inference for Text Style Transfer
    '''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(RNNClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.RNN(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        '''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder(embedded_input)
        

        pred_label = self.fc(hn)

        return pred_label

    


# ------------------------------------------------------------------------------------------------- #



class GRUClassifier(nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs with GRU units

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    style_dim : int, dimension of the style space within the latent space
    content_dim : int, dimension of the content space within the latent 
    vocab_size : int, number of unique tokens in the dataset
    sos_token : torch tensor of the 'start of the sequence' token
    num_layers : int, number of RNNs layers

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    reconstruction(x) : inference for reconstruction
    TST(x, new_style) : inference for Text Style Transfer
    '''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(GRUClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        '''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, hn = self.encoder(embedded_input)
        

        pred_label = self.fc(hn)

        return pred_label
    

# ------------------------------------------------------------------------------------------------- #



class LSTMClassifier(nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs with GRU units

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    style_dim : int, dimension of the style space within the latent space
    content_dim : int, dimension of the content space within the latent 
    vocab_size : int, number of unique tokens in the dataset
    sos_token : torch tensor of the 'start of the sequence' token
    num_layers : int, number of RNNs layers

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    reconstruction(x) : inference for reconstruction
    TST(x, new_style) : inference for Text Style Transfer
    '''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(LSTMClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True) # (N,B,H) N batches, B sequence length, H input dim
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        '''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        _, (hn, cn) = self.encoder(embedded_input)
        

        pred_label = self.fc(hn)

        return pred_label
    

# ------------------------------------------------------------------------------------------------- #


class TClassifier(nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs with GRU units

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    style_dim : int, dimension of the style space within the latent space
    content_dim : int, dimension of the content space within the latent 
    vocab_size : int, number of unique tokens in the dataset
    sos_token : torch tensor of the 'start of the sequence' token
    num_layers : int, number of RNNs layers

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    reconstruction(x) : inference for reconstruction
    TST(x, new_style) : inference for Text Style Transfer
    '''

    def __init__(self, embedding_matrix, hidden_dim, num_layers):
        super(TClassifier, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(300, 10, batch_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)
        self.fc = nn.Linear(self.embedding_dim, 3)

    def forward(self, x):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        

        Returns
        -------
        '''

        # embedding input and GRU encoder pass
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)
        out = self.encoder(embedded_input)
        
        out = out.mean(1).unsqueeze(1)
        out = out.permute(1,0,2)
        pred_label = self.fc(out)

        return pred_label