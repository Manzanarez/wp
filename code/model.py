import torch.nn as nn
import torch

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
## GAM~ A simple lookup table that stores embeddings of a fixed dictionary and size.
## GAM ~ This module is often used to store word embeddings and retrieve them using indices.
## The input to the module is a list of indices, and the output is the corresponding word embeddings.
        self.encoder = nn.Embedding(ntoken, ninp) #GAM~  Adds weight intialized by the normal distribution
                                                # ntoken= len(corpus.dictionary) , ninp = size of word embedding
        if rnn_type in ['LSTM', 'GRU']:
#input_size – The number of expected features in the input x
#hidden_size – The number of features in the hidden state h
#num_layers – Number of recurrent layers.E.g., setting num_layers = 2 would mean stacking two RNNs together to form
# a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results.Default: 1
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout) #GAM ~ Original
##            self.rnn = getattr(nn, rnn_type)(ninp, nhid, 35, dropout=dropout) #GAM ~ it takes bptt = 35 as nlayers
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
 ##           self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken) ##GAM ~ Applies a linear transformation to the incoming data: y=xAT+by = xA^T + by=xAT+b

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

##    def forward(self, input, hidden): #GAM ~ Original
    def forward(self, input, hidden, datax , stories): #GAM ~ 14may19
        emb = self.drop(self.encoder(input)) #encoder calls nn.Embedding
## GAM - Borrar       emb = self.encoder(input )
##GAM - Borrar        emb = self.drop(emb)
        if stories == 1:
##            emb = torch.cat((emb,datax),0) #GAM ~ 14may19
            emb = torch.add(emb,datax) #GAM ~ 5jun19

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):  #GAM ~ Original
##    def init_hidden(self, bsz, bptt): #GAM ~ 15may19

        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid), ## GAM ~ Original
                    weight.new_zeros(self.nlayers, bsz, self.nhid)) ## GAM ~ Original
## GAM ~ Change first parameter (self.bptt) to have the same lenght of the embedding (Prompt or Storie)
##            return (weight.new_zeros(bptt, bsz, self.nhid), ## GAM ~ 15may19
##                    weight.new_zeros(bptt, bsz, self.nhid)) ## GAM ~ 15may19

        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
