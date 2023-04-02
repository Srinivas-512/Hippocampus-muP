import torch
import torch.nn as nn 
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.bidirectional = True

    def forward(self, x):
        #out = self.embedding(x)#.unsqueeze(1)
        out = x
        print(out.shape)
        out, (hidden, cell) = self.rnn(out)
        out = self.rnn(x)
        return out, hidden, cell
    
    # def init_hidden(self, batch_size):
    #     hidden = torch.zeros(1+int(self.bidirectional), batch_size, self.hidden_size)
    #     cell = torch.zeros(1+int(self.bidirectional), batch_size, self.hidden_size)
    #     return hidden, cell

obj = Encoder(66, 512, 2)
input = torch.randint(2, (16, 17, 7))
print(input)
out, _, _ = obj(input)
print(out.shape)