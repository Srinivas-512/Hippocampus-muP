
import torch
import torch.nn as nn 
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer



class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=True)
        self.bidirectional = True

    def forward(self, x, hidden, cell):
        out = self.embedding(x)#.unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = torch.randn((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.randn((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell

# obj = EncoderRNN(66, 128, 512)
# hidden, cell = obj.init_hidden(3)
# input = torch.randint(40, (3, 1))
# print(input)
# out, _, _ = obj(input, hidden, cell)
# print(out.shape)


'''
Ww will use data like [1, 4, 19, 15..]
10 
lets say n data points
m batch size 
m x n
embed into 512 dim space
m x n x 128
-- encoder = '''