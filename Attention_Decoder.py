import torch 
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    def __init__(self, device, hidden_size, output_vocab, max_length, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_vocab
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.MultiheadAttention(self.hidden_size, num_heads=1, batch_first=True)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2*self.hidden_size, self.output_size)
        self.bidirectional = True

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_applied, _ = self.attn(embedded, encoder_outputs, encoder_outputs)
        output = torch.cat((embedded, attn_applied), -1)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        output = F.log_softmax(self.out(output), dim=-1)
        return output, hidden, attn_applied

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1+int(self.bidirectional), batch_size, self.hidden_size)
        cell = torch.zeros(1+int(self.bidirectional), batch_size, self.hidden_size)
        return hidden, cell
    
obj = AttentionDecoder(device=torch.device("cpu"), hidden_size=1024, output_vocab=66, max_length=10)
hidden, cell = obj.init_hidden(3)
input = torch.randint(40, (3, 10))
enc_out = torch.rand(3, 10, 1024)
print(input)
out, _, _ = obj(input, hidden, cell, enc_out)
print(out.shape)
