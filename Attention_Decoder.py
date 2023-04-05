import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionDecoder(nn.Module):
    def __init__(self, device, hidden_size, output_vocab, max_length, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_vocab
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        self.embedding_decoder = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.MultiheadAttention(self.hidden_size, num_heads=1, batch_first=True)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2*self.hidden_size, self.output_size)
        self.bidirectional = True

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded_decoder = self.embedding_decoder(input)
        embedded_decoder = self.dropout(embedded_decoder)
        attn_applied, _ = self.attn(embedded_decoder, encoder_outputs, encoder_outputs)
        output = torch.cat((embedded_decoder, attn_applied), -1)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        output = F.softmax(self.out(output), dim=-1)
        output = torch.log(output)
        return output, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1+int(self.bidirectional), batch_size, self.hidden_size, device = device)
        cell = torch.zeros(1+int(self.bidirectional), batch_size, self.hidden_size, device = device)
        return hidden, cell

# obj = AttentionDecoder(device=torch.device("cpu"), hidden_size=1024, output_vocab=66, max_length=10)
# hidden, cell = obj.init_hidden(3)
# print(hidden.shape)
# input = torch.randint(40, (3, 1))
# enc_out = torch.rand(3, 10, 1024)
# print(input)
# out, _, _ = obj(input, hidden, cell, enc_out)
# print(out.shape)
