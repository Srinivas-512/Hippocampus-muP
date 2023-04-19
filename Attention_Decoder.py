import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def softmax(preds):
  temperature = 0.01
  print(preds.shape)
  ex = torch.exp(preds/temperature)
  print(ex.shape)
  print( torch.sum(ex, axis=1).shape)
  return ex / (torch.sum(ex, axis=1).unsqueeze(1))

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len = 11):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         print(position.shape)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         print(div_term.shape)
#         pe = torch.zeros(1,max_len, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor, shape ``[batch_Size, seq_len, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(1)]
#         return self.dropout(x)

# x = torch.randn((32,11,128))
# pos = PositionalEncoding(d_model = 10000, max_len=10)
# print(x.shape)
# x = pos(x)
# print(x.shape)

class AttentionDecoder(nn.Module):
    def __init__(self, device, hidden_size, output_vocab,embed_size, max_length, dropout_p=0.1, num_layers = 1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_vocab
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device
        self.num_layers = num_layers
        self.embedding_decoder = nn.Embedding(self.output_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.energy = nn.Linear(hidden_size*4,1)
        self.softmax = nn.Softmax(dim = 0) #Need to think about the dimension here
        self.relu = nn.ReLU()


        self.rnn = nn.LSTM( 2*hidden_size+self.embed_size, hidden_size, num_layers = 1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2*self.hidden_size, self.output_size)
        self.out_activation = nn.LogSoftmax(dim=-1)
        self.bidirectional = True

        
        # self.encoder_init_hidden = torch.zeros((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size, device = device)

    def forward(self, input, hidden, cell, encoder_outputs):
        # self.encoder_init_hidden = encoder_hidden
        embedded_decoder = self.embedding_decoder(input)
        embedded_decoder = self.dropout(embedded_decoder)

        encoder_outputs = torch.permute(encoder_outputs,(1,0,2)) #(seq_len,N,hidden_size*2) 
        sequence_length = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        hidden_size = encoder_outputs.shape[2]


        #Encoder Outputs : (11,32,512)
        #Hidden : (2,32,256)
        temp = hidden.transpose(0,2).contiguous()
        hidden_reshape = temp.reshape(1,batch_size,hidden_size)
        h_reshaped = hidden_reshape.repeat(sequence_length,1,1)
        energy = self.relu(self.energy(torch.cat((h_reshaped,encoder_outputs), dim = 2)))

        encoder_outputs = torch.permute(encoder_outputs,(1,0,2))

        attention = self.softmax(energy)
  
        attention = attention.permute(1,2,0)

        context_vector = torch.bmm(attention,encoder_outputs)
        rnn_input = torch.cat((context_vector,embedded_decoder), dim = 2)

        
        #encoder_outputs : (32,11,512)
        #attention : (32,11,1)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = F.relu(self.out(output))
        output = F.log_softmax(output,dim=-1)
        
        
        
        if(torch.isnan(output).any() == True):
            print(f"Decoder Input = {input.T}")
            print(f"Embedded Decoder = {embedded_decoder} ")
            print(f"attn_applied = {attn_applied} ")
            print(f"output1 = {output1} ")
            print(f"output2 = {output2} ")
            print(f"output3 = {output3} ")
            print(f"output4 = {output4} ")
            print(f"output5 = {output5} ")
            print(f"output6 = {output6} ")
            print(f"Min output6 = {torch.min(output6,dim=-1)} ")
            print(f"Max output6 = {torch.max(output6,dim=-1)} ")
            print(f"output = {output} ")
            print(f"Decoder Input has nan = {torch.isnan(input).any()}")
            print(f"Embedded Decoder has nan= {torch.isnan(embedded_decoder).any()} ")
            print(f"attn_applied has nan= {torch.isnan(attn_applied).any()} ")
            print(f"output1 has nan= {torch.isnan(output1).any()} ")
            print(f"output2 has nan = {torch.isnan(output2).any()} ")
            print(f"output3 has nan= {torch.isnan(output3).any()} ")
            print(f"output4 has nan= {torch.isnan(output4).any()} ")
            print(f"output5 has nan= {torch.isnan(output5).any()} ")
            print(f"output6 has nan= {torch.isnan(output6).any()} ")
            print(f"output has nan= {torch.isnan(output).any()} ")
            exit(1)
        return output, hidden, cell

    def init_hidden(self,encoder_hidden, encoder_cell):
        hidden = encoder_hidden
        cell = encoder_cell
        return hidden, cell