import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Encoder import EncoderRNN
from Attention_Decoder import AttentionDecoder

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, max_length, batch_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.batch_size = batch_size
        encoder = EncoderRNN(self.vocab_size, self.embed_dim, self.hidden_size)
        self.encoder = encoder
        decoder = AttentionDecoder(torch.device("cpu"), 2*self.hidden_size, self.vocab_size, self.max_length)
        self.decoder = decoder
        self.encoder_hidden, self.encoder_cell = encoder.init_hidden(self.batch_size)
        self.decoder_hidden, self.decoder_cell = decoder.init_hidden(self.batch_size)
        self.sos = torch.ones((self.batch_size, 1), dtype=torch.long)*65
    
    def forward(self, x, target_length, input_length, target_tensor, criterion):
        loss = 0
        target_length = self.max_length
        input_length = self.max_length
        encoder_outputs = torch.zeros(self.batch_size, self.max_length, 2*self.hidden_size)
        for i in range(input_length):
            out, self.encoder_hidden, self.encoder_cell = self.encoder(x[:,i].unsqueeze(1), self.encoder_hidden, self.encoder_cell)
            encoder_outputs[:,i] = out[0,0]
        # self.decoder_hidden, self.decoder_cell = self.encoder_hidden, self.encoder_cell
        decoder_input = self.sos
        for i in range(target_length):
            out, self.decoder_hidden, self.decoder_cell = self.decoder(decoder_input, self.decoder_hidden, self.decoder_cell, encoder_outputs)
            topv, topi = out.topk(1, dim=-1)
            decoder_input = topi.squeeze(1).detach()
            print(out.shape)
            loss += criterion(out, target_tensor[:,i])
        
        return loss, loss.item() / target_length

c = nn.MSELoss()
obj = Model(66, 128, 512, 10, 3)
input = torch.randint(40, (3, 10))
output = torch.randint(40, (3, 10))
loss, avg = obj.forward(input, 10, 10, output, c)
print(avg)
'''
input is -- > (batch size, number of numbers -- sequence length)
'''

# obj = Hippocampus(10, 5)
# input = torch.rand(1,5)
# obj.hidden_init()
# out= obj(input)
# print(out)
# print(lstm1_hidden)
# print(lstm2_hidden)
# lstm1_hidden = torch.zeros((1,self.hidden_size))
# lstm2_hidden = torch.zeros((1,self.hidden_size))     , lstm1_hidden, lstm2_hidden

# data = Dataset_generator.trainDataGenerator(8, 5)
# model = Hippocampus(300, 7)

# loss_function = nn.BCELoss()
# weight_decay = 0.1
# T_max = 30
# optimizer = torch.optim.AdamW(model.parameters(), lr =  0.01, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

# torch.autograd.set_detect_anomaly(True)

# for epoch in range(1000):
#     for i in range(len(data)//2):
#         optimizer.zero_grad()
#         x_key = "X"+str(i+1)
#         y_key = "Y"+str(i+1)
#         x = data[x_key]
#         y = data[y_key]
        
#         optimizer.zero_grad()
#         outputs = model(x)
#         loss = loss_function(outputs, y)
#         #loss = torch.autograd.Variable(loss, requires_grad = True)
#         loss.backward() #retain_graph = True)  
#         optimizer.step()

#     if epoch%10 == 0:
#         print(f"Loss = {loss}")
            
