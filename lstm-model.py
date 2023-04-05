import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Encoder import EncoderRNN
from Attention_Decoder import AttentionDecoder
import Dataset_generator
import matplotlib.pyplot as plt
import time
import random
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

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
        self.sos = torch.ones((self.batch_size, 1), dtype=torch.long)*127
        self.count = 0
    
    def train(self, x, target_length, input_length, target_tensor, criterion,optimizer):
        optimizer.zero_grad()
        loss = 0
        target_length = self.max_length
        input_length = self.max_length
        encoder_outputs = torch.zeros(self.batch_size, self.max_length, 2*self.hidden_size)
        loss = []
        for i in range(input_length):
            out, self.encoder_hidden, self.encoder_cell = self.encoder(x[:,i].unsqueeze(1), self.encoder_hidden, self.encoder_cell)
            encoder_outputs[:,i] = out[0,0]
        # self.decoder_hidden, self.decoder_cell = self.encoder_hidden, self.encoder_cell
        
        decoder_input = self.sos
        for i in range(target_length):
            out, self.decoder_hidden, self.decoder_cell = self.decoder(decoder_input, self.decoder_hidden, self.decoder_cell, encoder_outputs)
            topv, topi = out.topk(1, dim=-1)
            if(self.count==0):
                print(out)
            decoder_input = topi.squeeze(1).detach()
            loss.append(criterion(out.squeeze(1), target_tensor[:,i]))
        self.count += 1
        loss = torch.mean(torch.Tensor(loss).requires_grad_(True))
        loss.backward()
        optimizer.step()
        return loss, loss.item()/target_length

    # def forward(self, x, target_length, input_length, target_tensor, criterion):
    #     loss = 0
    #     target_length = self.max_length
    #     input_length = self.max_length
    #     encoder_outputs = torch.zeros(self.batch_size, self.max_length, 2*self.hidden_size)
    #     loss = []
    #     for i in range(input_length):
    #         out, self.encoder_hidden, self.encoder_cell = self.encoder(x[:,i].unsqueeze(1), self.encoder_hidden, self.encoder_cell)
    #         encoder_outputs[:,i] = out[0,0]
    #     decoder_input = self.sos

        
        


'''
input is -- > (batch size, number of numbers -- sequence length)
'''


trainDataLength = 4096
max_size = 8
dataSizeHere = 5
trainData = Dataset_generator.trainDataGenerator(trainDataLength, dataSizeHere)
pairs = Dataset_generator.PairGenerator(trainData)



def trainIters(pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    model = Model(128, 128, 256, 17, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1 ):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # print(target_tensor.shape)
        # loss = Dataset_generator.train(input_tensor, target_tensor, encoder,
        #              decoder, encoder_optimizer, decoder_optimizer, criterion)
        target_tensor = torch.argmax(target_tensor, dim=1)
        

        loss, avg = model.train(input_tensor, target_tensor.shape[0], input_tensor.shape[0], target_tensor, criterion,optimizer)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.7f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    

    plt.showPlot(plot_losses)

trainIters(pairs, 75000)