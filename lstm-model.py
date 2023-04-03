import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Encoder import EncoderRNN
from Attention_Decoder import AttentionDecoder
import Dataset_generator
import matplotlib.pyplot as plt
import time
import random

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
            loss += criterion(out.squeeze(1), target_tensor[:,i])
        
        # loss.backward()
        # optimizer.step()
        return loss, loss.item()/target_length

'''
input is -- > (batch size, number of numbers -- sequence length)
'''


trainDataLength = 10
max_size = 8
dataSizeHere = 5
trainData = Dataset_generator.trainDataGenerator(trainDataLength, dataSizeHere)
pairs = Dataset_generator.PairGenerator(trainData)
#print(data)


def trainIters(pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    model = Model(128, 128, 512, 17, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # loss = Dataset_generator.train(input_tensor, target_tensor, encoder,
        #              decoder, encoder_optimizer, decoder_optimizer, criterion)

        loss, avg= model.forward(input_tensor, target_tensor.shape[0], input_tensor.shape[0], target_tensor, criterion)
        loss.backward()
        optimizer.step()
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    plt.showPlot(plot_losses)

trainIters(pairs, 75000)