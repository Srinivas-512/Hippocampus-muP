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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        encoder = EncoderRNN(self.vocab_size, self.embed_dim, self.hidden_size).to(device)
        self.encoder = encoder
        decoder = AttentionDecoder(torch.device("cpu"), 2*self.hidden_size, self.vocab_size, self.max_length).to(device)
        self.decoder = decoder

        #self.sos =torch.ones((self.batch_size, 1), dtype=torch.long, device=device)*127
        self.count = 0

    def forward(self, x):
        encoder_hidden, encoder_cell = self.encoder.init_hidden(self.batch_size)
        decoder_hidden, decoder_cell = self.decoder.init_hidden(self.batch_size)
        target_length = self.max_length
        input_length = self.max_length
        encoder_outputs = torch.zeros(self.batch_size, self.max_length, 2*self.hidden_size, device=device)
        for i in range(input_length):
            out, encoder_hidden, encoder_cell = self.encoder(x[:,i].unsqueeze(1), encoder_hidden, encoder_cell)
            encoder_outputs[:,i] = out[0,0]
        # self.decoder_hidden, self.decoder_cell = self.encoder_hidden, self.encoder_cell
        # print(target_tensor)
        output_stack = []
        decoder_input = torch.ones((self.batch_size, 1), dtype=torch.long, device=device)*127
        for i in range(target_length):
            out, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            output_stack.append(out)
            topv, topi = out.topk(1, dim=-1)
            decoder_input = topi.squeeze(1)#.detach()
        output_stack = torch.stack(output_stack)
        output_stack = torch.permute(torch.squeeze(output_stack.requires_grad_(), 1), [1, 0, 2])
        return output_stack

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

def trainIters(pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    model = Model(128, 128, 256, 17, 1)
    #model.load_state_dict(torch.load('attn_translation.pt'))
    #print("Model loaded")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1 ):
        with torch.autograd.detect_anomaly():
            optimizer.zero_grad()
            training_pair = training_pairs[iter-1]
            input_tensor = training_pair[0].cuda()
            target_tensor = training_pair[1].cuda()
            # loss = Dataset_generator.train(input_tensor, target_tensor, encoder,
            #              decoder, encoder_optimizer, decoder_optimizer, criterion)
            target_tensor = torch.argmax(target_tensor, dim=-1)

            out = model(input_tensor)
            loss = criterion(out.squeeze(0), target_tensor.squeeze(0))#torch.mean(torch.Tensor(loss).requires_grad_(True))
            loss.backward(retain_graph=True)
            optimizer.step()

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every

            print_loss_total = 0
            print('%s (%d %d%%) %.7f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            torch.save(model.state_dict(), 'attn_translation.pt')
            print("model saved")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #
    #
    # plt.showPlot(plot_losses)

trainDataLength = 4096
max_size = 8
dataSizeHere = 5
trainData = Dataset_generator.trainDataGenerator(trainDataLength, dataSizeHere)
pairs = Dataset_generator.PairGenerator(trainData)
trainIters(pairs, 75000)
