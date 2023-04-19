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
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from positional_encodings.torch_encodings import PositionalEncodingPermute1D


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
        decoder = AttentionDecoder(torch.device("cpu"), self.hidden_size, self.vocab_size,self.embed_dim, self.max_length).to(device)
        self.decoder = decoder

        #self.sos =torch.ones((self.batch_size, 1), dtype=torch.long, device=device)*127
        self.count = 0

    def forward(self, x):
        encoder_hidden, encoder_cell = self.encoder.init_hidden(self.batch_size)
        
        target_length = self.max_length
        input_length = self.max_length
        encoder_outputs = torch.zeros(self.batch_size, self.max_length, 2*self.hidden_size, device=device)
        for i in range(input_length):
            out, encoder_hidden, encoder_cell = self.encoder(x[:,i].unsqueeze(1), encoder_hidden, encoder_cell)
            encoder_outputs[:,i,:] = out.squeeze(1)
            

        # print(target_tensor)
        decoder_hidden, decoder_cell = self.decoder.init_hidden(encoder_hidden,encoder_cell)
        output_stack = []
        decoder_input = torch.ones((self.batch_size, 1), dtype=torch.long, device=device)*126
        for i in range(target_length):
            out, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            output_stack.append(out)
            decoder_input = torch.argmax(out, dim = -1)
        # exit(1)
        output_stack = torch.stack(output_stack)
        # print(output_stack.shape)
        output_stack = torch.permute(torch.squeeze(output_stack.requires_grad_(), 2), [1, 0, 2])
        # print(output_stack.shape)
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

def trainIters(pairs, n_iters, print_every=10, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    model = Model(128, 32, 256, 11, 32)
    # model.load_state_dict(torch.load('attn_translation.pt'))
    # print("Model loaded")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()
    batch_size = 32
    count = 0
    for iter in range(1, n_iters+1 ):
        training_pairs = [random.choice(pairs) for i in range(batch_size)]
        input_tensor = []
        target_tensor = []
        for i in range(batch_size):
            input_tensor.append(training_pairs[i][0])
            target_tensor.append(training_pairs[i][1])
        # target_tensor = 

        input_tensor = torch.stack(input_tensor).squeeze(1).long()
        target_tensor = torch.stack(target_tensor).squeeze(1).long()
        
        # print(input_tensor.shape)
        # with torch.autograd.detect_anomaly():
        optimizer.zero_grad()
        # loss = Dataset_generator.train(input_tensor, target_tensor, encoder,
        #              decoder, encoder_optimizer, decoder_optimizer, criterion)
        target_tensor = torch.argmax(target_tensor, dim=-1)
        
        
        # print(torch.argmax(out[1], dim = 0).shape)
        # exit(1)
        # print(out.shape)

        EOS  = torch.ones((batch_size,1))*127
        EOS = EOS
        input_tensor = torch.cat((input_tensor,EOS), dim = -1).long()
        target_tensor = torch.cat((target_tensor,EOS), dim = -1).long()
        # print(out.dtype)
        # print(target_tensor.dtype)
        out = model(input_tensor)
        out = torch.permute(out,[0,2,1])
        if (count%10 == 0):
            print(f'Input Tensor:{input_tensor[1]}')
            print(f'Target Tensor:{target_tensor[1]}')
            # print(out[1].shape)
            print(f'Out Tensor:{torch.argmax(out[1], dim = 0)}')
        count+=1
        loss = criterion(out, target_tensor)#torch.mean(torch.Tensor(loss).requires_grad_(True))
        # if (math.isnan(loss) == True):
        #     print(out)
        loss.backward()
        optimizer.step()

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every

            print_loss_total = 0
            print('%s (%d %d%%) %.7f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # torch.save(model.state_dict(), 'attn_translation.pt')
            # print("model saved")                             

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #
    #
    # plt.showPlot(plot_losses)

trainDataLength = 4096*2
max_size = 5
dataSizeHere = 5
trainData = Dataset_generator.trainDataGenerator(trainDataLength, dataSizeHere)
pairs = Dataset_generator.PairGenerator(trainData)

trainIters(pairs, 2500)
