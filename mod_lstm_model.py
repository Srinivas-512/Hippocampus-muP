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

def accuracy(predicted_classes, target_classes):

    assert predicted_classes.shape == target_classes.shape, "Predicted classes and target classes must have the same shape."

    predicted_classes = predicted_classes.numpy()
    target_classes = target_classes.numpy()

    correct = (predicted_classes == target_classes).sum().item()
    total = target_classes.size
    accuracy = correct / total * 100

    return accuracy



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

def trainIters(pairs, n_iters, print_every=10, plot_every=10, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_val_loss_total = 0
    plot_val_loss_total = 0 

    model = Model(128, 32, 256, 11, 32)
    # model.load_state_dict(torch.load('attn_translation.pt'))
    # print("Model loaded")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()
    batch_size = 32
    count = 0
    for iter in range(1, n_iters+1 ):
        training_pairs = [random.choice(pairs) for i in range(batch_size)]
        val_pairs = [random.choice(valPairs) for i in range(batch_size)]

        input_tensor = []
        target_tensor = []
        val_input_tensor = []
        val_target_tensor = []

        for i in range(batch_size):
            input_tensor.append(training_pairs[i][0])
            target_tensor.append(training_pairs[i][1])

        input_tensor = torch.stack(input_tensor).squeeze(1).long()
        target_tensor = torch.stack(target_tensor).squeeze(1).long()
        
        
        target_tensor = torch.argmax(target_tensor, dim=-1)

        EOS  = torch.ones((batch_size,1))*127
        EOS = EOS

        input_tensor = torch.cat((input_tensor,EOS), dim = -1).long()
        target_tensor = torch.cat((target_tensor,EOS), dim = -1).long()

        for i in range(batch_size):
            val_input_tensor.append(val_pairs[i][0])
            val_target_tensor.append(val_pairs[i][1])

        val_input_tensor = torch.stack(val_input_tensor).squeeze(1).long()
        val_target_tensor = torch.stack(val_target_tensor).squeeze(1).long()
        
        
        val_target_tensor = torch.argmax(val_target_tensor, dim=-1)

        val_input_tensor = torch.cat((val_input_tensor,EOS), dim = -1).long()
        val_target_tensor = torch.cat((val_target_tensor,EOS), dim = -1).long()



        optimizer.zero_grad()
        out = model(input_tensor)
        out = torch.permute(out,[0,2,1])
        if (count%10 == 0):
            print(f'Train Input Tensor:{input_tensor[1]}')
            print(f'Train Target Tensor:{target_tensor[1]}')
            print(f'Train Out Tensor:{torch.argmax(out[1], dim = 0)}')
        count+=1
        loss = criterion(out, target_tensor)

        loss.backward()
        optimizer.step()

        val_out = model(val_input_tensor)
        val_out = torch.permute(val_out,[0,2,1])
        val_loss = criterion(val_out, val_target_tensor)

    
        

        print_loss_total += loss
        plot_loss_total += loss


        print_val_loss_total += val_loss
        plot_val_loss_total += val_loss


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_val_loss_avg = print_val_loss_total / print_every

            print_loss_total = 0
            print_val_loss_total=0

            print('%s (%d %d%%) %.7f %.7f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg , print_val_loss_avg))
            # torch.save(model.state_dict(), 'attn_translation.pt')
            # print("model saved")                             

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_val_loss_avg= plot_val_loss_total/plot_every

            train_accuracy_calc = accuracy(target_tensor[1], torch.argmax(out[1], dim = 0))
            val_accuracy_calc = accuracy(val_target_tensor[1], torch.argmax(val_out[1], dim = 0))

            plot_losses.append(plot_loss_avg.detach().numpy())
            val_losses.append(plot_val_loss_avg.detach().numpy())
            train_accuracy.append(train_accuracy_calc)
            val_accuracy.append(val_accuracy_calc)

            plot_loss_total = 0
            plot_val_loss_total=0


    #
    #
    
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(plot_losses)
    axs[0,0].set_ylabel('Train Loss')
    axs[0,1].plot(train_accuracy)
    axs[0,1].set_ylabel('Train Accuracy')
    axs[1,0].plot(val_losses)
    axs[1,0].set_ylabel('Val Loss')
    axs[1,1].plot(val_accuracy)
    axs[1,1].set_ylabel('Val Accuracy')




    # plt.plot(plot_losses)
    # plt.plot(train_accuracy)
    # plt.plot(val_accuracy)
    # plt.plot()
    # plt.legend(['Train Loss', 'Train Accuracy'])
    # plt.xlabel('Iterations')
    plt.show()
    



max_size = 5
dataSizeHere = 5

trainDataLength = 4096*2
trainData = Dataset_generator.trainDataGenerator(trainDataLength, dataSizeHere)
pairs = Dataset_generator.PairGenerator(trainData)

valDataLength = 256
valData = Dataset_generator.trainDataGenerator(valDataLength, dataSizeHere)
valPairs = Dataset_generator.PairGenerator(trainData)



trainIters(pairs, 2500)
