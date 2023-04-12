import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random
import Dataset_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainDataLength = 8192*2
max_size = 5
dataSizeHere = 5
trainData = Dataset_generator.trainDataGenerator(trainDataLength, dataSizeHere)
pairs = Dataset_generator.PairGeneratorForSeq2Seq(trainData)
batch_size = 32
MAX_LENGTH = 11

SOS_token = 126
EOS_token = 127

class Encoder(nn.Module):
   def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
       super(Encoder, self).__init__()
       self.input_dim = input_dim
       self.embbed_dim = embbed_dim
       self.hidden_dim = hidden_dim
       self.num_layers = num_layers
       self.embedding = nn.Embedding(self.input_dim, self.embbed_dim)
       self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
              
   def forward(self, src):

       embedded = self.embedding(src.unsqueeze(0))
       outputs, hidden = self.gru(embedded)
       return outputs, hidden


class Decoder(nn.Module):
   def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
       super(Decoder, self).__init__()
       self.embbed_dim = embbed_dim
       self.hidden_dim = hidden_dim
       self.output_dim = output_dim
       self.num_layers = num_layers


       self.embedding = nn.Embedding(output_dim, self.embbed_dim)
       self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
       self.out = nn.Linear(self.hidden_dim, output_dim)
       self.softmax = nn.LogSoftmax(dim=1)
      
   def forward(self, input, hidden):

# reshape the input to (1, batch_size)
       # print(input.shape)
       # exit(1)
       input = input.view(1, -1)
       embedded = F.relu(self.embedding(input))
       # print(embedded.shape)
       # print(hidden.shape)
       # exit(1)
       output, hidden = self.gru(embedded, hidden)       
       prediction = self.softmax(self.out(output[0]))
      
       return prediction, hidden

class Seq2Seq(nn.Module):
   def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
       super().__init__()
      
#initialize the encoder and decoder
       self.encoder = encoder
       self.decoder = decoder
       self.device = device
     
   def forward(self, source, target, teacher_forcing_ratio=0.5):

       input_length = source.size(0) #get the input length (number of words in sentence)
       batch_size = target.shape[1] 
       target_length = target.shape[0]
       vocab_size = self.decoder.output_dim
      
#initialize a variable to hold the predicted outputs
       outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

#encode every word in a sentence
       for i in range(input_length):
           encoder_output, encoder_hidden = self.encoder(source[i])

#use the encoder’s hidden layer as the decoder hidden
       decoder_hidden = encoder_hidden.to(device)
  
#add a token before the first predicted word
       decoder_input = torch.tensor([SOS_token]*batch_size, device=device).unsqueeze(0)  # SOS

#topk is used to get the top K value over a list
#predict the output word from the current target word. If we enable the teaching force,  then the #next decoder input is the next word, else, use the decoder output highest value. 

       for t in range(target_length):   
           
           decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
           outputs[t] = decoder_output
           teacher_force = random.random() < teacher_forcing_ratio
           topv, topi = decoder_output.topk(1)
           input = (target[t] if teacher_force else topi)
           if(teacher_force == False and input[0].item() == EOS_token):
               break

       return outputs


teacher_forcing_ratio = 0.5
count = 0

def clacModel(model, input_tensor, target_tensor, model_optimizer, criterion):
   model_optimizer.zero_grad()

   input_length = input_tensor.size(0)
   loss = 0
   epoch_loss = 0
   global count

   output = model(input_tensor, target_tensor)

   num_iter = output.size(0)
   
   Tensor1 = torch.zeros((2*max_size+2,1))
   Tensor2 = torch.zeros((2*max_size+2,1))

   for ot in range(num_iter):
       Tensor1[ot] = target_tensor[ot][0]
       Tensor2[ot] = torch.argmax(output[ot], dim = -1)[0]
       loss = loss + (criterion(output[ot], target_tensor[ot])).requires_grad_(requires_grad=True)

   if count%50 == 0:
       print(f"Input Tensor: {input_tensor[:,0].T} , Target Tensor : {Tensor1.T}, Output Tensor: {Tensor2.T}")  
   loss.backward()
   count += 1
   model_optimizer.step()
   epoch_loss = loss.item() / num_iter

   return epoch_loss

def trainModel(model, pairs, num_iteration=20000):
   model.train()
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   criterion = nn.NLLLoss()
   total_loss_iterations = 0
   for iter in range(1, num_iteration+1):
        for i in range(batch_size):
            training_pairs = [random.choice(pairs) for i in range(batch_size)]
            input_tensor = []
            target_tensor = []
            for i in range(batch_size):
                input_tensor.append(training_pairs[i][0])
                target_tensor.append(training_pairs[i][1]) 
        input_tensor = torch.stack(input_tensor).T.long()
        target_tensor = torch.stack(target_tensor).T.long()
        loss = clacModel(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss
        
        if iter % 50 == 0:
            avarage_loss= total_loss_iterations / 50
            total_loss_iterations = 0
            print('%d %.4f' % (iter, avarage_loss))
            torch.save(model.state_dict(), 'mytraining.pt')
   return model
    

    
embed_size = 32
hidden_size = 256
num_layers = 4
num_iteration = 25000
input_size = 128

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(input_size, hidden_size, embed_size, num_layers)

model = Seq2Seq(encoder, decoder, device).to(device)
model = trainModel(model, pairs, num_iteration)

    
