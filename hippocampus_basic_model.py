import torch
import torch.nn as nn 
import torch.nn.functional as F
from flipflop_neuron_pytorch import FlipFlop

class Model(nn.Module):
    def __init__(self, input_size, FF_hidden_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.ff_hidden_size = FF_hidden_size
        self.linear1 = nn.Linear(input_size, input_size)
        self.ff1 = FlipFlop(FF_hidden_size, 1)
        self.ff2 = FlipFlop(FF_hidden_size, 1)
        self.linear2 = nn.Linear(FF_hidden_size, input_size)
    
    def hidden_init(self):
        self.out_initial = torch.zeros((1, self.input_size))
        self.ff2_out_initial = torch.zeros((1, self.input_size))
        self.ff1_hidden = torch.zeros((1,self.ff_hidden_size))
        self.ff2_hidden = torch.zeros((1,self.ff_hidden_size))

    def forward(self):
        x = self.linear1(self.ff2_out_initial)
        x = torch.cat((x, self.out_initial),1)
        x, ff1_hidden = self.ff1(x, self.ff2_hidden)
        x, ff2_hidden = self.ff2(x, self.ff2_hidden)
        x = self.linear2(x)
        return x, ff1_hidden, ff2_hidden

obj = Model(7, 14)
input = torch.rand((1,7))
obj.hidden_init()
out, ff1_hidden, ff2_hidden = obj.forward()
print(out)
print(ff1_hidden)
print(ff2_hidden)


