import torch
import torch.nn as nn

class FlipFlop(nn.Module):

    def __init__(self, input_size, units, **kwargs):
        super(FlipFlop, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.input_size = input_size
        self.j_h = nn.Linear(self.units, self.units) # jk of flip flop
        self.j_x = nn.Linear(self.input_size, self.units) # basicaaly models lstm with some gru features
        self.k_h = nn.Linear(self.units, self.units) # units is no of neurons basically
        self.k_x = nn.Linear(self.input_size, self.units) # units --> hyperparameter
        self.activation = nn.Sigmoid()

    def forward(self, inputs, states):
        prev_output = states[0]
        j = self.activation(self.j_x(inputs) + self.j_h(prev_output))
        k = self.activation(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]

# obj = FlipFlop(5,10)
# t = torch.rand((1,5))
# s = torch.rand((1,10))
# print(obj(t, s))