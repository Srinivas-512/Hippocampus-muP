import torch
import numpy as np
import torch.nn as nn

class FlipFlop(nn.Module):

    def __init__(self, units, **kwargs):
        super(FlipFlop, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.j_h = nn.Linear(7, self.units)
        self.j_x = nn.Linear(7, self.units)
        self.k_h = nn.Linear(7, self.units)
        self.k_x = nn.Linear(7, self.units)
        self.activation = nn.Sigmoid()
    
    def build(self, input_shape):
        self.built = True

    def forward(self, inputs, states):
        prev_output = states[0]
        j = self.activation(self.j_x(inputs) + self.j_h(prev_output))
        k = self.activation(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]
    
obj = FlipFlop(2)
states = torch.rand((1,7))
inputs = torch.rand((1,7))
output, _ = obj(inputs, states)
print(output)