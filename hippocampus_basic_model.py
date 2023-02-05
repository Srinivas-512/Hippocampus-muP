import torch
import torch.nn as nn 
import torch.nn.functional as F
from flipflop_neuron_pytorch import FlipFlop
import Dataset_generator


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.ff_hidden_size = input_size*2
        self.linear1 = nn.Linear(self.ff_hidden_size+self.input_size, input_size)
        self.ff1 = FlipFlop(self.ff_hidden_size, 1)
        self.ff2 = FlipFlop(self.ff_hidden_size, 1)
        self.linear2 = nn.Linear(self.ff_hidden_size, input_size)
        self.activation = nn.Sigmoid()
    
    def forward(self, input, output, ff2_out, ff1_hidden, ff2_hidden):
        x = self.linear1(torch.cat((ff2_out, input), 1))
        x = torch.cat((x, output),1)
        x, ff1_hidden = self.ff1(x, ff1_hidden)
        x, ff2_hidden = self.ff2(x, ff2_hidden)
        ff2_out = x.clone()
        x = self.activation(self.linear2(x))
        return x, ff1_hidden, ff2_hidden, ff2_out
    
    def hidden_init(self):
        out = torch.zeros((1, self.input_size))
        ff2_out = torch.zeros((1, self.ff_hidden_size))
        ff1_hidden = torch.zeros((1,self.ff_hidden_size))
        ff2_hidden = torch.zeros((1,self.ff_hidden_size))
        return out, ff2_out, ff1_hidden, ff2_hidden
    
# obj = Model(7)
# input = torch.rand(1,7)
# obj.hidden_init()
# out, ff1_hidden, ff2_hidden = obj(input)
# print(out)
# print(ff1_hidden)
# print(ff2_hidden)

data = Dataset_generator.trainDataGenerator(100, 5)
model = Model(7)
out, ff2_out, ff1_hidden, ff2_hidden = model.hidden_init()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

torch.autograd.set_detect_anomaly(True)

for epoch in range(10):
    for i in range(len(data)//2):
        x_key = "X"+str(i+1)
        y_key = "Y"+str(i+1)
        x = data[x_key]
        y = data[y_key]
        loss = torch.zeros(x.shape[0])
        for j in range(x.shape[0]):
            input = x[j].clone().reshape(1,-1)
            out, ff1_hidden, ff2_hidden, ff2_out = model(input, out, ff2_out, ff1_hidden, ff2_hidden)
            loss[j] = loss_function(out.reshape(-1), y[j])
        loss = torch.mean(loss)
        print(f"Loss = {loss}")
        loss.backward(retain_graph = True)  
        optimizer.step()
