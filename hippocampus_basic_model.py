import torch
import torch.nn as nn 
import torch.nn.functional as F
from flipflop_neuron_pytorch import FlipFlop
import Dataset_generator


class Model(nn.Module):
    def __init__(self, input_size, m):
        super(Model, self).__init__()
        self.input_size = input_size
        self.ff_hidden_size = input_size*m
        self.linear1 = nn.Linear(self.ff_hidden_size+self.input_size, self.input_size)
        self.ff1 = FlipFlop(self.input_size+self.input_size, self.ff_hidden_size)
        self.ff2 = FlipFlop(self.ff_hidden_size, self.ff_hidden_size)
        self.linear2 = nn.Linear(self.ff_hidden_size, input_size)
        self.activation = nn.Sigmoid()
        # self.activation2 = nn.Softmax()
    
    def forward(self, X):
        #(t,input_dim)
        out, ff2_out, ff1_hidden, ff2_hidden = self.hidden_init()
        outputs = torch.zeros(X.shape[0], X.shape[-1])
        #print(x.shape)

        for j in range(X.shape[0]):
            #print(x[j])
            input = X[j:j+1]#.reshape(1, -1)
            #print(input.shape)
            x = self.linear1(torch.cat((ff2_out, input), 1))
            x = torch.cat((x, out),1)
            x, ff1_hidden = self.ff1(x, ff1_hidden)
            x, ff2_hidden = self.ff2(x, ff2_hidden)
            ff2_out = x
            out = self.activation(self.linear2(x))
            outputs[j] = out

        return outputs
    
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

data = Dataset_generator.trainDataGenerator(8, 5)
model = Model(7, 10)

loss_function = nn.BCELoss()
weight_decay = 0.1
T_max = 30
optimizer = torch.optim.AdamW(model.parameters(), lr =  0.001, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

torch.autograd.set_detect_anomaly(True)

#with torch.no_grad():
for epoch in range(1000):
    for i in range(len(data)//2):
        x_key = "X"+str(i+1)
        y_key = "Y"+str(i+1)
        x = data[x_key]
        y = data[y_key]
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y)
        loss.backward(retain_graph = True)  
        optimizer.step()

        # loss = torch.mean(loss)
    if epoch%10 == 0:
        print(f"Loss = {loss}")
        #print(outputs, y)
            
