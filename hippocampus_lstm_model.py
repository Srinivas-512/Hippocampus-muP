import torch 
import torch.nn as nn 
import torch.nn.functional as F
import Dataset_generator

class Hippocampus(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(Hippocampus, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.linear1 = nn.Linear(self.hidden_size+self.input_size, self.input_size)
        self.lstm1 = nn.LSTM(self.input_size+self.input_size, self.hidden_size, batch_first = False)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = False)
        self.linear2 = nn.Linear(self.hidden_size, input_size)
        self.activation = nn.Sigmoid()
    
    def forward(self, X):
        out, lstm2_out= self.hidden_init()
        outputs = torch.zeros(X.shape[0], X.shape[-1])
        for j in range(X.shape[0]):
            input = X[j:j+1]
            x = self.linear1(torch.cat((lstm2_out, input), 1))
            x = torch.cat((x, out),1)
            dummy1, (lstm1_hidden, lstm1_cell) = self.lstm1(x)
            x = lstm1_hidden[:, :]
            dummy2, (lstm2_hidden, lstm2_cell) = self.lstm2(x)
            x = lstm2_hidden[:, :]
            lstm2_out = x
            out = self.activation(self.linear2(x))
            outputs[j] = out
        return outputs
    
    def hidden_init(self):
        out = torch.zeros((1, self.input_size))
        lstm2_out = torch.zeros((1, self.hidden_size))
        return out, lstm2_out

# obj = Hippocampus(10, 5)
# input = torch.rand(1,5)
# obj.hidden_init()
# out= obj(input)
# print(out)
# print(lstm1_hidden)
# print(lstm2_hidden)
# lstm1_hidden = torch.zeros((1,self.hidden_size))
# lstm2_hidden = torch.zeros((1,self.hidden_size))     , lstm1_hidden, lstm2_hidden

data = Dataset_generator.trainDataGenerator(8, 5)
model = Hippocampus(300, 7)

loss_function = nn.BCELoss()
weight_decay = 0.1
T_max = 30
optimizer = torch.optim.AdamW(model.parameters(), lr =  0.01, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

torch.autograd.set_detect_anomaly(True)

for epoch in range(1000):
    for i in range(len(data)//2):
        optimizer.zero_grad()
        x_key = "X"+str(i+1)
        y_key = "Y"+str(i+1)
        x = data[x_key]
        y = data[y_key]
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y)
        #loss = torch.autograd.Variable(loss, requires_grad = True)
        loss.backward() #retain_graph = True)  
        optimizer.step()

    if epoch%10 == 0:
        print(f"Loss = {loss}")
            
