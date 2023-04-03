import numpy as np
import random
import torch
import torch.nn.functional as F

dataSizeHere = 5
trainDataLength = 10
max_size = 8


# ### Sequential Input Generator
# 
# The function described generates a sequence of X. It stores random data of size dataSize, (which is 5 bits here and can be changed above), until the time step reaches storeLength. Then, it performs a "Store Terminate" operation (opcode 01) before retrieving the stored data for the same duration as the store operation. The available opcodes are:
# 
# 1. Store - 00
# 2. Store Terminate - 01
# 3. Retrieve - 10
#X generator

def generator_X(storeLength, dataSize):
    
    #Store
    X = np.zeros(shape=(1, dataSize+2), dtype = float)
    for i in range(storeLength):
        Y = [0] * (dataSize+2)
        Y = np.array(Y)
        for k in range(dataSize):
            Y[2+k] = random.getrandbits(1)
        X = np.vstack((X,Y))
    X = np.delete(X , 0, 0)
    
    #End Store
    X = np.vstack(( X, np.array([0,1] + [0]*dataSize) ))
    
    #Retrieve
    for i in range(storeLength):
        Y = [1,0] + [0]*dataSize
        Y = np.array(Y)
        X = np.vstack((X,Y))   
    return X

#Y generator based on the X generated

def generator_Y(X, dataSize):
    Y = np.zeros(shape=(1, dataSize+2), dtype = float)
    for i in range(X.shape[0]):
        if ((X[i][0:2] == [0,0]).all()):
            J = [0] * (dataSize+2)
            J = np.array(J)
            Y = np.vstack((Y,J))
        elif ((X[i][0:2] == [0,1]).all()):
            J = [0] * (dataSize+2)
            J = np.array(J)
            Y = np.vstack((Y,J))
        elif ((X[i][0:2] == [1,0]).all()):
            Y = np.vstack(( Y, X[-(i+1)].reshape(1, -1) ))
    Y = np.delete(Y , 0, 0)
    return Y
    
    
#Y = generator_Y(X, dataSizeHere)
# print('X=',X)
# print('Y=',Y)
#X_Train data generator

def trainDataGenerator(trainDataLength, dataSize):
    trainData = {}
    
    for i in range(trainDataLength):
        storeLengthHere = 8
        X_temp = generator_X(storeLengthHere, dataSize)
        trainData['X'+str(i+1)] = X_temp
        trainData['Y'+str(i+1)] = generator_Y(X_temp, dataSize)
        
    for i in range(trainDataLength) :
        data1 = trainData['X'+str(i+1)]
        data2 = trainData['Y'+str(i+1)]
        tensor1 = torch.tensor(data1).type(torch.float32)
        tensor2 = torch.tensor(data2).type(torch.float32)
        trainData['X'+str(i+1)] = tensor1
        trainData['Y'+str(i+1)] = tensor2
    return trainData
        
# trainData = trainDataGenerator(trainDataLength, dataSizeHere)



#Converting dictionary to a tensor after padding
def dict_to_tensor(trainData):
    l = len(trainData)//2
    inputs = torch.zeros((l,2*max_size+1,dataSizeHere+2))
    outputs = torch.zeros((l,2*max_size+1,dataSizeHere+2))
    pad_indices = np.zeros((1,l))
    for i in range(l):
        pad_indices[0][i] = (trainData['X'+str(i+1)]).shape[0] 
    pad_indices = pad_indices.reshape(-1)

    for i in range(l):
        if (len(trainData['X'+str(i+1)]) < 2*max_size+1):
            l = len(trainData['X'+str(i+1)])
            for j in range(l,2*max_size+1):
                zeros_row = torch.zeros((1, dataSizeHere+2))
                trainData['X'+str(i+1)]= torch.cat((trainData['X'+str(i+1)], zeros_row), dim=0)
        if (len(trainData['Y'+str(i+1)]) < 2*max_size+1):
            l = len(trainData['Y'+str(i+1)])
            for j in range(l,2*max_size+1):
                zeros_row = torch.zeros((1, dataSizeHere+2))
                trainData['Y'+str(i+1)]= torch.cat((trainData['Y'+str(i+1)], zeros_row), dim=0)
        inputs[i] = trainData['X'+str(i+1)]
        outputs[i] = trainData['Y'+str(i+1)]
         
    return inputs,outputs,pad_indices

# inputs,outputs,pad_indices = dict_to_tensor(trainData)

#Converting binary data to decimal
def binary_to_int(binary_array):
    binary_string = ''.join(str(bit.item()) for bit in binary_array)
    int_val = int(binary_string, 2)
    return float(int_val)

def binary_tensor_to_int(tensor):
    tensor = tensor.long()
    num_rows = tensor.size(0)
    int_tensor = torch.zeros(num_rows, dtype=torch.int64)
    for i in range(num_rows):
        int_val = binary_to_int(tensor[i])
        int_tensor[i] = int_val
    return int_tensor


def binary_to_decimal(trainData):
    no_of_inputs = len(trainData)//2

    inputs = torch.zeros((no_of_inputs, 2*max_size+1))
    outputs = torch.zeros((no_of_inputs,2*max_size+1))
    for i in range(no_of_inputs):
        if (len(trainData['X'+str(i+1)]) < 2*max_size+1):
            l = len(trainData['X'+str(i+1)])
            for j in range(l,2*max_size+1):
                zeros_row = torch.zeros((1, dataSizeHere+2))
                trainData['X'+str(i+1)]= torch.cat((trainData['X'+str(i+1)], zeros_row), dim=0)
        if (len(trainData['Y'+str(i+1)]) < 2*max_size+1):
            l = len(trainData['Y'+str(i+1)])
            for j in range(l,2*max_size+1):
                zeros_row = torch.zeros((1, dataSizeHere+2))
                trainData['Y'+str(i+1)]= torch.cat((trainData['Y'+str(i+1)], zeros_row), dim=0)
        inputs[i] = binary_tensor_to_int(trainData['X'+str(i+1)])
        outputs[i] = binary_tensor_to_int(trainData['Y'+str(i+1)])
    return inputs,outputs


def DataGenerator(trainData):
    inputs, outputs = binary_to_decimal(trainData)
    return inputs,outputs


def PairGenerator(trainData):
    inputs, outputs = DataGenerator(trainData)
    l = len(inputs)
    one_hot_outputs = torch.zeros(l,17,128)
    one_hot_tensor = F.one_hot(torch.arange(0,128))
    for i in range(l):
        k = len(outputs[i])
        for j in range(k):
            one_hot_outputs[i][j] = one_hot_tensor[int(outputs[i][j].item())]
    pairs = []
    for i in range(l):
        pairs.append((inputs[i].reshape(1,-1).type(torch.int),one_hot_outputs[i].unsqueeze(0)))
    return pairs

# data = PairGenerator(trainData)
# print(data[2][1].shape)


