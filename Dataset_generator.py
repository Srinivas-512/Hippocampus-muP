#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import random
import pandas
import torch

dataSizeHere = 5
trainDataLength = 100


# ### Sequential Input Generator
# 
# The function described generates a sequence of X. It stores random data of size dataSize, (which is 5 bits here and can be changed above), until the time step reaches storeLength. Then, it performs a "Store Terminate" operation (opcode 01) before retrieving the stored data for the same duration as the store operation. The available opcodes are:
# 
# 1. Store - 00
# 2. Store Terminate - 01
# 3. Retrieve - 10

# In[28]:


#X generator

def generator_X(storeLength, dataSize):
    
    #Store
    X = np.zeros(shape=(1, dataSize+2), dtype = np.uint8)
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



# ### Sequential Output Generator
# 
# The below function generates the sequence of outputs Y based on a given X, mimicking the stack operation.

# In[29]:


#Y generator based on the X generated

def generator_Y(X, dataSize):
    Y = np.zeros(shape=(1, dataSize+2), dtype = np.uint8)
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
    
    
Y = generator_Y(X, dataSizeHere)
# print('X=',X)
# print('Y=',Y)


            
            
        


# ### Train Data Generator
# The following function generates a dictionary containing "trainDataLength" pairs of randomly generated X and Y. Giving us the required data set for training our model.

# In[30]:


#X_Train data generator

def trainDataGenerator(trainDataLength, dataSize):
    trainData = {}
    
    for i in range(trainDataLength):
        storeLengthHere = random.randint(1,10)
        X_temp = generator_X(storeLengthHere, dataSize)
        trainData['X'+str(i+1)] = X_temp
        trainData['Y'+str(i+1)] = generator_Y(X_temp, dataSize)
        
    for i in range(trainDataLength) :
        data1 = trainData['X'+str(i+1)]
        data2 = trainData['Y'+str(i+1)]
        tensor1 = torch.tensor(data1)
        tensor2 = torch.tensor(data2)
        trainData['X'+str(i+1)] = tensor1
        trainData['Y'+str(i+1)] = tensor2
    return trainData
        
trainData = trainDataGenerator(trainDataLength, dataSizeHere)
trainData


# In[ ]:




