import numpy as np
import random
import torch
import torch.nn.functional as F

dataSizeHere = 5
AddressSizeHere = 5
trainDataLength = 10
max_size = 6
ALU_ops = [np.array([0,1,0]), np.array([0,1,1]), np.array([1,0,0]), np.array([1,0,1]), np.array([1,1,0])]

#The available opcodes are:
# 1. Store - 000
# 2. Retrieve - 001
# 3. Add - 010
# 4. Subtract- 011
# 5. AND - 100
# 6. OR - 101
# 7. XOR - 110
# 8. Buffer - 111


def signed_binary_to_int(binary_array):
    binary_array = binary_array.astype(int)
    if binary_array[0] == 1:
        inverted_bits = 1 - binary_array
        inverted_int = int("".join(map(str, inverted_bits)), 2)
        absolute_value = inverted_int + 1
        return -absolute_value
    else:
        return int("".join(map(str, binary_array)), 2)


def int_to_signed_binary(num, bit_width):

    if bit_width <= 0:
        raise ValueError("bit_width must be greater than 0.")

    max_positive_value = 2**(bit_width - 1) - 1
    min_negative_value = -2**(bit_width - 1)
    if num > max_positive_value or num < min_negative_value:
        print(num, bit_width)
        raise ValueError("Input number is out of range for the given bit_width.")

    if num >= 0:
        binary_array = np.binary_repr(num, bit_width)
    else:
        binary_array = np.binary_repr(2**bit_width + num, bit_width)

    binary_list = list(map(int, binary_array))


    binary_array = np.array(binary_list)

    return binary_array


def ALU(x, data1, data2):
    data1 = data1.astype(int)
    data2 = data2.astype(int)
    if (np.array_equal(x[0:3], ALU_ops[0])):
        result = signed_binary_to_int(data1)+signed_binary_to_int(data2)
        result = int_to_signed_binary(result, 7)
    if (np.array_equal(x[0:3], ALU_ops[1])):
        result = signed_binary_to_int(data1)-signed_binary_to_int(data2)
        result = int_to_signed_binary(result, 7)
    if (np.array_equal(x[0:3], ALU_ops[2])):
        if (len(data1) == len(data2)):
            result = np.bitwise_and(data1, data2)
        else:
            if(len(data1) > len(data2)):
                for i in range(len(data1) - len(data2)):
                    data2 = np.insert(data2,0,0)
                result = np.bitwise_and(data1, data2)
            else:
                for i in range(len(data2) - len(data1)):
                    data1 = np.insert(data1,0,0)
                result = np.bitwise_and(data1, data2)

    if (np.array_equal(x[0:3], ALU_ops[3])):
        if (len(data1) == len(data2)):
            result = np.bitwise_or(data1, data2)
        else:
            if(len(data1) > len(data2)):
                for i in range(len(data1) - len(data2)):
                    data2 = np.insert(data2,0,0)
                result = np.bitwise_or(data1, data2)
            else:
                for i in range(len(data2) - len(data1)):
                    data1 = np.insert(data1,0,0)
                result = np.bitwise_or(data1, data2)
    if (np.array_equal(x[0:3], ALU_ops[4])):
        if (len(data1) == len(data2)):
            result = np.bitwise_xor(data1, data2)
        else:
            if(len(data1) > len(data2)):
                for i in range(len(data1) - len(data2)):
                    data2 = np.insert(data2,0,0)
                result = np.bitwise_xor(data1, data2)
            else:
                for i in range(len(data2) - len(data1)):
                    data1 = np.insert(data1,0,0)
                result = np.bitwise_xor(data1, data2)

    return result
        

def generator_X(AddressSizeHere, dataSizeHere):
    
    X = np.zeros((max_size,(3+dataSizeHere+AddressSizeHere)))
    #opcodes
    X[0:2,0:3] = np.zeros((2,3))
    X[4:6,0:2] = np.zeros((2,2))
    X[4:6,2] = np.ones((2,1)).squeeze(1)
    X[2,0:3] = random.choice(ALU_ops)
    X[3,0:3] = random.choice(ALU_ops)

    #Address and Data
    r1 = np.random.randint(0, 2, size=AddressSizeHere)
    r2 = np.random.randint(0, 2, size=AddressSizeHere)
    data1 = np.random.randint(0, 2, size=dataSizeHere)
    data2 = np.random.randint(0, 2, size=dataSizeHere)
    address_list = [r1,r2]

    X[0,3:8] = r1
    X[0,8:] = data1
    X[1,3:8] = r2
    X[1,8:] = data2
    X[2,3:8] = random.choice(address_list)
    X[2,8:] = random.choice(address_list)
    X[3,3:8] = random.choice(address_list)
    X[3,8:] = random.choice(address_list)
    X[4,3:8] = r1
    X[5,3:8] = r2

    return(X)

X = generator_X(AddressSizeHere,dataSizeHere)



def generator_Y(X, AddressSizeHere, dataSizeHere):
    max_size = X.shape[0]
    Y = np.zeros((max_size,(3+dataSizeHere+AddressSizeHere)))
    store_buffer = np.array([1]*3+[0]*10)
    Y[0] = store_buffer
    Y[1] = store_buffer
    Y[2] = store_buffer
    Y[3] = store_buffer
    r1 = X[0,3:8]
    r2 = X[1,3:8]
    data1 = X[0,8:] 
    data2 = X[1,8:]
    
    if (np.array_equal(X[2,3:8], r1) and np.array_equal(X[2,8:], r1)):
        ALU_out1 = ALU(X[2], data1, data1)
        data1 = ALU_out1
    elif (np.array_equal(X[2,3:8], r1) and np.array_equal(X[2,8:], r2)):
        ALU_out1 = ALU(X[2], data1, data2)
        data1 = ALU_out1
    elif (np.array_equal(X[2,3:8], r2) and np.array_equal(X[2,8:], r1)):
        ALU_out1 = ALU(X[2], data2, data1)
        data2 = ALU_out1
    elif (np.array_equal(X[2,3:8], r2) and np.array_equal(X[2,8:], r2)):
        ALU_out1 = ALU(X[2], data2, data2)
        data2 = ALU_out1

    if (np.array_equal(X[3,3:8], r1) and np.array_equal(X[3,8:], r1)):
        ALU_out1 = ALU(X[3], data1, data1)
        data1 = ALU_out1
    elif (np.array_equal(X[3,3:8], r1) and np.array_equal(X[3,8:], r2)):
        ALU_out1 = ALU(X[3], data1, data2)
        data1 = ALU_out1
    elif (np.array_equal(X[3,3:8], r2) and np.array_equal(X[3,8:], r1)):
        ALU_out1 = ALU(X[3], data2, data1)
        data2 = ALU_out1
    elif (np.array_equal(X[3,3:8], r2) and np.array_equal(X[3,8:], r2)):
        ALU_out1 = ALU(X[3], data2, data2)
        data2 = ALU_out1

  

    len1 = len(data1)
    len2 = len(data2)

    Y[4, -len1:] = data1
    Y[5, -len2:] = data2 

    return Y

Y = generator_Y(X, dataSizeHere,AddressSizeHere)
trainDataLength = 25

def binary_to_int(binary_array):
    binary_string = ''.join(str(bit.item()) for bit in binary_array)
    int_val = int(binary_string, 2)
    return float(int_val)

def binary_tensor_to_int(tensor):
    tensor = tensor.long()
    num_rows = tensor.size(0)
    int_tensor = torch.zeros(num_rows, dtype=torch.long)
    for i in range(num_rows):
        int_val = binary_to_int(tensor[i])
        int_tensor[i] = int_val
    return int_tensor


def PairGenerator(trainDataLength):
    pairs = []
    for i in range(trainDataLength):
        X = generator_X(AddressSizeHere,dataSizeHere)
        Y = generator_Y(X, AddressSizeHere, dataSizeHere)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        pair = (binary_tensor_to_int(X), binary_tensor_to_int(Y))
        pairs.append(pair)
    return pairs


    







