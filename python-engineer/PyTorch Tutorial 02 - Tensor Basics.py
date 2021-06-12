#2Tensor basics

import torch
import numpy as np

#creating empty tensor
x = torch.empty(2, 3)
# print(x)

#creating tenspr from random values
x = torch.rand(3, 4)
# print(x)

x = torch.zeros(3, 4)
# print(x)

x = torch.ones(3, 4)
# print(x)

#We can give specific datatype
x = torch.ones(3, 4, dtype=torch.float32)
# print(x)

#tensor from python lists
list_test = [1, 5, 6, 9, 10]
x = torch.tensor(list_test, dtype=torch.float32)
# print(x)

#Elementvise addition

x = torch.rand(2, 2)
y = torch.rand(2, 2)
z = x + y
print(x)
print(y)
print(z)
w = torch.add(x,y)
print(w)

# _ trailing function in pytorch do a inplace  operation

y.add_(x)
print(y)

##################################
#MUL
print("ELEMENTWISE MUL----------------------------")

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
# z = x * y
z = torch.mul(x,y)
print(z)

#elementwise divisio also can be done

##################################
#SLICING
print("Slicing----------------------------")

x = torch.rand(5, 3)
print(x)
print(x[1,2])   #Returns a single tensor
print(x[1,2].item())  # when single tensor returns we can use .item() method get value


##################################
#Reshaping a tensor
print("Reshaping a tensor----------------------------")
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)

#
y = x.view(-1, 8)  #-1 if you want to put value in one dimention you can use this method. (-1 for unknown dimention)
#python automatically determine right side of it
print(y)
print(y.size())


##################################
#Numpy to tensor ----> visewersa
print("tensor to numpy----------------------------")

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
#carefull can share same memory
a.add_(1)
print(a)
print(b)

print("numpy to tensor----------------------------")
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a +=1
print(a)
print(b)  #tesnsor get modified too

########RUN ON GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) #Create on th gpu
    y = torch.ones(5)
    y = y.to(device) #OR first create tensor and move to GPU
    #y.numpy() will give error here numpy only can handle cpu tensors (convert to cpy y.to("cpu")