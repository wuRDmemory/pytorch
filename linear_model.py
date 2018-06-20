import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x*w

def loss(x, y):
    y_head = forward(x)
    return (y_head - y)*(y_head - y)

def gredient(x, y):
    return 2*x*(w*x - y)

w_list = list()
mse_list = list()

for time in range(10):
    print("time=",time,"\tw=", w, "\n")
    gredient_sum = 0
    loss_sum = 0
    for x, y in zip(x_data, y_data): 
        gredient_sum = gredient_sum + gredient(x, y)
        loss_sum = loss_sum + loss(x, y)
    ave_loss = loss_sum / 3.0
    ave_gredient = gredient_sum / 3.0
    w = w - 0.01*ave_gredient
    print("\tave_loss=", ave_loss, "\tave_gredient=", ave_gredient, "\n")

####################pytorch#######################
tensor_w = torch.FloatTensor([1.0])
variable_w = Variable(tensor_w, requires_grad=True)

def torch_loss(x, y):
    return (y-x*variable_w)**2

for time in range(10):
    print("time=",time,"\tw=", variable_w.data[0], "\n")
    for x, y in zip(x_data, y_data):
        torch_l = torch_loss(x, y)
        torch_l.backward()
        variable_w.data = variable_w.data - 0.01*variable_w.grad.data
        variable_w.grad.data.zero_()
    print("\tloss=", torch_l.data[0], "\tgredient=", variable_w.grad.data[0], "\n")
