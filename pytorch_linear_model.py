import torch
import torch.nn as nn
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_head = self.linear(x)
        return y_head

    def get_weight(self):
        return self.linear.weight

    def get_bias(self):
        return self.linear.bias

model = Model()

criterion = nn.MSELoss(size_average=False)
# criterion = nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    # use model predict
    y_head = model(x_data)
    # calc error
    loss = criterion(y_head, y_data)
    # 
    print("epoch: ", epoch, "\tloss: ", loss.data, "\n")
    # reset grad variables
    optimizer.zero_grad()
    # backward
    loss.backward()
    optimizer.step()

print("w: ", model.get_weight())
print("b: ", model.get_bias())

        
