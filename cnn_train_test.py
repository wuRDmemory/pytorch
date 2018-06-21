import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os, cv2

root = os.path.dirname(os.path.realpath(__file__))
datum_path = os.path.join(root, "datum")

def default_loader(path):
    return cv2.imread(path, 1)

class MyDataset(Dataset):
    imgs = list()
    transform = None
    target_transform = None
    loader = default_loader
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        f = open(txt, 'r')
        self.imgs = []
        for line in f.readlines():
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            self.imgs.append((words[0], int(words[1])))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class myNet(torch.nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64*3*3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        res  = out3.view(out3.size(0), -1)
        out  = self.dense(res)
        return out

							
train_data = MyDataset(txt=os.path.join(datum_path, "train.txt"), transform=transforms.ToTensor())
test_data  = MyDataset(txt=os.path.join(datum_path, "test.txt"),  transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(dataset=test_data,  batch_size=64)

model = myNet()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    print("epoch: ", epoch)
    # train
    train_loss = 0
    train_acc = 0
    train_cnt = 0
    for batch_x, batch_y in train_loader:
        # print("train_cnt: ", train_cnt)
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_function(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_cnt += len(batch_x)
    print("Train loss: ", train_loss/len(train_data), ", acc: ", train_acc/len(train_data))

    # test
    # when you want to test your model, remember change model into eval
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_function(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        eval_correct = (pred == batch_y).sum()
        eval_acc += eval_correct.data[0]
    print("Test loss: ", eval_loss/len(test_data), ", acc: ", eval_acc/len(test_data))