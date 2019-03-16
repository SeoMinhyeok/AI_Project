# -*- coding: utf-8 -*-
import torch
import set_variable

batch_size = set_variable.batch_size

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 40, kernel_size=3) # ,padding=1,stride=3)  O = (W-K + 2P) / S + 1
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(40, 30, kernel_size=3) # ,stride=31)
        # 이 부분을 고쳐 줘야 한다.


        self.conv3 = torch.nn.Linear(17280, 10)

    def forward(self, x):


       # print('input',x.size())
        x = self.conv1(x)

       # print("conv1 ", x.size())
        x = self.relu(x)
        x = self.conv2(x)

       # print("conv2", x.size())
        x = self.relu(x)
        x = x.view(batch_size, -1)


       # print("x", x.size())
        x = self.conv3(x)

       # print("conv3", x)

       # print("last x size", x.size())

        return x

