# -*- coding: utf-8 -*-
import torch
import set_variable

batch_size = set_variable.batch_size

class CnnModel(torch.nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 40, kernel_size=3) # ,padding=1,stride=3)  O = (W-K + 2P) / S + 1
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(40, 2, kernel_size=3) # ,stride=31)
        # 이 부분을 고쳐 줘야 한다.
        self.conv3 = torch.nn.Linear(175232, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(batch_size, -1)
        x = self.conv3(x)
        x = x.view(batch_size, 2)

        return x

