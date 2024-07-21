
import random

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

# Set the device
device_ = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current Device: {device_}")

# Set the device globally
# torch.set_default_device(device_)
class mlp(nn.Module):

  def __init__(self):
    super(mlp,self).__init__()

    self.fc1 = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280*14,1000),
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(1000,1000),
        nn.Sigmoid(),
        nn.Linear(1000,100),
        nn.Sigmoid(),
        nn.Linear(100,2),
        nn.Sigmoid(),
        nn.Linear(2,2),
        nn.ReLU()
    )


  def forward(self,x):

    x = x.view(x.size()[0],-1)
    x = self.fc1(x.to(torch.float32))

    return x

class cnn(nn.Module):

  def __init__(self):
    super(cnn,self).__init__()

    self.conv = nn.Sequential(
      nn.Dropout(0.1),
      nn.BatchNorm1d(14),
      nn.Conv1d(14,14,kernel_size=32,stride=2,padding=0), # OUTPUT: C=14, L=609
      nn.ReLU(),
      nn.AvgPool1d(kernel_size=2,stride=1), # OUTPUT: C=14, L=608
      nn.BatchNorm1d(14),
      nn.Dropout(0.1),
      nn.Conv1d(14,14,kernel_size=2,stride=2,padding=0), #OUTPUT: C=14, L=304
      nn.ReLU(),
      nn.AvgPool1d(kernel_size=2,stride=1), # OUTPUT: C=14, L=303
      nn.BatchNorm1d(14),
      nn.Dropout(0.1),
      nn.Conv1d(14,14,kernel_size=3,stride=2,padding=0), #OUTPUT: C=14, L=151
      nn.ReLU(),
      nn.AvgPool1d(kernel_size=4,stride=7), # OUTPUT: C=14, L=22
      nn.BatchNorm1d(14),
      nn.Dropout(0.1),
    )

    self.fc = nn.Sequential(
        # nn.Dropout(0.05),
        nn.Linear(14*22,98),
        nn.ReLU(),
        nn.BatchNorm1d(98),
        nn.Linear(98,50),
        nn.ReLU(),
        nn.BatchNorm1d(50),
        nn.Linear(50,25),
        nn.ReLU(),
        nn.BatchNorm1d(25),
        nn.Linear(25,2),
        nn.ReLU()
    )

  def forward(self,x):
    x = torch.transpose(x,1,2)
    x = self.conv(x)
    x = x.view(x.size()[0],-1)
    x = self.fc(x)

    return x
  
class cnn_emotion_predictor(nn.Module):

  def __init__(self):
    super(cnn_emotion_predictor,self).__init__()

    self.conv = nn.Sequential(
      nn.Dropout(0.1),
      nn.BatchNorm1d(14),
      nn.Conv1d(14,14,kernel_size=32,stride=2,padding=0), # OUTPUT: C=14, L=609
      nn.ReLU(),
      nn.AvgPool1d(kernel_size=2,stride=1), # OUTPUT: C=14, L=608
      nn.BatchNorm1d(14),
      nn.Dropout(0.1),
      nn.Conv1d(14,14,kernel_size=2,stride=2,padding=0), #OUTPUT: C=14, L=304
      nn.ReLU(),
      nn.AvgPool1d(kernel_size=2,stride=1), # OUTPUT: C=14, L=303
      nn.BatchNorm1d(14),
      nn.Dropout(0.1),
      nn.Conv1d(14,14,kernel_size=3,stride=2,padding=0), #OUTPUT: C=14, L=151
      nn.ReLU(),
      nn.AvgPool1d(kernel_size=4,stride=7), # OUTPUT: C=14, L=22
      nn.BatchNorm1d(14),
      nn.Dropout(0.1),
    )

    self.fc = nn.Sequential(
        # nn.Dropout(0.05),
        nn.Linear(14*22,98),
        nn.ReLU(),
        nn.BatchNorm1d(98),
        nn.Linear(98,50),
        nn.ReLU(),
        nn.BatchNorm1d(50),
        nn.Linear(50,25),
        nn.ReLU(),
        nn.BatchNorm1d(25),
        nn.Linear(25,4),
        nn.Softmax()
    )

  def forward(self,x):
    x = torch.transpose(x,1,2)
    x = self.conv(x)
    x = x.view(x.size()[0],-1)
    x = self.fc(x)

    return x


def r_squared(predicted, target):
    # Compute the mean of the target tensor
    target_mean = torch.mean(target)

    # Compute the total sum of squares (TSS)
    tss = torch.sum((target - target_mean) ** 2)

    # Compute the residual sum of squares (RSS)
    rss = torch.sum((target - predicted) ** 2)

    # Compute R-squared
    r_squared = 1 - (rss / tss)

    return r_squared
