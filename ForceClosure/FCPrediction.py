import torch
import torch.nn as nn

class FCPrediction(nn.Module):
  # Given an object and a grasp code, predict 
  # * 3 contact points on object
  # * 3 contact points on hand
  # * 3 friction vectors
  # * hand code
  def __init__(self):
    super(FCPrediction, self).__init__()
    self.h1 = nn.Linear(256 + 10, 1024)
    self.h2 = nn.Linear(1024, 1024)
    self.h3 = nn.Linear(1024, 1024)
    self.h4 = nn.Linear(1024, 12, bias=False)
    self.bn = nn.BatchNorm1d(1024)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, obj_code, grasp_code):
    x = torch.cat([obj_code, grasp_code], 1)
    x = self.dropout(self.relu(self.bn(self.h1(x))))
    x = self.dropout(self.relu(self.bn(self.h2(x))))
    x = self.dropout(self.relu(self.bn(self.h3(x))))
    x = self.h4(x)
    cp_o = x.view([-1, 4, 3])
    return cp_o