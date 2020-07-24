import torch
import torch.nn as nn

class TwoLinear(nn.Module):
  def __init__(self, size):
    super(TwoLinear, self).__init__()
    self.l1 = nn.Linear(size, size)
    self.b1 = nn.BatchNorm1d(size)
    self.l2 = nn.Linear(size, size)
    self.b2 = nn.BatchNorm1d(size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, x):
    y = self.dropout(self.relu(self.b1(self.l1(x))))
    y = self.dropout(self.relu(self.b2(self.l2(y))))
    return x + y

class ConvtGenerator(nn.Module):
  def __init__(self, input_size, num_channel, feature_size):
    super(ConvtGenerator, self).__init__()
    # Input is the latent vector Z.
    self.tconv1 = nn.ConvTranspose2d(input_size, feature_size*8,
        kernel_size=4, stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(feature_size*8)
    # Input Dimension: (ngf*8) x 4 x 4
    self.tconv2 = nn.ConvTranspose2d(feature_size*8, feature_size*4,
        4, 2, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(feature_size*4)
    # Input Dimension: (ngf*4) x 8 x 8
    self.tconv3 = nn.ConvTranspose2d(feature_size*4, feature_size,
        1, 1, 0, bias=False)
    self.bn3 = nn.BatchNorm2d(feature_size)
    # Input Dimension: (ngf) x 8 x 8
    self.tconv4 = nn.ConvTranspose2d(feature_size, num_channel,
        1, 1, 0, bias=False)
    #Output Dimension: (nc) x 8 x 8
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim=2)
    self.num_channel = num_channel
    self.input_size = input_size

  def forward(self, x):
    x = self.relu(self.bn1(self.tconv1(x.view([-1, self.input_size, 1, 1]))))
    x = self.relu(self.bn2(self.tconv2(x)))
    x = self.relu(self.bn3(self.tconv3(x)))
    x = self.tconv4(x)
    x = self.softmax(self.tanh(x).view([-1, self.num_channel, 8*8]))
    x = x / x.max(-1, keepdim=True)[0]
    return x.view([-1, self.num_channel, 8, 8])

class LinearGenerator(nn.Module):
  def __init__(self, num_cp, heatmap_size):
    super(LinearGenerator, self).__init__()
    self.layer = nn.Linear(1024, heatmap_size * heatmap_size * num_cp, bias=False)
    self.num_cp = num_cp
    self.heatmap_size = heatmap_size

  def forward(self, x):
    x = self.layer(x).view([-1, self.num_cp, self.heatmap_size * self.heatmap_size])
    x = x - x.min(-1, keepdim=True)[0]
    x = x / x.max(-1, keepdim=True)[0]
    return x.view([-1, self.num_cp, self.heatmap_size, self.heatmap_size])

class LinearHandContactPoint(nn.Module):
  def __init__(self, num_cp, num_handpoint):
    super(LinearHandContactPoint, self).__init__()
    self.layer = nn.Linear(1024, num_cp * num_handpoint)
    self.num_cp = num_cp
    self.num_handpoint = num_handpoint

  def forward(self, x):
    x = self.layer(x).view([-1, self.num_handpoint, self.num_cp])
    return x

class GraspPrediction(nn.Module):
  # Given an object and a grasp code, predict 
  # * 3 contact points on object
  # * 3 contact points on hand
  # * 3 friction vectors
  # * hand code
  def __init__(self, num_cp, num_handpoint, hand_code_length):
    super(GraspPrediction, self).__init__()
    self.h1 = nn.Linear(256+64, 1024)
    self.h2 = TwoLinear(1024)
    self.h3 = TwoLinear(1024)
    self.contact_point_pred = nn.Linear(1024, num_cp * 3)
    self.hand_code_pred = nn.Linear(1024, hand_code_length, bias=False)
    self.num_cp = num_cp
    self.hand_code_length = hand_code_length
  
  def forward(self, obj_code, grasp_code):
    x = torch.cat([obj_code, grasp_code], dim=-1)
    x = self.h1(x)
    x = self.h2(x)
    x = self.h3(x)
    cp = self.contact_point_pred(x).view([-1, self.num_cp, 3])
    z = self.hand_code_pred(x)
    return cp, z