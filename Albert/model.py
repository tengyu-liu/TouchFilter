import torch
from torch import nn

from handprediction import Joint2MANO
from manopth.manolayer import ManoLayer
from smplx import MANO, lbs

class TwoLinear(nn.Module):
  def __init__(self, size):
    super(TwoLinear, self).__init__()
    self.l1 = nn.Linear(size, size)
    self.b1 = nn.BatchNorm1d(size)
    self.l2 = nn.Linear(size, size)
    self.b2 = nn.BatchNorm1d(size)
    self.relu = nn.ReLU()
    self.do1 = nn.Dropout(0.5)
    self.do2 = nn.Dropout(0.5)
  
  def forward(self, x):
    # x: B x L
    y = self.do1(self.relu(self.b1(self.l1(x))))
    y = self.do2(self.relu(self.b2(self.l2(y))))
    return y + x

class Vertex2MANO(nn.Module):
    def __init__(self, z_size=10, hidden_size=1024):
        super(Vertex2MANO, self).__init__()
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(z_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 16*3)
        self.two_linear_1 = TwoLinear(hidden_size)
        self.two_linear_2 = TwoLinear(hidden_size)
        self.relu = nn.ReLU()
        self.mano = MANO(model_path='models/MANO_RIGHT.pkl')
        self.j2m = Joint2MANO(device='cuda')
    
    def forward(self, z):
        hidden = self.relu(self.linear_in(z))
        hidden = self.two_linear_1(hidden)
        hidden = self.two_linear_2(hidden)
        vertices = self.linear_out(hidden).reshape([-1,778,3])
        joints = lbs.vertices2joints(self.mano.J_regressor, vertices)
        mano_parameter = self.j2m.computeTheta(joints)
        return vertices, mano_parameter

