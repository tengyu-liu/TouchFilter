import torch  

class EMA:
  def __init__(self, mu):
    self.mu = mu
    self.average = torch.tensor(1.0).float().cuda()

  def apply(self, x):
    for _x in x.abs():
      self.average = self.mu * _x + (1-self.mu) * self.average