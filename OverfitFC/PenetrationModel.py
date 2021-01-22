import torch
import torch.nn as nn
from HandModel import HandModel
from ObjectModel import ObjectModel

class PenetrationModel:
  def __init__(self, hand_model: HandModel, object_model: ObjectModel):
    self.hand_model = hand_model
    self.object_model = object_model
    self.relu = nn.ReLU()

  def get_penetration(self, obj_code, hand_code):
    hand_vertices = self.hand_model.get_vertices(hand_code)
    h2o_distances = self.object_model.distance(obj_code, hand_vertices)
    penetration = self.relu(h2o_distances).sum((-1,-2))
    return penetration

  def get_penetration_from_verts(self, obj_code, hand_vertices):
    h2o_distances = self.object_model.distance(obj_code, hand_vertices) # B x V x 1
    penetration = self.relu(h2o_distances).squeeze(-1)
    return penetration

  def get_max_penetration(self, obj_code, hand_code):
    hand_vertices = self.hand_model.get_vertices(hand_code)
    h2o_distances = self.object_model.distance(obj_code, hand_vertices)
    penetration = self.relu(h2o_distances).max(-1)[0]
    return penetration.squeeze()