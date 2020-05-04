import os
from config import flags
from utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dl = DataLoader(flags)
for obj_id, item_id, obs_hand, obs_obj in dl.fetch():
    print(obj_id, item_id, obs_hand)
    os.system('pause')