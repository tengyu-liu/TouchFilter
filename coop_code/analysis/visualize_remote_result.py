import os
import pickle
import sys

import numpy as np

sys.path.append('..')
from utils.viz_util import Visualizer

visualizer = Visualizer()

name = 'exp1'
epoch = 34

data = pickle.load(open(os.path.join('../logs/logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
obj_id, gen_hand, syn_hand, GE, SE = data
visualizer.visualize_distance(3, gen_hand[0], 'gen')
visualizer.visualize_distance(3, syn_hand[0], 'syn')
