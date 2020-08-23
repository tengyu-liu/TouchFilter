import shutil
import os
import json
import plotly 
import plotly.graph_objects as go
import numpy as np

path_dict = {}

for i in range(8):
    if i == 0:
        for g in os.listdir('adelm_%d/adelm_result'%i):
            for fn in os.listdir('adelm_%d/adelm_result/%s'%(i,g)):
                if '.json' in fn:
                    path_dict[fn] = 'adelm_%d/adelm_result/%s/%s.png'%(i, g, fn[:-5])
    else:
        for g in os.listdir('adelm_%d/adelm_result'%i):
            for fn in os.listdir('adelm_%d/adelm_result/%s'%(i,g)):
                if '.json' in fn:
                    if os.path.exists('adelm_%d/adelm_result/%s/%s.png'%(i, g, fn[:-5])) and 'minima' not in fn:
                        continue
                    print('adelm_%d/adelm_result/%s/%s'%(i, g, fn))
                    if fn in path_dict.keys() and 'minima' not in fn:
                        shutil.copy(path_dict[fn], 'adelm_%d/adelm_result/%s/%s.png'%(i, g, fn[:-5]))
                    else:
                        data = json.load(open('adelm_%d/adelm_result/%s/%s'%(i, g, fn)))
                        fig = go.Figure(data=data['data'], layout=data['layout'])
                        fig.write_image('adelm_%d/adelm_result/%s/%s.png'%(i, g, fn[:-5]))
                        path_dict[fn] = 'adelm_%d/adelm_result/%s/%s.png'%(i, g, fn[:-5])
