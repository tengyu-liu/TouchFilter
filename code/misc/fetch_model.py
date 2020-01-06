import os

exps = ['dynamic_z2_nobn', 'static_z2_nobn', 'dynamic_z2_physical_prior_nobn', 'static_z2_physical_prior_nobn']

exp = 'static_z2_nobn'
epoch = 43
batch = 0

os.makedirs(os.path.join(os.path.dirname(__file__), '../models', exp), exist_ok=True)
os.system('scp antelope:/home/tengyu/github/TouchFilter/code/models/%s/checkpoint %s'%(exp, os.path.join(os.path.dirname(__file__), '../models', exp)))
os.system('scp antelope:/home/tengyu/github/TouchFilter/code/models/%s/%04d-%d* %s'%(exp, epoch, batch, os.path.join(os.path.dirname(__file__), '../models', exp)))
