import os

for exp in ['dynamic_z2_nobn_unitz2', 'static_z2_nobn_unitz2', 'dynamic_z2_physical_prior_nobn_unitz2', 'static_z2_physical_prior_nobn_unitz2']:
    os.makedirs('../figs/%s'%exp, exist_ok=True)
    for epoch in range(60):
        for batch in [0]:
            os.system('scp antelope:/home/tengyu/github/TouchFilter/code/figs/%s/%04d-%d.pkl ../figs/%s/'%(exp, epoch, batch, exp))
            if os.path.exists(os.path.join(os.path.dirname(__file__), '../figs/%s/%04d-%d.pkl'%(exp, epoch, batch))):
                os.system('python visualize.py %s %d %d'%(exp, epoch, batch))

