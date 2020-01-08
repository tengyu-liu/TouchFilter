import os

exp = 'static_z2_nobn_unitz2-0016-0'

for i in range(10):
    for j in range(24):
        os.system('ffmpeg -i figs/%s-%d.pkl-%d-%%d.png -filter_complex scale=480:-1,tile=10x1 figs/%s-%d.pkl-%d.png'%(exp, i, j, exp, i, j))
        for k in range(10):
            os.remove('figs/%s-%d.pkl-%d-%d.png'%(exp, i, j, k))