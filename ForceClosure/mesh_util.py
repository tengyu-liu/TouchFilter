import numpy as np

def load_faces(f):
  f = open(f)
  lines = f.readlines()
  faces = []
  for l in lines:
    l = l.strip().split(' ')
    if l[0] == 'f':
      faces.append([int(l[1]), int(l[2]), int(l[3])])
  return np.array(faces)-1

def load_verts(f):
  f = open(f)
  lines = f.readlines()
  verts = []
  for l in lines:
    l = l.strip().split(' ')
    if l[0] == 'f':
      verts.append([float(l[1]), float(l[2]), float(l[3])])
  return np.array(verts)
