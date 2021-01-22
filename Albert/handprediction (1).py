import os
import sys
import torch
import smplx
from math import asin, sqrt

default = torch.tensor([
	[ 0.09566994,  0.00638343,  0.0061863 ],
	[ 0.00757268,  0.00118307,  0.0268723 ],
	[-0.02510622,  0.00519243,  0.02908936],
	[-0.04726213,  0.003894  ,  0.02897524],
	[ 0.00100949,  0.00490447,  0.00282876],
	[-0.03017318,  0.00676579, -0.00276574],
	[-0.05307782,  0.00551369, -0.00671026],
	[ 0.02688296, -0.0035569 , -0.03702303],
	[ 0.00986855, -0.00349507, -0.04952181],
	[-0.00599835, -0.00418623, -0.05985371],
	[ 0.01393437,  0.00242601, -0.02048689],
	[-0.0143799 ,  0.00449301, -0.02558542],
	[-0.03790042,  0.0028049 , -0.03321924],
	[ 0.07158022, -0.00913891,  0.03199915],
	[ 0.05194698, -0.00824762,  0.0556987 ],
	[ 0.02972924, -0.01368059,  0.07022282]
]).float()

removeYMtx = torch.tensor([[1,0,0],[0,1,0],[0,0,0]]).float()
removeZMtx = torch.tensor([[1,0,0],[0,0,0],[0,0,1]]).float()


def batch_rodrigues(rot_vecs, epsilon: float = 1e-8): 

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3)).float()

    zeros = torch.zeros((batch_size, 1)).float()
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def firstJointBatch(vertices):

	batch_size = vertices.shape[0]

	default_vertices = default[2::3] - default[1::3]
	actual_vertices = vertices[:,2::3] - vertices[:,1::3]

	nextYVerts = torch.matmul(actual_vertices, removeYMtx)
	nextYVals = torch.asin(torch.div(nextYVerts, torch.norm(nextYVerts, dim=[2, -1]).view(batch_size, 5, 1)))[:,:,1]
	nextDYVerts = torch.matmul(default_vertices, removeYMtx)
	nextDYVals = torch.asin((nextDYVerts.T / torch.norm(nextDYVerts, dim=1)).T)[:,1]

	nextZVerts  = torch.matmul(actual_vertices, removeZMtx)
	nextZVals = torch.asin(torch.div(nextZVerts, torch.norm(nextZVerts, dim=[2, -1]).view(batch_size, 5, 1)))[:,:,2]
	nextDZVerts = torch.matmul(default_vertices, removeZMtx)
	nextDZVals = torch.asin((nextDZVerts.T / torch.norm(nextDZVerts, dim=1)).T)[:,2]

	return torch.transpose(torch.stack((torch.zeros(batch_size, 5).float(), nextZVals - nextDZVals, nextDYVals - nextYVals), dim=-2), -2, -1)

def secondJointBatch(vertices, firstJointRes):

	batch_size = vertices.shape[0]

	default_vertices = default[3::3] - default[2::3]
	actual_vertices = vertices[:,3::3] - vertices[:,2::3]

	adjusted_actual = torch.matmul(actual_vertices.reshape(
		[batch_size, 5, 1, 3]),
		batch_rodrigues(firstJointRes.reshape([-1, 3])
		).float().reshape([batch_size, 5, 3, 3])).reshape([batch_size, 5, 3])

	nextYVerts = torch.matmul(adjusted_actual, removeYMtx)
	nextYVals = torch.asin(torch.div(nextYVerts, torch.norm(nextYVerts, dim=[2, -1]).view(batch_size, 5, 1)))[:,:,1]
	nextDYVerts = torch.matmul(default_vertices, removeYMtx)
	nextDYVals = torch.asin((nextDYVerts.T / torch.norm(nextDYVerts, dim=1)).T)[:,1]

	nextZVerts  = torch.matmul(adjusted_actual, removeZMtx)
	nextZVals = torch.asin(torch.div(nextZVerts, torch.norm(nextZVerts, dim=[2, -1]).view(batch_size, 5, 1)))[:,:,2]
	nextDZVerts = torch.matmul(default_vertices, removeZMtx)
	nextDZVals = torch.asin((nextDZVerts.T / torch.norm(nextDZVerts, dim=1)).T)[:,2]

	return torch.transpose(torch.stack((torch.zeros(batch_size, 5).float(), nextZVals - nextDZVals, nextDYVals - nextYVals), dim=-2), -2, -1)

def computeTheta(vertices):
	batch_size = vertices.shape[0]
	first = firstJointBatch(vertices)
	second = secondJointBatch(vertices, first)
	result = torch.zeros(batch_size, 15, 3)
	result[:,::3] = first
	result[:,1::3] = second	
	return result.reshape([batch_size, 45])

def createModels(batch_size=1):
	model = smplx.create(model_path="./models/MANO_RIGHT.pkl", 
	                    model_type="mano", 
	                    is_rhand= True,
	                    ext="pkl",
	                    create_hand_pose=True,
	                    hand_pose = torch.randn([batch_size, 6]) * -1,
	                    flat_hand_mean=True,
	                    plotting_module="matplotlib",
	                    batch_size=batch_size,
	                    use_pca=True)
	output = model()
	return output.joints

if __name__=='__main__':
	if len(sys.argv) != 1	:
		joints = createModels(int(sys.argv[1]))
	else:
		joints = createModels(1)
	print(joints.shape)
	print(computeTheta(joints))