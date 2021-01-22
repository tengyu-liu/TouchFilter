import os
import sys
import torch
from math import asin, sqrt

class Joint2MANO:
	def __init__(self, device='cpu'):
		self.device = device
		self.default = torch.tensor([
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
		], requires_grad=True, dtype=torch.float, device=device)

		self.removeYMtx = torch.tensor([[1,0,0],[0,1,0],[0,0,0]], requires_grad=True, dtype=torch.float, device=device)
		self.removeZMtx = torch.tensor([[1,0,0],[0,0,0],[0,0,1]], requires_grad=True, dtype=torch.float, device=device)

	def batch_rodrigues(self, rot_vecs, epsilon: float = 1e-8): 

		batch_size = rot_vecs.shape[0]
		device, dtype = rot_vecs.device, rot_vecs.dtype

		angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
		rot_dir = rot_vecs / angle

		cos = torch.unsqueeze(torch.cos(angle), dim=1)
		sin = torch.unsqueeze(torch.sin(angle), dim=1)

		# Bx1 arrays
		rx, ry, rz = torch.split(rot_dir, 1, dim=1)
		K = torch.zeros((batch_size, 3, 3), dtype=torch.float, device=self.device)

		zeros = torch.zeros((batch_size, 1), dtype=torch.float, device=self.device)
		K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
			.view((batch_size, 3, 3))

		ident = torch.eye(3, device=self.device).unsqueeze(dim=0)
		rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
		return rot_mat

	def firstJointBatch(self, vertices):

		batch_size = vertices.shape[0]

		default_vertices = self.default[2::3] - self.default[1::3]
		actual_vertices = vertices[:,2::3] - vertices[:,1::3]

		nextYVerts = torch.matmul(actual_vertices, self.removeYMtx)
		nextYVals = torch.asin(torch.div(nextYVerts, torch.norm(nextYVerts, dim=2).view(batch_size, 5, 1)))[:,:,1]
		nextDYVerts = torch.matmul(default_vertices, self.removeYMtx)
		nextDYVals = torch.asin((nextDYVerts.T / torch.norm(nextDYVerts, dim=1)).T)[:,1]

		nextZVerts  = torch.matmul(actual_vertices, self.removeZMtx)
		nextZVals = torch.asin(torch.div(nextZVerts, torch.norm(nextZVerts, dim=2).view(batch_size, 5, 1)))[:,:,2]
		nextDZVerts = torch.matmul(default_vertices, self.removeZMtx)
		nextDZVals = torch.asin((nextDZVerts.T / torch.norm(nextDZVerts, dim=1)).T)[:,2]

		return torch.transpose(torch.stack((torch.zeros(batch_size, 5 , dtype=torch.float, device=self.device), nextZVals - nextDZVals, nextDYVals - nextYVals), dim=-2), -2, -1)

	def secondJointBatch(self, vertices, firstJointRes):

		batch_size = vertices.shape[0]

		default_vertices = self.default[3::3] - self.default[2::3]
		actual_vertices = vertices[:,3::3] - vertices[:,2::3]

		adjusted_actual = torch.matmul(actual_vertices.reshape(
			[batch_size, 5, 1, 3]),
			self.batch_rodrigues(firstJointRes.reshape([-1, 3])
			).float().reshape([batch_size, 5, 3, 3])).reshape([batch_size, 5, 3])

		nextYVerts = torch.matmul(adjusted_actual, self.removeYMtx)
		nextYVals = torch.asin(torch.div(nextYVerts, torch.norm(nextYVerts, dim=2).view(batch_size, 5, 1)))[:,:,1]
		nextDYVerts = torch.matmul(default_vertices, self.removeYMtx)
		nextDYVals = torch.asin((nextDYVerts.T / torch.norm(nextDYVerts, dim=1)).T)[:,1]

		nextZVerts  = torch.matmul(adjusted_actual, self.removeZMtx)
		nextZVals = torch.asin(torch.div(nextZVerts, torch.norm(nextZVerts, dim=2).view(batch_size, 5, 1)))[:,:,2]
		nextDZVerts = torch.matmul(default_vertices, self.removeZMtx)
		nextDZVals = torch.asin((nextDZVerts.T / torch.norm(nextDZVerts, dim=1)).T)[:,2]

		return torch.transpose(torch.stack((torch.zeros(batch_size, 5, dtype=torch.float, device=self.device), nextZVals - nextDZVals, nextDYVals - nextYVals), dim=-2), -2, -1)

	def computeTheta(self, vertices):
		batch_size = vertices.shape[0]
		first = self.firstJointBatch(vertices)
		second = self.secondJointBatch(vertices, first)
		result = torch.zeros(batch_size, 15, 3, dtype=torch.float, device=self.device)
		result[:,::3] = first
		result[:,1::3] = second
		return result.reshape([batch_size, 45])


if __name__ == '__main__':
	import torch
	import smplx
	import numpy as np
	batch_size = 12
	j2m = Joint2MANO()
	mano = smplx.create(model_path="./models/MANO_RIGHT.pkl", model_type="mano",flat_hand_mean=True,batch_size=batch_size,use_pca=True)
	z = torch.normal(0,1.5,[batch_size,6], requires_grad=True)
	theta_gt = torch.einsum('bi,ij->bj', [z, mano.hand_components])
	result = mano(hand_pose=z, global_orient=torch.zeros([batch_size,3]), transl=torch.zeros([batch_size,3]))
	theta = j2m.computeTheta(result.joints)
	mano2 = smplx.create(model_path="./models/MANO_RIGHT.pkl", model_type="mano",flat_hand_mean=True,batch_size=batch_size,use_pca=False)
	res2 = mano2(hand_pose=theta)

	print('FINAL_GRAD')
	gradient = torch.autograd.grad(theta.sum(), z, retain_graph=True)

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	ax = plt.subplot(111, projection='3d')
	ax.scatter(result.vertices.detach().cpu().numpy()[0,:,0], result.vertices.detach().cpu().numpy()[0,:,1], result.vertices.detach().cpu().numpy()[0,:,2], s=3, c='b')
	ax.scatter(res2.vertices.detach().cpu().numpy()[0,:,0], res2.vertices.detach().cpu().numpy()[0,:,1], res2.vertices.detach().cpu().numpy()[0,:,2], s=3, c='r')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	plt.show()
