import os

import numpy as np
import scipy.io as sio
import tensorflow as tf
#tf.disable_v2_behavior() 
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from tf_hand_kinematics import kinematics



class HandModel:
    def __init__(self, batch_size):
        # Hand shape
        self.parts = ['palm',
                    'thumb0', 'thumb1', 'thumb2', 'thumb3',
                    'index0', 'index1', 'index2', 'index3',
                    'middle0', 'middle1', 'middle2', 'middle3',
                    'ring0', 'ring1', 'ring2', 'ring3',
                    'pinky0', 'pinky1', 'pinky2', 'pinky3']

        # self.surface_pts = {p: tf.constant(tm.load(os.path.join(os.path.dirname(__file__), '../data', 'hand', p + '.STL')).vertices, dtype=tf.float32) for p in self.parts}
        # self.surface_pts = {p: tf.constant(np.mean(np.load(os.path.join(os.path.dirname(__file__), '../data', p + '.faces.npy')), axis=1), dtype=tf.float32) for p in self.parts if '0' not in p}
        self.surface_pts = {p: tf.constant(np.load(os.path.join(os.path.dirname(__file__), './data', p + '.sample_points.npy')), dtype=tf.float32) for p in self.parts if '0' not in p}
        self.pts_normals = {p: tf.constant(np.load(os.path.join(os.path.dirname(__file__), './data', p + '.sample_normal.npy')), dtype=tf.float32) for p in self.parts if '0' not in p}
        self.pts_feature = tf.tile(tf.expand_dims(tf.concat([tf.constant(np.load(os.path.join(os.path.dirname(__file__), './data', p + '.sample_feat.npy')), dtype=tf.float32) for p in self.parts if '0' not in p], axis=0), axis=0), [batch_size, 1, 1])
        self.n_surf_pts = sum(x.shape[0] for x in self.surface_pts.values())

        # Input placeholder
        self.gpos = tf.placeholder(tf.float32, [batch_size, 3])
        self.grot = tf.placeholder(tf.float32, [batch_size, 3, 2])
        self.jrot = tf.placeholder(tf.float32, [batch_size, 22])

        # Build model
        self.tf_forward_kinematics(self.gpos, self.grot, self.jrot)

    def tf_forward_kinematics(self, gpos, grot, jrot):
        xpos, xquat = kinematics(gpos, grot, jrot)
        out_surface_key_pts = {n:tf.transpose(tf.matmul(xquat[n], tf.transpose(tf.pad(tf.tile(tf.expand_dims(self.surface_pts[n], axis=0), [gpos.shape[0], 1, 1]), paddings=[[0,0],[0,0],[0,1]], constant_values=1), perm=[0,2,1])), perm=[0,2,1])[...,:3] + tf.expand_dims(xpos[n], axis=1) for n in self.surface_pts}
        out_surface_normals = {n:tf.transpose(tf.matmul(xquat[n], tf.transpose(tf.pad(tf.tile(tf.expand_dims(self.pts_normals[n], axis=0), [gpos.shape[0], 1, 1]), paddings=[[0,0],[0,0],[0,1]], constant_values=1), perm=[0,2,1])), perm=[0,2,1])[...,:3] for n in self.pts_normals}

        all_hand_points = tf.zeros([0,3], dtype = tf.float32)

        for ky in out_surface_key_pts.keys():
            all_hand_points = tf.concat([all_hand_points, out_surface_key_pts[ky][0]], 0)

        return all_hand_points, out_surface_normals


import os
import numpy as np
import tensorflow as tf
#tf.disable_v2_behavior() 

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def compute_distance(hand_points, point):
    #hand_points: 5871*3
    #point: 1*3

    # repeat point 5871 times and find norm of difference between that point and other points.
    points = np.tile(point, [len(hand_points), 1])
    return np.linalg.norm(hand_points-points, axis = 1)

def generate_graph(thrhd_dist): #thrhd_dist is a temperary number
    parts = ['palmpts.npy',
                     'thumb1pts.npy', 'thumb2pts.npy', 'thumb3pts.npy',
                     'index1pts.npy', 'index2pts.npy', 'index3pts.npy',
                     'middle1pts.npy', 'middle2pts.npy', 'middle3pts.npy',
                    'ring1pts.npy', 'ring2pts.npy', 'ring3pts.npy',
                     'pinky1pts.npy', 'pinky2pts.npy', 'pinky3pts.npy']
    
    parts_len = []
    hd_pts = np.empty([0,3])
    fortest_each_part = []

    # 1. load all points
    for i in parts: # load all pts and stack them to a big array
        #print(i)
        temp = np.load(i)
        parts_len.append(len(temp))
        fortest_each_part.append(temp)
        hd_pts = np.vstack((hd_pts, temp))
    hd_pts = hd_pts.astype(np.float32)

    # 2. compute distance and get threshhold and initialte needed variables
    distance = compute_distance(hd_pts, hd_pts[1])
    sorted_distance = np.sort(distance) #we get a threashold at around 0.005 as distance to have an edge
    
    edge_list = []
    start_pts = np.load("start_pts.npy") # this is where each point is valid to find edge in
    new_should_connect = []
    new_should_connect_from_palm = []

    # 3. for all the nodes, do the following
    for i in range(len(hd_pts)):
        # 3.1. first compute distance from point i and dist < threshhold are the points should be connected
        distance = compute_distance(hd_pts, hd_pts[i])
        should_connect = np.where(distance < thrhd_dist)[0]

        #  3.2. consider only points in the range for edge, one range is what they have in start_pts, one is the palm range
        for j in range(len(start_pts)):
            if (i >= start_pts[j][0] and i < start_pts[j][1]):
                new_should_connect = should_connect[should_connect >= start_pts[j][0]]
                new_should_connect = new_should_connect[new_should_connect < start_pts[j][1]]
                new_should_connect_from_palm = should_connect[should_connect >= start_pts[0][0]]
                new_should_connect_from_palm = new_should_connect_from_palm[new_should_connect_from_palm < start_pts[0][1]]
        
        # 3.3. append all the edges to the edge lists
        if len(new_should_connect) == 0:
            print(sorted_distance[0])
            print(np.where(distance == sorted_distance[0])[0])
        new_should_connect = should_connect
        for k in new_should_connect:
            edge_list.append((i, int(k), float(distance[k])))
        for k in new_should_connect_from_palm:
            edge_list.append((i, int(k), float(distance[k])))
    
    # 4. graph it
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)

    # 5. get adj_mtx
    A =  nx.to_numpy_matrix(G)

    return A


def manifold_dist_btw_two_nodes(adj_mtx, a):
    
    graph = csr_matrix(adj_mtx)
    dist_matrix = dijkstra(csgraph=graph, directed=False, indices=a, return_predecessors=False)
    print(dist_matrix)
    mnfld_dist = np.linalg.norm(dist_matrix, axis = 0)
    print(mnfld_dist)
    print("manifold_dist_test")
    return mnfld_dist

#def points_sampling(all_pts):

def select_pts(all_pts, pts_distfrom_obj, manifold_dist):
    #!!!!!!!!!! energy too small

    #check pts_distfrom_obj is inversed or not

    #all_pts is tensor
    #pts_distfrom_obj is tensor ?
    #adj_mtx is np_array

    manifold_constant = 10

    # 1. compute points contacted to objects
    #first_point_index_choices = np.where(pts_distfrom_obj == pts_distfrom_obj.min())[0]
    #first_point_index = np.random.choice(first_point_index_choices)
    first_point_index_choices = tf.reshape(tf.where(tf.equal(pts_distfrom_obj, tf.reduce_min(pts_distfrom_obj))), [-1])#? if not touching, we may need to add some more points
    first_point_index = tf.random.shuffle(first_point_index_choices)[0]#assume at least one point on the obj
    

    # 2. compute the first point and compute loss
    first_point = all_pts[first_point_index]
    first_point_manifold_dist = -manifold_dist[first_point_index] #manifold_dist_btw_two_nodes(adj_mtx, first_point_index) # manifold_dist small should cause more loss
    loss_from_firstpoint = pts_distfrom_obj + first_point_manifold_dist*manifold_constant # we first don't have any weight for these two loss functions

    # 3. compute threshhold for second point and find the second point
    loss_thrhd_scdpoint = tf.sort(loss_from_firstpoint)[100] # 100 is a temperary number
    second_point_index_choices = tf.reshape(tf.where(tf.less(loss_from_firstpoint, loss_thrhd_scdpoint)), [-1])
    second_point_index = tf.random.shuffle(second_point_index_choices)[0]
    #second_point_index_choices = np.where(loss_from_firstpoint <= loss_thrhd_scdpoint)[0][1:]
    #second_point_index = np.random.choice(second_point_index_choices)
    #while(second_point_index == first_point_index): # the smallest shoudl the first_point itself, but just in case we use a while loop to check
    #    second_point_index = np.random.choice(second_point_index_choices)
    second_point = all_pts[second_point_index]

    # 4. find loss for the third point
    second_point_manifold_dist = -manifold_dist[second_point_index]# -manifold_dist_btw_two_nodes(adj_mtx, second_point_index)
    loss_from_firstpoint_and_secondpoint = loss_from_firstpoint + second_point_manifold_dist*manifold_constant # manifold dist should be less important as it should be on objects first

    # 5. compute threshhold for the third point and find the third point
    loss_thrhd_thirdpoint = tf.sort(loss_from_firstpoint_and_secondpoint)[100] # 100 is a temperary number
    third_point_index_choices = tf.reshape(tf.where(tf.less(loss_from_firstpoint_and_secondpoint, loss_thrhd_thirdpoint)), [-1])
    third_point_index = tf.random.shuffle(third_point_index_choices)[0]
    #third_point_index_choices = np.where(loss_from_firstpoint_and_secondpoint <= loss_thrhd_thirdpoint)[0][1:]
    #third_point_index = np.random.choice(third_point_index_choices) 
    #while((third_point_index == first_point_index) or (third_point_index == second_point_index)):# the smallest shoudl the first_point itself, but just in case we use a while loop to check 
    #    third_point_index = np.random.choice(third_point_index_choices) 
    third_point = all_pts[third_point_index]

    # 6. return all the threes points with their indices, with form (index, point)
    return first_point_index, second_point_index, third_point_index
    #[(first_point_index, first_point), (second_point_index, second_point), (third_point_index, third_point)]

def distance_from_sphere(sphere_center, pts, sphere_radius):
    temp_allpts = tf.identity(pts)
    xs = temp_allpts[:,0] - sphere_center[0]
    ys = temp_allpts[:,1] - sphere_center[1]
    zs = temp_allpts[:,2] - sphere_center[2]
    nmlzd_pts = tf.transpose(tf.stack([xs,ys,zs]))
    nmlzd_pts = tf.cast(nmlzd_pts, dtype = tf.float32)
    dist = tf.abs(tf.norm(nmlzd_pts, axis = 1) - sphere_radius)
    return dist

class ForceClosure:
    def __init__(self, hm, eps, fric, mu, manifold_dist, center_of_obj, batch_size):
        # const initilization
        self.hdmodel = hm
        #self.objmodel = om
        self.epsilon = eps
        self.friction = tf.constant([[0,0,1],[0,0,1],[0,0,1]], dtype = tf.float32) #tf.placeholder(tf.float32, [3, 3]) # change 3,3 to number,3
        self.mu = mu
        self.norm = tf.placeholder(tf.float32, [3, 3])
        self.sphere_center = tf.placeholder(tf.float32, [3])
        self.sphere_radius = 0.07
        self.manifold_dist = manifold_dist
        self.center_of_obj = center_of_obj
        self.all_pts = tf.placeholder(tf.float32, [None, 3])
        self.gpos = tf.placeholder(tf.float32, [batch_size, 3])
        self.grot = tf.placeholder(tf.float32, [batch_size, 3, 2])
        self.jrot = tf.placeholder(tf.float32, [batch_size, 22])
        self.a = tf.constant(0)
        self.b = tf.constant(0)
        self.c = tf.constant(0)
        self.select_point_times = 20
        self.fric_opt_times = 350

        #self.loss_wa = 1
        #self.loss_wb = 1
        #self.loss_wc = 1

        # get loss
        self.epoch = -1
        self.x = tf.placeholder(tf.float32, [3, 3])
        self.total_loss(self.x, self.epsilon, self.friction, self.norm, self.mu, self.sphere_center, self.sphere_radius)
        self.friction_energy(self.x, self.friction, self.norm, self.mu)
        self.norm_on_sphere_tf(self.sphere_center, self.x)
        self.training()
        self.for_tst()

    def for_tst(self):
        print(self.norm)
        print(self.x)
        
    
    def x_to_G(self, x): # x: n * 3, contact points
        #G : 6*3n
        tf_x = tf.convert_to_tensor(x)
        tf_T = tf.constant([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]], dtype = tf.float32)
        xi_cross = tf.reshape(tf.transpose(tf.reshape(tf.matmul(tf_x,tf_T), [-1,3,3]), [1,0,2]),[3,-1])

        # 1. get I 
        I = tf.reshape(tf.transpose(tf.eye(3, batch_shape = [int(tf_x.shape[0])], dtype = tf.float32), perm = [1,0,2]), [3, 3*int(tf_x.shape[0])])
        # 2. resulted G
        result = tf.reshape(tf.stack([I, xi_cross]), [6, int(I.shape[1])])
        # result is 6*n and xi_cross is 3*n
        return result, xi_cross

    def loss_8a(self, G, epsilon):
        Gt = tf.transpose(G)    

        temp = tf.scalar_mul(epsilon, tf.eye(6))
        temp = tf.cast(temp, tf.float32)
        result_mtx = tf.subtract(tf.matmul(G, Gt), temp)

        eigval = tf.self_adjoint_eigvals(result_mtx)
        # we first subtract zero by resulted eigval and use relu to eliminate all positive eigenvalues of orginal result
        result = tf.nn.relu(tf.subtract(tf.constant(0, tf.float32, shape = eigval.get_shape().as_list()), eigval))
        result = tf.reduce_sum(tf.square(result)) # check >=
        return result

    def loss_8b(self, f, G): #tested
        #_, G = x_to_G(x)
        new_f = tf.identity(f)
        new_f = tf.reshape(new_f, [f.shape[0]*3,1])
        result_mtx = tf.matmul(G, new_f)
        return tf.nn.l2_loss(result_mtx)

    def loss_8c(self, norm, friction, x, mu):
        #all inputs are n*3, except mu, which is the friction coeff.
        # z is then n*3
        # this 
        ci = tf.math.add(x, norm)
        lefthand_result = tf.einsum('ij, ij->i', friction, ci) #fi' * ci
    
        righthand_result = tf.norm(friction, axis = 1)/tf.cast((tf.sqrt(mu**2+1)), tf.float32)
        result = (lefthand_result - righthand_result)
        return tf.reduce_sum(tf.nn.relu(0 - result[1:]))
    
    def dist_loss(self, sphere_center, pts, sphere_radius):
        
        nmlzd_pts = tf.stack([(pts[:,0] - sphere_center[0]),\
                            (pts[:,1] - sphere_center[1]),\
                            (pts[:,2] - sphere_center[2])])
        dist = tf.norm(nmlzd_pts, axis = 1) - sphere_radius

        return tf.reduce_sum(tf.abs(dist))

    def total_loss(self, x, epsilon, friction, norm, mu, sphere_center, sphere_radius):
        Ga, Gb = self.x_to_G(x)
        return (self.loss_8a(Ga, epsilon) + \
                            self.loss_8b(friction, Ga) + \
                            self.loss_8c(norm, friction, x, mu) + \
                            self.dist_loss(sphere_center, x, sphere_radius))

    def friction_energy(self, x, friction, norm, mu):
        Ga, Gb = self.x_to_G(x)
        return (self.loss_8b(friction, Ga) + \
                            self.loss_8c(norm, friction, x, mu))
    
    def norm_on_sphere_tf(self, sphere_center, pts):
        #
        nmlzd_pts = tf.stack([(pts[:,0] - sphere_center[0]),\
                                (pts[:,1] - sphere_center[1]),\
                                (pts[:,2] - sphere_center[2])])

        norm = tf.reshape(tf.norm(nmlzd_pts, axis = 1),[-1,1])
        norm = tf.tile(norm, [1,3])

        return (nmlzd_pts/norm)
    
    def stack_three_pts(self, a, b, c):
        return tf.stack([a, b, c])

    def training(self):
        step_size = 0.05

        #all_pts = tf_all_pts.eval(session=sess)
        self.all_pts, _ = self.hdmodel.tf_forward_kinematics(self.gpos, self.grot, self.jrot)

        pts_distfrom_obj = distance_from_sphere(self.center_of_obj, self.all_pts, self.sphere_radius)
        print(pts_distfrom_obj.shape)
        energy = tf.constant(np.inf)

        training_friction = tf.identity(self.friction)
        init_j = tf.constant(0)
        
        def condition_select_pts(j, *args):
            print("reach condition_select_pts")
            return j < self.select_point_times

        def body_select_pts(j, energy, pts_distfrom_obj, training_friction, aa, bb, cc):
            print("reach body_select_pts")
            a, b, c = select_pts(self.all_pts, pts_distfrom_obj, self.manifold_dist)
            the_x = tf.stack([self.all_pts[a], self.all_pts[b], self.all_pts[c]])
            the_norm = self.norm_on_sphere_tf(self.sphere_center, the_x)
            fric_energy = tf.constant(np.inf)#tf.constant(np.inf)
            the_friction = tf.identity(training_friction) #tf.zeros([1])
            init_k = tf.constant(0)

            def condition_friction_optimization(k, *args):
                print("reach condition_friction_optimization")
                return k < self.fric_opt_times

            def body_friction_optimization(k, training_friction, fric_energy, the_friction, the_x, the_norm):
                print("reach body_friction_optimization")
                temp_fric_energy = self.friction_energy(the_x, training_friction, the_norm, self.mu) 
                #if (temp_fric_energy < fric_energy):#temp_fric_energy.eval(session=sess) < fric_energy.eval(session=sess):
                the_friction = tf.cond((temp_fric_energy >= fric_energy), lambda: the_friction, lambda: training_friction)
                fric_energy = tf.cond((temp_fric_energy >= fric_energy), lambda: fric_energy, lambda: temp_fric_energy)
                grad_z = tf.gradients(temp_fric_energy, training_friction)[0]
                if k == 0:
                    print(grad_z)
                grad_z = tf.clip_by_norm(grad_z, 1)
                training_friction = training_friction - 0.1 * grad_z

                return tf.add(k, 1), training_friction, fric_energy, the_friction, the_x, the_norm

            init_k, training_friction, fric_energy, the_friction, the_x, the_norm = \
                tf.while_loop(condition_friction_optimization, body_friction_optimization, [init_k, training_friction, fric_energy, the_friction, the_x, the_norm])

            energy_temp = self.total_loss(the_x, self.epsilon, the_friction, the_norm, self.mu, self.sphere_center, self.sphere_radius)
            a = tf.cast(a, tf.int32)
            b = tf.cast(b, tf.int32)
            c = tf.cast(c, tf.int32)

            #self.a = tf.cond((energy_temp < energy), lambda: a, lambda: self.a) #tf.cast(a, tf.int32)
            #self.b = tf.cond((energy_temp < energy), lambda: b, lambda: self.b) #tf.cast(b, tf.int32)
            #self.c = tf.cond((energy_temp < energy), lambda: c, lambda: self.c) #tf.cast(c, tf.int32)

            (energy, self.a, self.b, self.c) = tf.cond((energy_temp < energy), lambda: (energy_temp, a, b, c), lambda: (energy, self.a, self.b, self.c ))
            return tf.add(j, 1), energy, pts_distfrom_obj, training_friction, self.a, self.b, self.c

        init_j, energy, pts_distfrom_obj, training_friction, self.a, self.b, self.c = \
                                    tf.while_loop(condition_select_pts, body_select_pts, [init_j, energy, pts_distfrom_obj, training_friction, self.a, self.b, self.c])
            


        # 3. update z according to energy gradient
        grad_z = tf.gradients(energy, [self.jrot,self.grot,self.gpos]) 
        grad_z_jrot = tf.clip_by_norm(grad_z[0], 1)
        grad_z_grot = tf.clip_by_norm(grad_z[1], 1)
        grad_z_gpos = tf.clip_by_norm(grad_z[2], 1)
        output_jrot = self.jrot - step_size * grad_z_jrot
        output_grot = self.grot - step_size * grad_z_grot
        output_gpos = self.gpos - step_size * grad_z_gpos

        print("done with epoch" + str(self.epoch))
        self.epoch += 1
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return output_jrot, output_grot, output_gpos, self.all_pts, grad_z_jrot, grad_z_grot, grad_z_gpos, self.a, self.b, self.c, energy

if __name__ == "__main__":
    #TODO_question:
    #   test case what's normal if not touching // and if no touching, we cannot find the first point
    #   DONE -- where should the ball be
    #   Remember to plot every update of z

    center_of_obj = [-0.139, -0.245, 0.22 ]#[-0.15, -0.23, 0.38 ]
    radius = 0.07


    epoch = 100
    step_size = 0
    select_point_times = 10
    fric = [[0,0,1],[0,0,1],[0,0,1]]#tf.constant([[0,0,1],[0,0,1],[0,0,1]], dtype = tf.float32)
    eps = 0.01
    mu = 0.3

    adj_mtx = generate_graph(thrhd_dist = 0.0055)
    
    glove_data = sio.loadmat('./data/grasp/cup%d/cup%d_grasping_60s_1.mat'%(1, 1))['glove_data'] # is this z?

    i = 0
    gpos = glove_data[:,4:7]
    grot = np.array([Q(glove_data[i,1 + 28 * 3 + 4 : 1 + 28 * 3 + 8]).rotation_matrix[:,:2] for i in range(len(glove_data))])
    jrot = np.zeros(glove_data[:,1 + 28 * 3 + 28 * 4 + 7 : 1 + 28 * 3 + 28 * 4 + 29].shape)
    #jrot = np.cos(jrot)
    hm = HandModel(glove_data.shape[0])

    gpos = np.ndarray.astype(gpos, np.float32)
    grot = np.ndarray.astype(grot, np.float32)
    jrot = np.ndarray.astype(jrot, np.float32)
    
    
    #tf operations
    out_key_pts, out_normals = hm.tf_forward_kinematics(tf.constant(gpos), tf.constant(grot), tf.constant(jrot))

    manifold_dist = []
    manifold_dist_path = "./manifold_distance_new.npy"
    sess = tf.Session()

    if os.path.exists(manifold_dist_path):
        manifold_dist = np.load(manifold_dist_path)
        #for i in range(temp.shape[0]):
        #    manifold_dist.append(np.linalg.norm(temp[i], axis = 0))
        #    print(manifold_dist[i].shape)
        #print("done_normalizing")
        #np.save(manifold_dist_path, manifold_dist)
        #manifold_dist = np.asarray(manifold_dist)
    else:
        count = 0
        for i in out_key_pts.eval(session=sess):
            #print(i)
            if(count%60 == 0):
                print(count)
            count += 1
            
            manifold_dist.append(manifold_dist_btw_two_nodes(adj_mtx, i))
        manifold_dist = np.asarray(manifold_dist)
        np.save(manifold_dist_path, manifold_dist)

    manifold_dist = np.ndarray.astype(manifold_dist, dtype = np.float32)
    manifold_dist = tf.constant(manifold_dist)

    fc = ForceClosure(hm, eps, fric, mu, manifold_dist, tf.constant(center_of_obj), glove_data.shape[0])
    training_for_fc = fc.training()

    #start tf session
    

    for i in range(epoch):
        print("one more epoch")
        fric = np.asarray(fric, dtype = np.float32)
        print(fric.shape)
        jrot, grot, gpos, the_all_pts, jrot_grad, grot_grad, gpos_grad, a, b, c, energy = sess.run(training_for_fc, feed_dict = {fc.sphere_center: center_of_obj, fc.jrot: jrot, fc.grot: grot, fc.gpos: gpos})
        all_pts = the_all_pts

        print("Where are the three points")
        print(a)
        print(b)
        print(c)
        print(energy)

        plt.ion()
        ax = plt.subplot(111, projection='3d')

        #from here to plot the sphere
        center_of_obj = [-0.139, -0.245, 0.22 ]
        r = 0.06
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x_sphere = r*np.sin(phi)*np.cos(theta) + center_of_obj[0]
        y_sphere = r*np.sin(phi)*np.sin(theta) + center_of_obj[1]
        z_sphere = r*np.cos(phi) + center_of_obj[2]

        fr = 0
        ax.cla()
        xmin, xmax, ymin, ymax, zmin, zmax = None, None, None, None, None, None
        
        ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], s=1)
        _xmin, _xmax, _ymin, _ymax, _zmin, _zmax = min(all_pts[:,0]), max(all_pts[:,0]), min(all_pts[:,1]), max(all_pts[:,1]), min(all_pts[:,2]), max(all_pts[:,2])
        if xmin is None or _xmin < xmin:
            xmin = _xmin
        if xmax is None or _xmax > xmax:
            xmax = _xmax
        if ymin is None or _ymin < ymin:
            ymin = _ymin
        if ymax is None or _ymax > ymax:
            ymax = _ymax
        if zmin is None or _zmin < zmin:
            zmin = _zmin
        if zmax is None or _zmax > zmax:
            zmax = _zmax

        ax.plot_surface(
            x_sphere, y_sphere, z_sphere,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
        x = (xmin + xmax)/2
        y = (ymin + ymax)/2
        z = (zmin + zmax)/2
        ax.set_xlim([x-0.15, x+0.15])
        ax.set_ylim([y-0.15, y+0.15])
        ax.set_zlim([z-0.15, z+0.15])
        plt.pause(1e-5)
        #plt.savefig('./result/result '+ str(i) + '.png')
        print(i)