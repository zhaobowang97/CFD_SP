import torch
import numpy as np
import os

from sympy import *

from mesh_io.template_mesh import Template_Mesh
from mesh_io.geometry import Edge_Set

point_number = 13289
face_number = 13056

# hard code of lines to get boundary, contour
# todo: get the first layer
face_start_line = 2
face_end_line = 13058
point_start_line = 13059
point_end_line = 26348
boundary_start_line = 26351
boundary_end_line = 26487


def read_points(filename):
    lines = open(filename, 'r').readlines()
    points = np.zeros([point_number, 2], dtype=float)
    for i, line in enumerate(lines[point_start_line : point_end_line]):
        line = line.strip().split('\t')
        points[i,0] = float(line[0])
        points[i,1] = float(line[1])
    return points
    
def read_faces(filename):
    lines = open(filename, 'r').readlines()
    faces = np.zeros([face_number, 4], dtype=int)
    for i, line in enumerate(lines[face_start_line : face_end_line]):
        line = line.strip().split('\t')
        faces[i,0] = int(line[1])
        faces[i,1] = int(line[2])
        faces[i,2] = int(line[3])
        faces[i,3] = int(line[4])
    return faces

def read_contour_id_list(filename):
    contour_id_list = [i for i in range(137)]
    return np.array(contour_id_list, dtype=int)

def read_boundary_id_list(filename):
    lines = open(filename, 'r').readlines()
    boundary_id_list = []
    for line in lines[boundary_start_line : boundary_end_line]:
        line = line.strip().split('\t')
        boundary_id_list.append([int(line[1]), int(line[2])])
    return np.array(boundary_id_list, dtype=int)

points = read_points("mesh_flatplate_turb_137x97.su2")
contour_list =  read_contour_id_list("mesh_flatplate_turb_137x97.su2")
contour = points[contour_list,:]
a = np.array([1,50], dtype=int)
points_new = points * a

raw_file = "mesh_flatplate_turb_137x97.su2"
output_point_file = "test_mesh_scale_50_same_upper.su2"
lines = open(raw_file, 'r').readlines()

# fix the contour
points_new[0: 137, :] = contour
y_origin = points[137,1]
y_new =  points_new[137,1]
u_origin = points[-1, 1]
u_new = points_new[-1,1]

Sn = u_origin - points[0,1]
a1 = y_new - points[0,1]
q = 1.07
"""for i in range(1000):
    s = a1 * (1 - q ** 96) / (1 - q)
    print(q,s)
    if s > Sn:
        break
    q += 0.01"""


points_new[274:,1] = (points[274:,1] - y_origin) / (u_origin - y_origin) * (u_origin - y_new) + y_new

tmp = 0
for i, j in enumerate(np.arange(point_start_line, point_end_line)):
    t = i // 137
    if t == 0:
        tmp = 0
    elif (i % 137) == 0:
        tmp += a1 * (q ** (t -1))
        print(t)
    
    if t != 96:
        lines[j] = '\t{}\t{}\t{}\n'.format(points_new[i,0], points[0,1] + tmp, i)
    else:
        lines[j] = '\t{}\t{}\t{}\n'.format(points_new[i,0], u_origin, i)
    



            
f = open(output_point_file, 'w')
f.write(''.join(lines))
f.close()

# save
    

class Naca0012_Invicid_Trimesh_2d(Template_Mesh):
    def __init__(self, cache_name=None, su2Mesh_dir=None):
        super().__init__()
        self.cache_name = cache_name
        self.su2Mesh_dir = su2Mesh_dir
        
    # init CFD mesh, make or use cache file
    # the structure of cache file:
    # dict
    #   deformed_area
    #     points [num_deformed_pt, 2]
    #     edge_set
    #     faces [num_deformed_faces, 3]
    #     x/y_movable_mask [num_deformed_pt]
    #     airfoil_contour_id_list [num_pt_af, 2]
    #     boundary_contour_id_list [num_pt_b, 2]
    def init_mesh(self, use_cache = True):
        if self.cache_name is None:
            self.cache_name = './mesh_io/naca0012_su2/init_mesh_cache.pymesh'
            
        # if use cache, load cache file directly
        if (use_cache == True) and (os.path.isfile(self.cache_name) == True):
            self.CFD_mesh = torch.load(self.cache_name)
            return self.CFD_mesh
        
        # else not use cache, then init a cache file
        if self.su2Mesh_dir is None:
            raw_file = './mesh_io/naca0012_su2/mesh_NACA0012_inv.su2'
        else:
            raw_file = os.path.join(self.su2Mesh_dir, 'mesh_NACA0012_inv.su2')
        
        points = read_points(raw_file)
        faces = read_faces(raw_file)
        edge_set = Edge_Set()
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                edge_set.add_edge(faces[i,j], faces[i,j-1])
        contour_id_list = read_contour_id_list(raw_file)
        boundary_id_list = read_boundary_id_list(raw_file)
        
        movable_mask = np.ones(points.shape).astype(bool)
        movable_mask[np.unique(boundary_id_list.flatten()), :] = 0
        
        self.CFD_mesh = dict()
        self.CFD_mesh['deformed_area'] = dict()
        self.CFD_mesh['deformed_area']['points'] = points
        self.CFD_mesh['deformed_area']['faces'] = faces
        self.CFD_mesh['deformed_area']['edge_set'] = edge_set
        self.CFD_mesh['deformed_area']['airfoil_contour_id_list'] = contour_id_list
        self.CFD_mesh['deformed_area']['boundary_contour_id_list'] = boundary_id_list
        self.CFD_mesh['deformed_area']['movable_mask'] = movable_mask
        
        # save and return cache file
        torch.save(self.CFD_mesh, self.cache_name)
        return self.CFD_mesh
        
    # read raw .su2 file and replace the points coordinates with new ones
    # points, [num_pt, 2]
    def write_to_su2Mesh(self, points, output_point_file):
        assert(points.ndim == 2)
        assert(points.shape[0] == point_number)
        
        if self.su2Mesh_dir is None:
            raw_file = './mesh_io/naca0012_su2/mesh_NACA0012_inv.su2'
        else:
            raw_file = os.path.join(self.su2Mesh_dir, 'mesh_NACA0012_inv.su2')
        
        lines = open(raw_file, 'r').readlines()
        for i, j in enumerate(np.arange(point_start_line, point_end_line)):
            lines[j] = '\t{}\t{}\t{}\n'.format(points[i,0], points[i,1], i)
            
        f = open(output_point_file, 'w')
        f.write(''.join(lines))
        f.close()