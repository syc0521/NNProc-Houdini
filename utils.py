import os
import sys
import math
from tqdm import tqdm
import trimesh
from trimesh import transformations
import numpy as np
import pyvista as pv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ShapeInfo
sys.path.insert(0, os.path.join('/', 'mnt', 'Research', 'Codebase', 'DatasetMaker'))
from proc_shape.procedure import Procedure
import open3d as o3d
import torch

pl = pv.Plotter(off_screen=True, window_size=(256, 256))

def orient_mesh(mesh: trimesh.Trimesh):
    rot_matrix = transformations.rotation_matrix(angle=(math.pi / 2), direction=[1, 0, 0], point=[0, 0, 0])
    t = np.sum(mesh.bounding_box.bounds, axis=0) / 2
    mesh.apply_translation(-t)
    s = np.max(mesh.extents)
    mesh.apply_scale(1 / s)
    mesh.apply_transform(rot_matrix)
    return mesh

def get_proc_meshes(shape, params):
    proc = Procedure(shape)
    vectors = proc.paramvecdef.decode(params)
    meshes = []
    with tqdm(total=len(vectors), file=sys.stdout, desc='Generating shapes') as progress:
        for vector in vectors:
            meshes.append(proc.get_shape(vector))
            progress.update()
    return meshes

def render(tmesh, orient=True):
    if orient:
        tmesh = orient_mesh(tmesh)
    v, f = tmesh.vertices, np.hstack([np.ones((tmesh.faces.shape[0], 1), dtype=int) * 3, tmesh.faces])
    pvmesh = pv.PolyData(v, f)
    pvmesh = pvmesh.rotate_z(90)
    actor = pl.add_mesh(pvmesh)
    pl.reset_camera()
    img = pl.show(return_img=True, auto_close=False)
    pl.remove_actor(actor)
    return img

def evaluate_scalar(y, y_pred):
    diff = np.abs(y_pred - y)
    diff = np.mean(diff, axis=0)
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print(diff, end='')
    print()

def evaluate_discrete(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=1, average=None)
    recall = recall_score(y, y_pred, zero_division=1, average=None)
    f1 = f1_score(y, y_pred, zero_division=1, average=None)
    scores = np.array([accuracy, precision.mean(), recall.mean(), f1.mean()])
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(scores, end='')
    print()

def print_report(shape, predictions, targets):
    shape_info = ShapeInfo(shape=shape)
    for i, param in enumerate(shape_info.params):
        y = targets[i]
        y_pred = predictions[i]
        if param.paramtype == 'scalar':
            evaluate_scalar(y, y_pred)
        elif param.paramtype == 'integer' or param.paramtype == 'type':
            evaluate_discrete(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        else:
            evaluate_discrete(y, np.round(y_pred))

def voxelize_mesh(mesh: trimesh.Trimesh, visualize=False):
    rsl = 64
    omesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh(omesh, voxel_size=(1.0 / (rsl - 1)))
    if visualize:
        o3d.visualization.draw_geometries([voxels])
    voxels = voxels.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    indices = indices + (np.array([rsl-1, rsl-1, rsl-1]) - np.max(indices, axis=0)) // 2
    voxel_arr = np.zeros((rsl, rsl, rsl), dtype=np.uint8)
    voxel_arr[tuple(indices.T)] = 1
    return torch.tensor(voxel_arr, dtype=torch.float32)