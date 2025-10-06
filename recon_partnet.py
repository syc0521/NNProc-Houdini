import os
import sys
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import trimesh
import torch
import torch.nn as nn
import open3d as o3d
from model import NNProc
from utils import get_proc_meshes, render
sys.path.insert(0, os.path.join('/', 'mnt', 'Research', 'Codebase', 'DatasetMaker'))
from proc_shape.procedure import Procedure
from applications import print_chamfer_distance


def voxelize_mesh(mesh: trimesh.Trimesh):
    rsl = 64
    omesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh(omesh, voxel_size=(1.0 / (rsl - 1)))
    voxels = voxels.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    indices = indices + (np.array([rsl-1, rsl-1, rsl-1]) - np.max(indices, axis=0)) // 2
    voxel_arr = np.zeros((rsl, rsl, rsl), dtype=np.uint8)
    voxel_arr[tuple(indices.T)] = 1
    return voxel_arr


if __name__ == '__main__':
    shape = 'table'
    shapeassembly_dir = os.path.join('/', 'home', 'ishtiaque', 'Documents', 'ShapeAssembly', 'code', 'recon', shape)
    partnet_dir = os.path.join('/', 'mnt', 'Volume4', 'data_v0',)
    shapeassembly_inds = [int(x[:-4]) for x in os.listdir(shapeassembly_dir)]
    partnet_inds = [int(x) for x in os.listdir(partnet_dir)]
    inds = [str(x) for x in sorted(list(set(shapeassembly_inds).intersection(set(partnet_inds))))]
    inds = inds[:1500]
    gt_meshes, sa_meshes, voxels  = [], [], []
    with tqdm(total=len(inds), file=sys.stdout, desc='Reading data') as pbar:
        for ind in inds:
            gt_mesh = o3d.geometry.TriangleMesh()
            for obj_file in os.listdir(os.path.join(partnet_dir, ind, 'objs')):
                temp_mesh = o3d.io.read_triangle_mesh(os.path.join(partnet_dir, ind, 'objs', obj_file))
                gt_mesh += temp_mesh
            gt_mesh = trimesh.Trimesh(vertices=[list(x) for x in gt_mesh.vertices], faces=[list(x) for x in gt_mesh.triangles])
            t = np.sum(gt_mesh.bounding_box.bounds, axis=0) / 2
            gt_mesh.apply_translation(-t)
            s = np.max(gt_mesh.extents)
            gt_mesh.apply_scale(1 / s)

            sa_mesh = trimesh.load(os.path.join(shapeassembly_dir, '{}.obj'.format(ind)), force='mesh')
            t = np.sum(sa_mesh.bounding_box.bounds, axis=0) / 2
            sa_mesh.apply_translation(-t)
            s = np.max(sa_mesh.extents)
            sa_mesh.apply_scale(1 / s)

            gt_meshes.append(gt_mesh)
            voxels.append(voxelize_mesh(gt_mesh))
            sa_meshes.append(sa_mesh)
            pbar.update()
    voxels = torch.stack([torch.unsqueeze(torch.tensor(x, dtype=torch.float32), dim=0) for x in voxels])
    model = NNProc(shape)
    model.load_state_dict(torch.load(os.path.join('models', '{}_model.pt'.format(shape))))
    model.eval()
    batch_size = 32
    params = []
    i = 0
    start_time = time.time()
    while i < len(inds):
        params.append(model.param_dec.predict(model.voxel_enc.predict(voxels[i:i+batch_size])))
        i = i + batch_size
    print("--- %s seconds ---" % (time.time() - start_time))
    params = [np.vstack([params[i][j] for i in range(len(params))]) for j in range(len(params[0]))]
    pred_meshes = get_proc_meshes(shape, params)
    counter = 0
    print_chamfer_distance(gt_meshes, sa_meshes)
    print_chamfer_distance(gt_meshes, pred_meshes)
    for gt, sa, pr in zip(gt_meshes, sa_meshes, pred_meshes):
        image = np.vstack([render(gt), render(sa), render(pr)])
        Image.fromarray(image).save('visuals/sa/{}_{}.png'.format(shape, counter))
        counter = counter + 1
    print()