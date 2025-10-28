import sys
import math
from tqdm import tqdm
import trimesh
from trimesh import transformations
import numpy as np
import pyvista as pv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import json

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
    from proc_shape.procedure import Procedure
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
    from model import ShapeInfo
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

def voxelize_mesh_faster(mesh: trimesh.Trimesh, visualize=False):
    import open3d as o3d
    rsl = 64

    vertices = np.asarray(mesh.vertices)
    # calculate bounding box
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    # calculate scale
    scale = 1.0 / np.max(max_bound - min_bound)

    mesh.vertices = o3d.utility.Vector3dVector((vertices - min_bound) * scale)
    omesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        omesh,
        voxel_size=1 / 63
    )
    if visualize:
        o3d.visualization.draw_geometries([voxel_grid])
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    indices = indices + (np.array([rsl-1, rsl-1, rsl-1]) - np.max(indices, axis=0)) // 2
    voxel_arr = np.zeros((rsl, rsl, rsl), dtype=np.uint8)
    voxel_arr[tuple(indices.T)] = 1
    del o3d
    return torch.tensor(voxel_arr, dtype=torch.float32)

def read_param_def(name):
    param_defs = {}
    with open("hdas/{}_param_def.json".format(name), "r") as f:
        param_defs = json.load(f)
        f.close()
    return param_defs

def get_param_vector_def(param_defs):
    scalar_params = []
    toggle_count = 0
    choice_params = []

    if 'float' in param_defs:
        for param in param_defs['float']:
            scalar_params.append([0.0, 1.0])

    if 'bool' in param_defs:
        toggle_count = toggle_count + len(param_defs['bool'])

    if 'choice' in param_defs:
        for param in param_defs['choice']:
            if param['choices'] is not None:
                choice_params.append(param['choices'])

    from proc_shape import paramvectordef
    pvd = paramvectordef.ParamVectorDef()
    if len(scalar_params) > 0:
        pvd.append_param(paramvectordef.ParamType.SCALAR, scalar_params)
    for i in range(toggle_count):
        pvd.append_param(paramvectordef.ParamType.BINARY, None)
    for choice in choice_params:
        pvd.append_param(paramvectordef.ParamType.TYPE, choice)

    return pvd