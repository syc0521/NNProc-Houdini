import os
import sys
import time
import h5py
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from tqdm import tqdm
from utils import print_report

sys.path.insert(0, os.path.join('/', 'mnt', 'Research', 'Codebase', 'DatasetMaker'))
from procedure2 import Procedure

proc = None
#mesh = proc.get_shape([0.5, 0.5, 0.4, 'round', 'solid', 'vbar'])
#mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
target_pcd = None

def get_distance(mesh):
    omesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    pcd = omesh.sample_points_uniformly(number_of_points=10000)
    dist = pcd.compute_point_cloud_distance(target_pcd)
    return sum(dist)

def clip(a):
    return np.clip(a, a_min=0, a_max=1)

def ohe(a):
    return np.eye(a.shape[1])[np.argmax(a, axis=1)]

def rint(a):
    return np.rint(clip(a))

def bed_function(args):
    args = args.reshape(1, 11)
    mesh = proc.get_shape(proc.paramvecdef.decode([args[:,0:5], args[:,5:8], args[:,8:11]])[0])
    return get_distance(mesh)

def chair_function(args):
    args = args.reshape(1, 15)
    mesh = proc.get_shape(proc.paramvecdef.decode([args[:,0:3], args[:,3:7], args[:,7:11], args[:,11:15]])[0])
    return get_distance(mesh)

def shelf_function(args):
    args = args.reshape(1, 16)
    mesh = proc.get_shape(proc.paramvecdef.decode([args[:,0:3], args[:,3:8], args[:,8:13], args[:,13:14], args[:,14:15], args[:,15:16]])[0])
    return get_distance(mesh)

def table_function(args):
    args = args.reshape(1, 11)
    mesh = proc.get_shape(proc.paramvecdef.decode([args[:,0:4], args[:,4:5], args[:,5:11]])[0])
    return get_distance(mesh)

def sofa_function(args):
    args = args.reshape(1, 18)
    mesh = proc.get_shape(proc.paramvecdef.decode([args[:,0:9], args[:,9:12], args[:,12:15], args[:,15:18]])[0])
    return get_distance(mesh)

if __name__ == '__main__':
    shapedict = {'sofa': sofa_function}
    #shapedict = {'bed': bed_function, 'chair': chair_function, 'shelf': shelf_function, 'table': table_function, 'sofa': sofa_function}
    for shape in shapedict.keys():
        f = h5py.File(os.path.join('/', 'mnt', 'Research', 'Data', 'dataset', 'pml', '{}.hdf5'.format(shape)), 'r')
        prm = [f['test/prm/{}'.format(i)][:] for i in range(len(f['test/prm']))]
        verts = [f['test/msh/{}/v'.format(i)][:] for i in range(len(f['test/msh']))]
        faces = [f['test/msh/{}/f'.format(i)][:] for i in range(len(f['test/msh']))]
        proc = Procedure(shape)
        prm_pred = []
        start_time = time.time()
        with tqdm(total=len(verts), file=sys.stdout, desc='Fitting') as progress:
            for v, f in zip(verts, faces):
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
                target_pcd = mesh.sample_points_uniformly(number_of_points=10000)
                guess = np.concatenate(proc.paramvecdef.encode(proc.paramvecdef.get_random_vectors(1)), axis=1)[0]
                bounds = None
                res = minimize(shapedict[shape], guess, method='COBYLA', bounds=bounds, options={'maxiter': 100, 'disp': False})
                prm_pred.append(res.x)
                progress.update()
        prm_pred = np.vstack(prm_pred)
        if shape == 'bed':
            prm_pred = [clip(prm_pred[:,0:5]), ohe(prm_pred[:,5:8]), ohe(prm_pred[:,8:11])]
        elif shape == 'chair':
            prm_pred = [clip(prm_pred[:,0:3]), ohe(prm_pred[:,3:7]), ohe(prm_pred[:,7:11]), ohe(prm_pred[:,11:15])]
        elif shape == 'shelf':
            prm_pred = [clip(prm_pred[:,0:3]), ohe(prm_pred[:,3:8]), ohe(prm_pred[:,8:13]), rint(prm_pred[:,13:14]), rint(prm_pred[:,14:15]), rint(prm_pred[:,15:16])]
        elif shape == 'table':
            prm_pred = [clip(prm_pred[:,0:4]), rint(prm_pred[:,4:5]), ohe(prm_pred[:,5:11])]
        else:
            prm_pred = [clip(prm_pred[:,0:9]), ohe(prm_pred[:,9:12]), ohe(prm_pred[:,12:15]), ohe(prm_pred[:,15:18])]
        prm_pred = proc.paramvecdef.encode(proc.paramvecdef.decode(prm_pred))
        np.savez(os.path.join('predictions', '{}_msh_bsl.npz'.format(shape)), *prm_pred)
        print_report(shape, prm, prm_pred)
        print("--- %s seconds ---" % (time.time() - start_time))