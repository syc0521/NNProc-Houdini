import io
import os
import sys
import time
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import trimesh
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import save_image
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
)
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from dataset import ShapeDataset
from model import z_dim, kl, NNProc, VoxelLoss
from utils import get_proc_meshes, render, print_report
sys.path.insert(0, os.path.join('/', 'mnt', 'Research', 'Codebase', 'DatasetMaker'))
#from procedure2 import Procedure

datafilestr = os.path.join('/', 'mnt', 'Research', 'Data', 'dataset', 'pml', '{}.hdf5')

class OptimDataset(ShapeDataset):
    def __init__(self, data_file, mode='train'):
        super(OptimDataset, self).__init__(data_file, mode)
        self.mu = nn.Parameter(torch.zeros(self.num_data_points, z_dim), requires_grad=True)
        self.logvar = nn.Parameter(torch.ones(self.num_data_points, z_dim), requires_grad=True)

    def __getitem__(self, index):
        datum = super().__getitem__(index)
        datum['mu'] = self.mu[index]
        datum['logvar'] = self.logvar[index]
        return datum

    def __len__(self):
        return self.num_data_points

class VolRenderModel(nn.Module):
    def __init__(self, shape, camera_locations):
        super(VolRenderModel, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.proc = NNProc(shape)
        self.proc.load_state_dict(torch.load(os.path.join('models', shape + '_model.pt')))
        self.proc.eval()

        render_size = 128
        volume_extent_world = 1.0
        volume_size = 128
        self.voxel_size = volume_extent_world / volume_size
        self.R, self.T = [], []
        for cl in camera_locations:
            R, T = look_at_view_transform(
                dist=cl[0],
                elev=cl[1],
                azim=cl[2],
                device=self.device
            )
            self.R.append(R)
            self.T.append(T)
        self.v_renderer = VolumeRenderer(
            raysampler=NDCMultinomialRaysampler(
                image_width=render_size,
                image_height=render_size,
                n_pts_per_ray=150,
                min_depth=-1.5,
                max_depth=1.5,
            ),
            raymarcher=EmissionAbsorptionRaymarcher(),
            sample_mode='nearest'
        )

    def forward(self, mu, logvar):
        mu, logvar = mu.to(self.device), logvar.to(self.device)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar).to(self.device)
        self.proc.eval()
        densities = self.proc.voxel_dec(z)
        volumes = Volumes(
            densities=densities,
            features=(densities.repeat(1, 3, 1, 1, 1)),
            voxel_size=self.voxel_size,
        )
        images = []
        for R, T in zip(self.R, self.T):
            cameras = FoVPerspectiveCameras(
                fov=30,
                device=self.device,
                R=R.repeat(len(volumes), 1, 1),
                T=T.repeat(len(volumes), 1),
                znear=-100,
                zfar=100)
            colors, opacity = self.v_renderer(cameras=cameras, volumes=volumes)[0].split([3, 1], dim=-1)
            #images.append(colors[:, :, :, 0])
            images.append(opacity[:, :, :, 0])

        return torch.stack(images, dim=3)

    def huber(self, x, y, scaling=0.1):
        diff_sq = (x - y) ** 2
        loss = (((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)).abs().mean()
        return loss

def predict_voxel_from_param_direct(shape):
    dataset = ShapeDataset(datafilestr.format(shape), 'test')
    params = dataset.prm
    model = NNProc(shape)
    model.load_state_dict(torch.load(os.path.join('models', '{}_model.pt'.format(shape))))
    model.eval()
    print('Predicting voxels from parameters.')
    voxel_preds = model.voxel_dec.predict(model.param_enc.predict(params))
    voxel_preds = [voxel_preds[i][0] for i in range(voxel_preds.shape[0])]
    np.savez(os.path.join('predictions', '{}_prm_dir.npz'.format(shape)), *voxel_preds)

def predict_param_from_voxel_direct(shape):
    dataset = ShapeDataset(datafilestr.format(shape), 'test')
    voxels = dataset.vxl
    model = NNProc(shape)
    print('Predicting parameters from voxels.')
    model.load_state_dict(torch.load(os.path.join('models', '{}_model.pt'.format(shape))))
    model.eval()
    param_preds = model.param_dec.predict(model.voxel_enc.predict(voxels))
    np.savez(os.path.join('predictions', '{}_vxl_dir.npz'.format(shape)), *param_preds)

def predict_param_from_voxel_optim(shape):
    dataset = OptimDataset(datafilestr.format(shape), 'test')
    data_loader = dataset.get_data_loader()
    model = NNProc(shape)
    model.load_state_dict(torch.load(os.path.join('models', shape + '_model.pt')))
    model.eval()
    loss_fn = VoxelLoss()
    optim = torch.optim.Adam([dataset.mu, dataset.logvar], lr=0.1)
    num_epochs = 1000
    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.01, total_iters=num_epochs)
    for e in range(num_epochs):
        train_loss = 0.0
        it = iter(data_loader)
        for i, batch in enumerate(tqdm(it, file=sys.stdout)):
            optim.zero_grad()
            mu = batch['mu'].cuda()
            logvar = batch['logvar'].cuda()
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar).cuda()
            predictions = model.voxel_dec(z)
            loss = loss_fn(predictions, batch['vxl']) + kl(mu, logvar)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(data_loader.sampler)
        scheduler.step()
        print('Epoch {}: Loss: {:.5f}'.format(e+1, train_loss))

    param_preds = model.param_dec.predict(dataset.mu)
    np.savez(os.path.join('predictions', '{}_vxl_opt.npz'.format(shape)), *param_preds)

def predict_param_from_image_optim(shape):
    dataset = OptimDataset(datafilestr.format(shape), 'test')
    params = dataset.prm
    data_loader = dataset.get_data_loader(batch_size=4)
    model = VolRenderModel(shape, [[1.1, i, j] for i in [20, 35] for j in [150, 165]])
    optim = torch.optim.Adam([dataset.mu, dataset.logvar], lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    num_epochs = 100
    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    for e in range(num_epochs):
        train_loss = 0.0
        it = iter(data_loader)
        for i, batch in enumerate(tqdm(it, file=sys.stdout)):
            optim.zero_grad()
            mu = batch['mu'].cuda()
            logvar = batch['logvar'].cuda()
            prediction = model(mu, logvar)
            target = batch['sil'].cuda()
            loss = loss_fn(prediction, target) + kl(mu, logvar)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(data_loader.sampler)
        scheduler.step()
        print('Epoch {}: Loss: {:.5f}'.format(e + 1, train_loss))
        with (torch.no_grad()):
            img = model(dataset.mu[0:1], dataset.logvar[0:1])[0]
            img = torch.cat(
                [
                    torch.cat([img[:, :, 0], img[:, :, 1]], dim=1),
                    torch.cat([img[:, :, 2], img[:, :, 3]], dim=1)
                ], dim=0
            )
            save_image(img, 'img.jpg')
            print_report(
                shape,
                model.proc.param_dec.predict(
                    data_loader.dataset.mu
                ),
                [x.numpy() for x in params]
            )
            torch.cuda.empty_cache()
    param_preds = model.proc.param_dec.predict(dataset.mu)
    np.savez(os.path.join('predictions', '{}_img_opt.npz'.format(shape)), *param_preds)

def read_params(filename):
    container = np.load(os.path.join('predictions', filename))
    data = [container[key] for key in container]
    return data


def gather_shapes(shape):
    f = h5py.File(os.path.join('/', 'mnt', 'Research', 'Data', 'dataset', 'pml', shape + '.hdf5'), 'r')
    meshes = [trimesh.Trimesh(vertices=f['test/msh/{}/v'.format(i)], faces=f['test/msh/{}/f'.format(i)]) for i in range(len(f['test/msh']))]
    voxels = [f['test/vxl'][i][0] for i in range(f['test/vxl'].shape[0])]
    voxel_preds = [voxel for voxel in read_params('{}_prm_dir.npz'.format(shape))]
    meshes_baseline = get_proc_meshes(shape, read_params('{}_msh_bsl.npz'.format(shape)))
    meshes_vxl_direct = get_proc_meshes(shape, read_params('{}_vxl_dir.npz'.format(shape)))
    meshes_vxl_optim = get_proc_meshes(shape, read_params('{}_vxl_opt.npz'.format(shape)))
    meshes_img_optim = get_proc_meshes(shape, read_params('{}_img_opt.npz'.format(shape)))
    return meshes, voxels, voxel_preds, meshes_baseline, meshes_vxl_direct, meshes_vxl_optim, meshes_img_optim


def print_voxel_similarity(voxels, voxel_preds):
    sim = np.sum(np.stack(voxels) ^ np.stack(voxel_preds)) * 100 / 64 ** 3 / len(voxels)
    print(sim)


def print_chamfer_distance(meshes, pred_meshes):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    batch_size = 200
    dist = 0.0
    with torch.no_grad():
        for i in range(0, len(meshes), batch_size):
            start, end = i, min(i + batch_size, len(meshes))
            verts = [torch.tensor(mesh.vertices, dtype=torch.float32) for mesh in meshes[start:end]]
            faces = [torch.tensor(mesh.faces, dtype=torch.int64) for mesh in meshes[start:end]]
            msh = Meshes(verts=verts, faces=faces).to(device)
            points = sample_points_from_meshes(msh)
            verts = [torch.tensor(mesh.vertices, dtype=torch.float32) for mesh in pred_meshes[start:end]]
            faces = [torch.tensor(mesh.faces, dtype=torch.int64) for mesh in pred_meshes[start:end]]
            msh = Meshes(verts=verts, faces=faces).to(device)
            pred_points = sample_points_from_meshes(msh)
            dist += chamfer_distance(points, pred_points, batch_reduction='sum')[0]
            torch.cuda.empty_cache()
    dist = dist / len(meshes)
    print(dist)

def save_predictions():
    #for shape in ['table']:
    for shape in ['bed', 'chair', 'shelf', 'table', 'sofa']:
        #predict_voxel_from_param_direct(shape)
        predict_param_from_voxel_direct(shape)
        #predict_param_from_voxel_optim(shape)
        #predict_param_from_image_optim(shape)

def save_visuals():
    #for shape in ['table']:
    for shape in ['bed', 'chair', 'shelf', 'table', 'sofa']:
        meshes, voxels, voxel_preds, meshes_baseline,  meshes_vxl_direct, meshes_vxl_optim, meshes_img_optim = gather_shapes(shape)
        print_voxel_similarity(voxels, voxel_preds)
        print_chamfer_distance(meshes, meshes_baseline)
        print_chamfer_distance(meshes, meshes_vxl_direct)
        print_chamfer_distance(meshes, meshes_vxl_optim)
        print_chamfer_distance(meshes, meshes_img_optim)
        continue
        voxels = [trimesh.voxel.VoxelGrid(voxel).as_boxes() for voxel in voxels]
        voxel_preds = [trimesh.voxel.VoxelGrid(voxel).as_boxes() for voxel in voxel_preds]
        counter = 0
        for m, v, vp, mb, mvd, mvo, mio in zip(meshes, voxels, voxel_preds, meshes_baseline, meshes_vxl_direct, meshes_vxl_optim, meshes_img_optim):
            image = np.hstack([render(m), render(v), render(vp), render(mb), render(mvd), render(mvo), render(mio)])
            Image.fromarray(image).save('visuals/{}_{}.png'.format(shape, counter))
            counter = counter + 1

def temp():
    for shape in ['bed', 'chair', 'shelf', 'table', 'sofa']:
        f = h5py.File(os.path.join('/', 'mnt', 'Research', 'Data', 'dataset', 'pml', shape + '.hdf5'), 'r')
        param = [f['test/prm/{}'.format(i)][:] for i in range(len(f['test/prm']))]
        param_preds = read_params('{}_img_opt.npz'.format(shape))
        print_report(
            shape,
            param,
            param_preds
        )

def predict_params_from_mesh(mesh, shape_type):
    # 1. 将mesh转换为64x64x64体素
    voxel_grid = trimesh.voxel.VoxelGrid(mesh, pitch=1.0/64)
    voxels = voxel_grid.matrix.astype(np.float32)
    voxels = torch.from_numpy(voxels).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度

    # 2. 加载训练好的模型
    model = NNProc(shape_type)
    model.load_state_dict(torch.load(os.path.join('models', f'{shape_type}_model.pt')))
    model.eval()

    # 3. 使用模型预测参数
    with torch.no_grad():
        # 先用体素编码器得到潜在编码
        latent = model.voxel_enc.predict(voxels)
        # 再用参数解码器得到最终参数
        params = model.param_dec.predict(latent)

    return params[0]  # 返回预测的参数

if __name__ == "__main__":
    #save_predictions()
    #save_visuals()
    temp()

