import io
import os
import sys
import open3d as o3d
import math
import shutil
import trimesh
import numpy as np
import pyvista as pv
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import save_image
from model import z_dim, kl, NNProc
from utils import get_proc_meshes, render, print_report
from applications import VolRenderModel
sys.path.insert(0, os.path.join('/', 'mnt', 'Research', 'Codebase', 'DatasetMaker'))
#from procedure2 import Procedure
from makerawdata import voxelize_mesh # todo voxelize_mesh

shapenetdict = {
    'bed': ('02818832', ['c2b65540d51fada22bfa21768064df9c']),
    'chair': ('03001627',['92c176547858fd2cf91663a74ccd2338']),
    'shelf': ('02871439',['3fe283e402ab8785937a1594070247a0']),
    'table': ('04379243',['7bbd4240f837e20a6716685cf333d2c1']),
    'sofa': ('04256520',['1d8716f741424998f29666f384be6c43'])
}

pix3ddict = {
    'bed': ('0905', [1.3, 10, 135]),
    'chair': ('3405', [1.2, 10, 135]),
    'shelf': ('0060', [1.0, 10, 110]),
    #'table': ('0125', [0.6, 30, 30]),
    'table': ('0283', [1.2, 10, 0]),
    'sofa': ('0104', [0.55, 30, 135])
}

def resize_image(imgpath, size):
    img = Image.open(imgpath)
    expected_size = (size, size)
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    img = ImageOps.expand(img, padding)
    return img

def shape_to_shape_direct(shape):
    meshes = []
    voxels = []
    for x in shapenetdict[shape][1]:
        shapedir = os.path.join('/', 'mnt', 'Research', 'ShapeNetCore.v2', shapenetdict[shape][0], x, 'models')
        mesh = trimesh.load(os.path.join(shapedir, 'model_normalized.obj'), force='mesh')
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        t = np.sum(mesh.bounding_box.bounds, axis=0) / 2
        mesh.apply_translation(-t)
        s = np.max(mesh.extents)
        mesh.apply_scale(1 / s)
        angle = math.pi
        direction = [0, 1, 0]
        center = [0, 0, 0]
        rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
        mesh.apply_transform(rot_matrix)
        meshes.append(mesh)
        voxels.append(
            torch.unsqueeze(
                torch.unsqueeze(
                    torch.tensor(
                        voxelize_mesh(mesh),
                        dtype=torch.float32),
                    dim=0
                ),
                dim=0
            )
        )
    voxels = torch.cat(voxels, dim=0).cuda()
    model = NNProc(shape)
    model.load_state_dict(torch.load(os.path.join('models', shape + '_model.pt')))
    model.eval()

    param_pred = model.param_dec.predict(model.voxel_enc.predict(voxels))
    mesh_pred = get_proc_meshes(shape, param_pred)
    for i in range(len(mesh_pred)):
        meshes[i].export(os.path.join('realdata', 'shapenet', shape + '_{}.obj'.format(i + 1)))
        mesh_pred[i].export(os.path.join('realdata', 'shapenet', shape + '_pred_{}.obj'.format(i + 1)))
        shutil.copy(
            os.path.join('/', 'mnt', 'Research', 'ShapeNetCoreImages.v2', 'thumbs', shapenetdict[shape][0],
                         shapenetdict[shape][1][i] + '.png'),
            os.path.join('realdata', 'shapenet', shape + '_{}.png'.format(i + 1))
        )
        p = pv.Plotter(off_screen=True, window_size=(256, 256))
        v, f = meshes[i].vertices, np.hstack(
            [np.ones((meshes[i].faces.shape[0], 1), dtype=int) * 3, meshes[i].faces])
        pvmesh = pv.PolyData(v, f)
        pvmesh = pvmesh.rotate_x(90)
        pvmesh = pvmesh.rotate_z(90)
        actor = p.add_mesh(pvmesh, style="wireframe", color='black')
        p.reset_camera()
        img = p.show(return_img=True, auto_close=False)
        p.remove_actor(actor)
        Image.fromarray(img).save(os.path.join('realdata', 'shapenet', shape + '_orig_{}.png'.format(i + 1)))
        Image.fromarray(render(mesh_pred[i])).save(os.path.join('realdata', 'shapenet', shape + '_pred_{}.png'.format(i + 1)))

def pc_to_shape_direct(shape):
    meshes = []
    voxels = []
    for x in shapenetdict[shape][1]:
        shapedir = os.path.join('/', 'mnt', 'Research', 'ShapeNetCore.v2', shapenetdict[shape][0], x, 'models')
        mesh = o3d.io.read_triangle_mesh(os.path.join(shapedir, 'model_normalized.obj'))
        mesh.translate(tuple(-mesh.get_center()))
        mesh.scale(1.0/np.max(mesh.get_max_bound()-mesh.get_min_bound()), mesh.get_center())
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi, 0)))
        meshes.append(mesh)
        pcd = mesh.sample_points_uniformly(number_of_points=500)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=1.0/63)
        vox_tensor = torch.zeros(1, 1, 64, 64, 64)
        offset = (63 - np.max(np.vstack([x.grid_index for x in voxel_grid.get_voxels()]), axis=0)) // 2
        for x in voxel_grid.get_voxels():
            co = x.grid_index + offset
            vox_tensor[0, 0, co[0], co[1], co[2]] = 1
        voxels.append(vox_tensor)
        #o3d.visualization.draw_geometries([pcd])
        '''
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=256, height=256)
        R = pcd.get_rotation_matrix_from_xyz((np.pi / 8, -np.pi / 4, 0))
        pcd.rotate(R, center=(0, 0, 0))
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 9.0
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join('realdata', 'shapenetpc', shape + '_1.png'), do_render=True)
        vis.destroy_window()
        '''
    voxels = torch.cat(voxels, dim=0).cuda()
    model = NNProc(shape)
    model.load_state_dict(torch.load(os.path.join('models', shape + '_model.pt')))
    model.eval()

    param_pred = model.param_dec.predict(model.voxel_enc.predict(voxels))
    mesh_pred = get_proc_meshes(shape, param_pred)
    for i in range(len(mesh_pred)):
        o3d.io.write_triangle_mesh(
            os.path.join('realdata', 'shapenetpc', shape + '_{}.obj'.format(i + 1)),
            meshes[i],
            write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False
        )
        mesh_pred[i].export(os.path.join('realdata', 'shapenetpc', shape + '_pred_{}.obj'.format(i + 1)))
        '''
        shutil.copy(
            os.path.join('/', 'mnt', 'Research', 'ShapeNetCoreImages.v2', 'thumbs', shapenetdict[shape][0],
                         shapenetdict[shape][1][i] + '.png'),
            os.path.join('realdata', 'shapenetpc', shape + '_{}.png'.format(i + 1))
        )
        '''
        Image.fromarray(render(mesh_pred[i])).save(os.path.join('realdata', 'shapenetpc', shape + '_pred_{}.png'.format(i + 1)))

def image_to_shape_optim(shape):
    imgpath = os.path.join('/', 'home', 'ishtiaque', 'Downloads', 'pix3d', 'mask', shape, pix3ddict[shape][0]+'.png')
    img = resize_image(imgpath, 128)
    target = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(img) / 255.0, dtype=torch.float32), dim=0), dim=3).cuda()
    #img.show()
    model = VolRenderModel(shape, [pix3ddict[shape][1]])
    mu = nn.Parameter(torch.zeros(1, z_dim), requires_grad=True)
    logvar = nn.Parameter(torch.ones(1, z_dim), requires_grad=True)
    optim = torch.optim.Adam([mu, logvar], lr=0.01)
    num_epochs = 500
    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.01, total_iters=num_epochs)
    loss_fn = nn.MSELoss()
    for e in range(num_epochs):
        optim.zero_grad()
        prediction = model(mu, logvar)
        loss = loss_fn(prediction, target) + 0.001 * kl(mu, logvar)
        loss.backward()
        optim.step()
        train_loss = loss.item()
        scheduler.step()
        print('Epoch {}: Loss: {:.5f}'.format(e + 1, train_loss))
        with (torch.no_grad()):
            img = model(mu, logvar)[0]
            save_image(img[:,:,0], 'img.jpg')
            torch.cuda.empty_cache()
    param_pred = model.proc.param_dec.predict(mu)
    mesh_pred = get_proc_meshes(shape, param_pred)
    for i in range(len(mesh_pred)):
        ext = '.jpg' if shape == 'table' else '.jpg'
        imgpath = os.path.join('/', 'home', 'ishtiaque', 'Downloads', 'pix3d', 'img', shape, pix3ddict[shape][0] + ext)
        resize_image(imgpath, 256).save(os.path.join('realdata', 'pix3d', shape + '_{}.png'.format(i+1)))
        mesh_pred[i].export(os.path.join('realdata', 'pix3d', shape + '_pred_{}.obj'.format(i+1)))
        Image.fromarray(render(mesh_pred[i])).save(os.path.join('realdata', 'pix3d', shape + '_pred_{}.png'.format(i + 1)))

def gs_to_shape_optim(shape):
    imgpath = os.path.join('/', 'mnt', 'Volume4', 'gaussian-splatting', 'output', '6e632a13-8', 'renders_30000', 'renders')
    num_images = len(os.listdir(imgpath)) // 2
    exclusions = [28]
    images = [np.array(Image.open(os.path.join(imgpath, '{0:05d}_2.png'.format(i))).convert('L')) / 255.0 for i in range(num_images) if i not in exclusions]
    images = [torch.unsqueeze(torch.unsqueeze(torch.tensor(img, dtype=torch.float32), dim=0), dim=3) for img in images]
    dist, azim = 1.1, 35
    elevs = [
        16, 8, 0, 352, 344, 336, 328, 320, 312, 304, 296, 288, 280, 272, 268, 261, 252, 243, 234, 225, 216, 207, 198, 189, 180,
        173, 164, 155, 146, 137, 128, 119, 110, 101, 92, 88, 80, 71, 63, 54, 45, 36, 27, 18, 9, 0, 352, 344, 336, 328
    ]
    cams = [[dist, azim, x] for x in elevs]
    #nums = 32
    #indices = np.linspace(0, 49, nums+1, dtype=int)[:-1].tolist()
    #images, cams = [images[x] for x in indices], [cams[x] for x in indices]
    model = VolRenderModel(shape, cams)
    mu = nn.Parameter(torch.zeros(1, z_dim), requires_grad=True)
    logvar = nn.Parameter(torch.ones(1, z_dim), requires_grad=True)
    optim = torch.optim.Adam([mu, logvar], lr=0.01)
    target = torch.cat(images, dim=3).cuda()
    num_epochs = 300
    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.01, total_iters=num_epochs)
    loss_fn = nn.MSELoss()
    for e in range(num_epochs):
        optim.zero_grad()
        prediction = model(mu, logvar)
        loss = loss_fn(prediction, target) + 0.0001 * kl(mu, logvar)
        loss.backward()
        optim.step()
        train_loss = loss.item()
        scheduler.step()
        print('Epoch {}: Loss: {:.5f}'.format(e + 1, train_loss))
        with (torch.no_grad()):
            img = model(mu, logvar)[0]
            save_image(img[:, :, 0], 'img.jpg')
            torch.cuda.empty_cache()
    param_pred = model.proc.param_dec.predict(mu)
    mesh_pred = get_proc_meshes(shape, param_pred)
    for i in range(len(mesh_pred)):
        mesh_pred[i].export(os.path.join('realdata', '3dgs', shape + '_pred_{}.obj'.format(i+1)))

def save_predictions():
    shape = 'shelf'
    for shape in ['chair']:
    #for shape in ['bed', 'chair', 'shelf', 'table']:
        #shape_to_shape_direct(shape)
        #pc_to_shape_direct(shape)
        #image_to_shape_optim(shape)
        if shape == 'chair':
            gs_to_shape_optim(shape)


if __name__ == '__main__':
    save_predictions()
