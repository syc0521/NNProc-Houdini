import h5py
import numpy as np
import torch
import trimesh
import utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

class ShapeDataset(Dataset):
    def __init__(self, data_file, mode='train'):
        super(ShapeDataset, self).__init__()
        self.f = h5py.File(data_file, 'r')
        self.f = self.f[mode]
        self.mode = mode
        self.num_data_points = len(self.f['prm'][0])

        mesh_group = self.f['msh']
        voxels = []
        for key in range(mesh_group.__len__()):
            mesh_sub_group = mesh_group[key.__str__()]
            v = np.array(mesh_sub_group['v'])
            f = np.array(mesh_sub_group['f'])
            mesh = trimesh.Trimesh(vertices=v, faces=f)
            voxel = utils.voxelize_mesh(mesh)
            voxels.append(torch.unsqueeze(voxel, dim=0))

        self.vxl = torch.stack(voxels, dim=0)
        self.mesh_shape = self.vxl.shape[1:]
        self.prm = [torch.tensor(np.array(self.f['prm'][i]), dtype=torch.float32) for i in range(len(self.f['prm']))]
        print(self.vxl.shape, self.prm)
        # self.img = torch.tensor(np.array(self.f['img']), dtype=torch.float32) / 255.0
        # self.sil = torch.tensor(np.array(self.f['sil']), dtype=torch.float32) / 255.0

    def __getitem__(self, index):
        datum = {
            'prm': [prm[index] for prm in self.prm],
            'vxl': self.vxl[index],
            # 'img': self.img[index],
            # 'sil': self.sil[index],
        }
        return datum

    def __len__(self):
        return self.num_data_points

    def get_data_loader(self, batch_size=32):
        if self.mode == 'train':
            indices = list(range(self.num_data_points))
            split_point = int(np.floor(0.9 * self.num_data_points))
            train_indices, valid_indices = indices[:split_point], indices[split_point:]
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)
            train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler, num_workers=0)
            valid_loader = DataLoader(self, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
            return train_loader, valid_loader
        else:
            indices = list(range(self.num_data_points))
            sampler = SubsetRandomSampler(indices)
            loader = DataLoader(self, batch_size=batch_size, sampler=sampler, num_workers=0)
            return loader
