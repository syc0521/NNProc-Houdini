import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


z_dim = 128


def fc_layer_set(in_c, out_c, activation=True):
    fc_layer = nn.Sequential(
        nn.Linear(in_c, out_c)
    )
    if activation:
        fc_layer.append(nn.LeakyReLU())
    return fc_layer


def conv_layer_set(in_c, out_c, num=1):
    sz = np.linspace(in_c, out_c, num=num+1, dtype=np.int32)
    conv_layer = nn.Sequential()
    for i in range(len(sz)-1):
        conv_layer.append(nn.Conv3d(sz[i], sz[i+1], kernel_size=3, stride=1, padding=1))
        conv_layer.append(nn.LeakyReLU())
        conv_layer.append(nn.BatchNorm3d(sz[i+1]))
    conv_layer.append(nn.Conv3d(out_c, out_c, kernel_size=2, stride=2, padding=0))
    conv_layer.append(nn.LeakyReLU())
    conv_layer.append(nn.BatchNorm3d(out_c))
    return conv_layer


def deconv_layer_set(in_c, out_c, num=1, is_last=False):
    sz = np.linspace(in_c, out_c, num=num+1, dtype=np.int32)
    deconv_layer = nn.Sequential()
    deconv_layer.append(nn.ConvTranspose3d(in_c, in_c, kernel_size=2, stride=2, padding=0))
    deconv_layer.append(nn.LeakyReLU())
    deconv_layer.append(nn.BatchNorm3d(in_c))
    for i in range(len(sz) - 1):
        deconv_layer.append(nn.ConvTranspose3d(sz[i], sz[i + 1], kernel_size=3, stride=1, padding=1))
        deconv_layer.append(nn.LeakyReLU())
        deconv_layer.append(nn.BatchNorm3d(sz[i + 1]))
    if is_last:
        deconv_layer.append(nn.Conv3d(out_c, out_c, kernel_size=1, stride=1, padding=0))
        deconv_layer.append(nn.Sigmoid())
    return deconv_layer


def kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def move_to_device(x, device):
    if isinstance(x, list):
        for i in range(len(x)):
            if not isinstance(x[i], torch.Tensor):
                x[i] = torch.tensor(x[i])
        return [i.to(device) if i.device != device else i for i in x]
    else:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.device != device:
            return x.to(device)
    return x


class ParamInfo:
    def __init__(self, paramtype, num_output):
        self.paramtype = paramtype
        self.num_output = num_output


class ShapeInfo:
    def __init__(self, shape):
        self.params = []
        self.shape = shape
        if shape == 'bed':
            self.params.append(ParamInfo('scalar', 5))
            self.params.append(ParamInfo('type', 3))
            self.params.append(ParamInfo('integer', 3))
        elif shape == 'chair':
            self.params.append(ParamInfo('scalar', 3))
            self.params.append(ParamInfo('type', 4))
            self.params.append(ParamInfo('type', 4))
            self.params.append(ParamInfo('type', 4))
        elif shape == 'shelf':
            self.params.append(ParamInfo('scalar', 3))
            self.params.append(ParamInfo('integer', 5))
            self.params.append(ParamInfo('integer', 5))
            self.params.append(ParamInfo('binary', 1))
            self.params.append(ParamInfo('binary', 1))
            self.params.append(ParamInfo('binary', 1))

        elif shape == 'table':
            self.params.append(ParamInfo('scalar', 4))
            self.params.append(ParamInfo('binary', 1))
            self.params.append(ParamInfo('type', 6))

        elif shape == 'sofa':
            self.params.append(ParamInfo('scalar', 9))
            self.params.append(ParamInfo('integer', 3))
            self.params.append(ParamInfo('integer', 3))
            self.params.append(ParamInfo('integer', 3))


class ParamEncoder(nn.Module):
    def __init__(self, shape):
        super(ParamEncoder, self).__init__()
        self.shape = ShapeInfo(shape=shape)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        in_c = sum([param.num_output for param in self.shape.params])
        self.fc1 = fc_layer_set(in_c, z_dim * 2)
        self.fc2 = fc_layer_set(z_dim * 2, z_dim * 2)
        self.fc3 = fc_layer_set(z_dim * 2, z_dim, activation=False)
        self.to(self.device)

    def forward(self, x):
        x = move_to_device(x, self.device)
        x = torch.hstack(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            torch.cuda.empty_cache()
            return x


class ParamDecoder(nn.Module):
    def __init__(self, shape):
        super(ParamDecoder, self).__init__()
        self.shape = ShapeInfo(shape=shape)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.fc1 = fc_layer_set(z_dim, z_dim * 2)
        self.fc2 = fc_layer_set(z_dim * 2, z_dim * 2)
        self.out_modules = nn.ModuleList()
        for param in self.shape.params:
            module = list()
            module.append(fc_layer_set(z_dim * 2, param.num_output, activation=False))
            if param.paramtype == 'scalar' or param.paramtype == 'binary':
                module[0].append(nn.Sigmoid())
            module = nn.Sequential(*module)
            self.out_modules.append(module)
        self.to(self.device)

    def forward(self, x):
        x = move_to_device(x, self.device)
        x = self.fc1(x)
        x = self.fc2(x)
        x = [module(x) for module in self.out_modules]
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            y = []
            for i, param in enumerate(self.shape.params):
                if param.paramtype == 'type' or param.paramtype == 'integer':
                    y.append(F.one_hot(torch.argmax(F.softmax(x[i], dim=1), dim=1), num_classes=param.num_output).cpu().numpy())
                elif param.paramtype == 'scalar':
                    y.append(x[i].cpu().numpy())
                else:
                    y.append(torch.round(x[i]).cpu().numpy())
            torch.cuda.empty_cache()
        return y


class VoxelEncoder(nn.Module):
    def __init__(self):
        super(VoxelEncoder, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.conv1 = conv_layer_set(1, 8, 1)
        self.conv2 = conv_layer_set(8, 16, 1)
        self.conv3 = conv_layer_set(16, 24, 1)
        self.conv4 = conv_layer_set(24, 32, 1)
        self.conv5 = conv_layer_set(32, 40, 1)
        self.flatten = nn.Flatten()
        self.fc = fc_layer_set(320, 2 * z_dim)
        self.fc_mu = fc_layer_set(2 * z_dim, z_dim, activation=False)
        self.fc_logvar = fc_layer_set(2 * z_dim, z_dim, activation=False)
        self.to(self.device)

    def forward(self, x):
        x = move_to_device(x, self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if self.training:
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar).to(self.device)
        else:
            x = mu
        return x, mu, logvar

    def predict(self, x):
        with torch.no_grad():
            _, mu, _ = self.forward(x)
            torch.cuda.empty_cache()
            return mu


class VoxelDecoder(nn.Module):
    def __init__(self):
        super(VoxelDecoder, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.fc = fc_layer_set(z_dim, 320)
        self.unflatten = nn.Unflatten(1, (40, 2, 2, 2))
        self.deconv1 = deconv_layer_set(40, 32, 1)
        self.deconv2 = deconv_layer_set(32, 24, 1)
        self.deconv3 = deconv_layer_set(24, 16, 1)
        self.deconv4 = deconv_layer_set(16, 8, 1)
        self.deconv5 = deconv_layer_set(8, 1, 1, is_last=True)
        self.to(self.device)

    def forward(self, x):
        x = move_to_device(x, self.device)
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            torch.cuda.empty_cache()
            return np.array(np.rint(x.cpu().numpy()), dtype=np.uint8)


class NNProc(nn.Module):
    def __init__(self, shape):
        super(NNProc, self).__init__()
        self.param_enc = ParamEncoder(shape=shape)
        self.param_dec = ParamDecoder(shape=shape)
        self.voxel_enc = VoxelEncoder()
        self.voxel_dec = VoxelDecoder()

    def forward(self, param, voxel):
        p_z = self.param_enc(param)
        v_z, mu, logvar = self.voxel_enc(voxel)
        param_recon = self.param_dec(mu)
        voxel_recon = self.voxel_dec(v_z)
        return param_recon, voxel_recon, p_z, mu, logvar


class ParamLoss(nn.Module):
    def __init__(self, shape):
        super(ParamLoss, self).__init__()
        self.shape = ShapeInfo(shape=shape)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.loss_fn = []
        for param in self.shape.params:
            if param.paramtype == 'scalar':
                self.loss_fn.append(nn.MSELoss(reduction='sum'))
            elif param.paramtype == 'type':
                self.loss_fn.append(nn.CrossEntropyLoss(reduction='sum'))
            elif param.paramtype == 'integer':
                self.loss_fn.append(nn.CrossEntropyLoss(reduction='sum'))
            elif param.paramtype == 'binary':
                self.loss_fn.append(nn.BCELoss(reduction='sum'))
        if torch.cuda.is_available():
            self.to(self.device)

    def forward(self, predictions, targets):
        targets = [i.to(self.device) for i in targets]
        losses = []
        for i, loss in enumerate(self.loss_fn):
            losses.append(loss(predictions[i], targets[i]))
        total_loss = torch.stack(losses).sum()
        return total_loss


class VoxelLoss(nn.Module):
    def __init__(self):
        super(VoxelLoss, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.to(self.device)

    def forward(self, predictions, targets):
        targets = targets.to(self.device)
        loss = self.loss_fn(predictions, targets)
        return loss


class LatentLoss(nn.Module):
    def __init__(self):
        super(LatentLoss, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.to(self.device)

    def forward(self, z_, z):
        loss = self.loss_fn(z_, z)
        return loss


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.to(self.device)

    def forward(self, param, z):
        x = torch.hstack(param).to(self.device)
        loss = self.loss_fn(
            self.cos(x.repeat(x.shape[0], 1), torch.repeat_interleave(x, x.shape[0], dim=0)),
            self.cos(z.repeat(z.shape[0], 1), torch.repeat_interleave(z, z.shape[0], dim=0))
        )
        return loss


class NNProcLoss(nn.Module):
    def __init__(self, shape):
        super(NNProcLoss, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.param_loss_fn = ParamLoss(shape)
        self.voxel_loss_fn = VoxelLoss()
        self.latent_loss_fn = LatentLoss()
        self.cosine_loss_fn = CosineLoss()

    def forward(self, param_recon, voxel_recon, p_z, mu, logvar, param, voxel):
        losses = torch.stack(
            [
                self.param_loss_fn(param_recon, param),
                self.voxel_loss_fn(voxel_recon, voxel),
                self.latent_loss_fn(p_z, mu),
                self.cosine_loss_fn(param, mu),
                kl(mu, logvar)
            ]
        )
        return losses.sum()
