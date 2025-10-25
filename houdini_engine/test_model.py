from dataset import ShapeDataset
from proc_shape import paramvectordef
import json, os
import torch
from model import NNProc

param_defs = {}
datafilestr = '../dataset/{}_example.hdf5'


def read_param_def(name):
    with open("../hdas/{}_param_def.json".format(name), "r") as f:
        global param_defs
        param_defs = json.load(f)
        f.close()

def get_param_vector_def():
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

    pvd = paramvectordef.ParamVectorDef()
    if len(scalar_params) > 0:
        pvd.append_param(paramvectordef.ParamType.SCALAR, scalar_params)
    for i in range(toggle_count):
        pvd.append_param(paramvectordef.ParamType.BINARY, None)
    for choice in choice_params:
        pvd.append_param(paramvectordef.ParamType.TYPE, choice)

    return pvd

def predict_param_from_voxel_direct(shape):
    read_param_def(shape)
    pvd = get_param_vector_def()
    dataset = ShapeDataset(datafilestr.format(shape), 'test')
    voxels = dataset.vxl
    params_origin = dataset.prm
    model = NNProc(shape)
    print('Predicting parameters from voxels.')
    model.load_state_dict(torch.load(os.path.join('../new_models', '{}_model.pt'.format(shape))))
    model.eval()
    param_preds_origin = model.param_dec.predict(model.voxel_enc.predict(voxels))
    param_preds = pvd.decode(param_preds_origin)
    params = pvd.decode(params_origin)
    for i in range(len(param_preds)):
        print('予測の結果：{}\n元パラメータ：{}\n'.format(param_preds[i], params[i]))

def main():
    shape = 'table'
    predict_param_from_voxel_direct(shape)

if __name__ == "__main__":
    main()