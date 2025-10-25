import he_init
from proc_shape import paramvectordef
import json, os
import torch
from model import NNProc
from he_manager import he_instance
import numpy as np
import trimesh

param_defs = {}

def read_param_def(name):
    with open("../hdas/{}_param_def.json".format(name), "r") as f:
        global param_defs
        param_defs = json.load(f)
        f.close()

def init_houdini(shape_type):
    session = he_init.create_session()
    if session is None:
        print("Failed to create the Houdini Engine session.")
        return False

    hda_folder = "../hdas"
    hda_loaded, asset_name = he_instance.loadAsset(os.path.join(hda_folder, shape_type) + ".hda")
    if not hda_loaded:
        print("Failed to load the HDA.")
        return False

    hda_cooked = he_instance.createAndCookNode(asset_name, 0)
    if not hda_cooked:
        print("Failed to create and cook the HDA node.")
        return False

    return True

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

def generate_model(params):
    float_value = []
    int_value = []
    bool_value = []
    choice_value = []

    for param in params:
        if type(param) == float:
            float_value.append(param)
        elif type(param) == bool:
            bool_value.append(param)
        elif type(param) == int:
            int_value.append(param)
        else:
            choice_value.append(param)

    target_node = he_instance.getNode(0)
    if 'float' in param_defs:
        float_start_idx = param_defs['float'][0]['internal_id']
        length = len(param_defs['float'])

        for i in range(len(float_value)):
            def_min = param_defs['float'][i]['min']
            def_max = param_defs['float'][i]['max']
            float_value[i] = def_min + float_value[i] * (def_max - def_min)

        target_node.setParmFloatValues(float_value, float_start_idx, length)

    if 'bool' in param_defs:
        for i in range(len(bool_value)):
            param_name = param_defs['bool'][i]['label']
            target_node.setParmBoolValue(param_name, bool_value[i])

    if 'choice' in param_defs:
        for i in range(len(choice_value)):
            param_name = param_defs['choice'][i]['label']
            choice_list = param_defs['choice'][i]['choices']
            choice_idx = choice_list.index(choice_value[i])
            target_node.setParmIntValue(param_name, choice_idx)

    vertices, faces = target_node.readGeometry()

    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
    return vertices, faces

def voxelize_mesh(mesh: trimesh.Trimesh, visualize=False):
    rsl = 64
    pitch = 1.0 / rsl

    extents = mesh.bounding_box.extents
    scale = 1.0 / np.max(extents)
    mesh.apply_scale(scale)

    voxels = trimesh.voxel.creation.local_voxelize(mesh, (0, 0, 0), pitch=pitch, radius=rsl, fill=True)

    if visualize:
        voxels.show()

    voxel_arr = np.zeros((rsl, rsl, rsl), dtype=np.uint8)
    if len(voxels.sparse_indices) > 0:
        indices = np.stack(voxels.sparse_indices)
        indices = np.clip(indices, 0, rsl - 1)
        voxel_arr[tuple(indices.T)] = 1

    return torch.tensor(voxel_arr, dtype=torch.float32)


def predict_param_from_voxel_direct(shape):
    init_houdini(shape)
    read_param_def(shape)
    pvd = get_param_vector_def()
    vectors = pvd.get_random_vectors(5)
    vector = vectors[3]
    vertices, faces = generate_model(vector)
    before_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    voxels = []
    voxel = voxelize_mesh(before_mesh, visualize=False)
    voxels.append(torch.unsqueeze(voxel, dim=0))
    voxels = torch.stack(voxels, dim=0)
    model = NNProc(shape)

    print('Predicting parameters from voxels.')
    model.load_state_dict(torch.load(os.path.join('../new_models', '{}_model.pt'.format(shape))))
    model.eval()
    param_preds_origin = model.param_dec.predict(model.voxel_enc.predict(voxels))
    param_preds = pvd.decode(param_preds_origin)

    vertices, faces = generate_model(param_preds[0])
    after_mesh = trimesh.Trimesh(vertices=vertices, faces=faces).apply_translation([1.5, 0, 0])
    print('before', vector)
    print('after', param_preds[0])
    (before_mesh + after_mesh).show()

def main():
    shape = 'table'
    predict_param_from_voxel_direct(shape)

if __name__ == "__main__":
    main()