from proc_shape import paramvectordef
import json, os
import he_init
from he_manager import he_instance
import numpy as np
import h5py
import config_data
from tqdm import tqdm
import trimesh
import utils

param_defs = {}
node_id = 0

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

    target_node = he_instance.getNode(node_id)

    if 'float' in param_defs:
        float_start_idx = param_defs['float'][0]['float_value_index']
        length = len(param_defs['float'])

        for i in range(len(float_value)):
            def_min = param_defs['float'][i]['min']
            def_max = param_defs['float'][i]['max']
            float_value[i] = def_min + float_value[i] * (def_max - def_min)

        target_node.setParmFloatValues(float_value, float_start_idx, length)

    if 'bool' in param_defs:
        for i in range(len(bool_value)):
            target_node.setParmBoolValue(i, bool_value[i])

    if 'choice' in param_defs:
        for i in range(len(choice_value)):
            int_start_idx = param_defs['choice'][i]['int_value_index']
            choice_list = param_defs['choice'][i]['choices']
            choice_idx = choice_list.index(choice_value[i])
            target_node.setParmIntValues([choice_idx], int_start_idx, 1)

    vertices, faces = target_node.readGeometry()

    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
    return vertices, faces

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

    global node_id
    node_id = he_instance.createAndCookNode(asset_name, 0)

def write_data(amount, f, mode, pvd):
    param_vector = pvd.get_random_vectors(amount)
    encoded = pvd.encode(param_vector)
    subgroup = f.create_group(mode)
    mesh_group = subgroup.create_group('msh')
    for i in tqdm(range(amount)):  # 各モデルのメッシュを生成して保存
        vertices, faces = generate_model(param_vector[i])
        mesh_sub_group = mesh_group.create_group(i.__str__())
        mesh_sub_group.create_dataset('v', data=vertices)
        mesh_sub_group.create_dataset('f', data=faces)
    prm_group = subgroup.create_group('prm')
    for i in range(len(encoded)):
        prm_group.create_dataset(i.__str__(), data=encoded[i])

def generate_hdf5(shape_type, train_amount, test_amount):
    init_houdini(shape_type)
    read_param_def(shape_type)
    pvd = get_param_vector_def()

    with h5py.File('../dataset/table_example.hdf5', 'w') as f:
        write_data(train_amount, f, config_data.TrainingMode.train.value, pvd)
        write_data(test_amount, f, config_data.TrainingMode.test.value, pvd)

def main():
    generate_hdf5('table', 4096, 128)

def test():
    init_houdini('table')
    read_param_def('table')
    pvd = get_param_vector_def()
    vectors = pvd.get_random_vectors(1)
    print(vectors)
    vertices, faces = generate_model(vectors[0])
    utils.voxelize_mesh(trimesh.Trimesh(vertices=vertices, faces=faces), visualize=True)

if __name__ == '__main__':
    main()