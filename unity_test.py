from houdini_engine import he_init
from proc_shape import paramvectordef
import json, os
import torch
from model import NNProc
from houdini_engine.he_manager import he_instance
import numpy as np
import trimesh
import utils

param_defs = {}

def read_param_def(name):
    with open("hdas/{}_param_def.json".format(name), "r") as f:
        global param_defs
        param_defs = json.load(f)
        f.close()

def init_houdini(shape_type):
    session = he_init.create_session()
    if session is None:
        print("Failed to create the Houdini Engine session.")
        return False

    hda_folder = "hdas"
    hda_loaded, asset_name = he_instance.loadAsset(os.path.join(hda_folder, shape_type) + ".hda")
    if not hda_loaded:
        print("Failed to load the HDA.")
        return False

    he_instance.createAndCookNode(asset_name, 0)

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


def read_shape_info(shape):
    with open("unity_test/shape_info.json".format(shape), "r") as f:
        shape_info = json.load(f)
        f.close()
    return shape_info

def bmesh_to_trimesh(bm):
    verts = np.array([list(v.co) for v in bm.verts])
    verts = np.array([1, 1, -1]) * verts[:, [0, 2, 1]]
    faces = np.array([[l.vert.index for l in loop] for loop in bm.calc_loop_triangles()])
    tm = trimesh.Trimesh(vertices=verts, faces=faces)
    return tm

def normalize_mesh(mesh: trimesh.Trimesh):
    t = np.sum(mesh.bounding_box.bounds, axis=0) / 2
    mesh.apply_translation(-t)
    s = np.max(mesh.extents)
    mesh.apply_scale(1 / s)

def create_cube(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0)):
    mesh = trimesh.creation.box(scl).apply_translation(loc)
    return mesh

def create_cylinder(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0)):
    mesh = trimesh.load_mesh("meshes/cylinder.obj")
    mesh = mesh.apply_scale(scl).apply_translation(loc)
    return mesh

def visualize_omesh(meshes):
    import open3d as o3d
    o3d.visualization.draw_geometries(meshes)

def build_model(shape_info):
    meshes = []
    output = None
    for shape in shape_info:
        pos = (shape['Position']['x'], shape['Position']['y'], shape['Position']['z'])
        rot = (shape['Rotation']['x'], shape['Rotation']['y'], shape['Rotation']['z'])
        scale = (shape['Scale']['x'], shape['Scale']['y'], shape['Scale']['z'])
        shape_type = shape['Type']
        mesh = None
        if shape_type == 2:
            mesh = create_cylinder(pos, scale)
        elif shape_type == 0:
            mesh = create_cube(pos, scale)

        meshes.append(mesh)
        if output is None:
            output = mesh
        else:
            output += mesh

    output.show()
    return output

def predict_param_from_voxel_direct(shape):
    init_houdini(shape)
    read_param_def(shape)
    shape_info = read_shape_info(shape)
    before_mesh = build_model(shape_info)
    pvd = get_param_vector_def()

    voxels = []
    voxel = utils.voxelize_mesh_faster(before_mesh, visualize=False)
    voxels.append(torch.unsqueeze(voxel, dim=0))
    voxels = torch.stack(voxels, dim=0)
    model = NNProc(shape)

    print('Predicting parameters from voxels.')
    model.load_state_dict(torch.load(os.path.join('new_models', '{}_model.pt'.format(shape))))
    model.eval()
    param_preds_origin = model.param_dec.predict(model.voxel_enc.predict(voxels))
    param_preds = pvd.decode(param_preds_origin)
    print('after', param_preds[0])

    vertices, faces = generate_model(param_preds[0])
    after_mesh = trimesh.Trimesh(vertices=vertices, faces=faces).apply_translation([1.5, 0.5, 0.5])
    (before_mesh + after_mesh).show()

def main():
    shape = 'table'
    #load_save_mesh(shape)
    predict_param_from_voxel_direct(shape)

if __name__ == "__main__":
    main()