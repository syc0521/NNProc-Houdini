from proc_shape import paramvectordef
import json, os
import pyvista as pv
import trimesh
import he_init
from he_manager import he_instance
import numpy as np

param_defs = {}

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

def generate_model(params, asset_name):
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

    if 'float' in param_defs:
        float_start_idx = param_defs['float'][0]['internal_id']
        length = len(param_defs['float'])

        for i in range(len(float_value)):
            def_min = param_defs['float'][i]['min']
            def_max = param_defs['float'][i]['max']
            float_value[i] = def_min + float_value[i] * (def_max - def_min)

        he_instance.setParmFloatValues(0, float_value, float_start_idx, length)

    if 'bool' in param_defs:
        for i in range(len(bool_value)):
            param_name = param_defs['bool'][i]['label']
            he_instance.setParmBoolValue(0, param_name, bool_value[i])

    hda_cooked = he_instance.createAndCookNode(asset_name, 0)
    if not hda_cooked:
        print("Failed to create and cook the HDA node.")
        return

    points, vertexes = he_instance.readGeometry(0)

    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    vertexes = np.array(vertexes, dtype=np.int32).reshape(-1, 3)
    # print("Generated model with {} points and {} vertexes.".format(len(points), len(vertexes) * 3))
    #
    # # mesh faces
    # faces = np.hstack(vertexes)
    # surf = pv.PolyData(points, faces)
    #
    # # plot each face with a different color
    # surf.plot(
    #     scalars=np.arange(3),
    #     cpos=[-1, 1, 0.5],
    #     show_scalar_bar=False,
    #     show_edges=True,
    #     line_width=5,
    # )

    trimesh.Trimesh(points=points, faces=vertexes).show()

def main():
    read_param_def("table")
    pvd = get_param_vector_def()
    rand_params = pvd.get_random_vectors(1)

    session = he_init.create_session()
    if session is None:
        print("Failed to create the Houdini Engine session.")
        return

    hda_folder = "../hdas"
    hda_name = "table"
    hda_loaded, asset_name = he_instance.loadAsset(os.path.join(hda_folder, hda_name) + ".hda")
    if not hda_loaded:
        print("Failed to load the HDA.")
        return

    for param in rand_params:
        generate_model(param, asset_name)

    print(rand_params)


if __name__ == '__main__':
    main()