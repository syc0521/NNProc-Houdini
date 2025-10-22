from he_typedef import ParamType
from proc_shape import paramvectordef
import json

param_defs = []

def read_param_def(name):
    with open("../hdas/{}_param_def.json".format(name), "r") as f:
        global param_defs
        param_defs = json.load(f)
        f.close()

def get_param_vector_def():
    scalar_params = []
    toggle_count = 0
    choice_params = []

    for param in param_defs:
        if param['param_type'] == ParamType.Float.value:
            scalar_params.append([0.0, 1.0])
        elif param['param_type'] == ParamType.Toggle.value:
            toggle_count = toggle_count + 1
        elif param['param_type'] == ParamType.Int.value and param['choices'] is not None:
            choice_params.append(param['choices'])

    pvd = paramvectordef.ParamVectorDef()
    if len(scalar_params) > 0:
        pvd.append_param(paramvectordef.ParamType.SCALAR, scalar_params)
    for i in range(toggle_count):
        pvd.append_param(paramvectordef.ParamType.BINARY, None)
    for choice in choice_params:
        pvd.append_param(paramvectordef.ParamType.TYPE, choice)

    return pvd

def generate_model(param):
    # todo
    pass

def main():
    read_param_def("table")
    pvd = get_param_vector_def()
    rand_params = pvd.get_random_vectors(5)
    for param in rand_params:
        generate_model(param)

    print(rand_params)


if __name__ == '__main__':
    main()