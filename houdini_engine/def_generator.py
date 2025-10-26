import he_init
import json, os
from he_manager import he_instance

def generate_param_def(folder, name):
    session = he_init.create_session()
    if session is None:
        print("Failed to create the Houdini Engine session.")
        return

    hda_loaded, asset_name = he_instance.loadAsset(os.path.join(folder, name) + ".hda")
    if not hda_loaded:
        print("Failed to load the HDA.")
        return

    target_node = he_instance.getNode(0)
    he_instance.createAndCookNode(asset_name, target_node.node_id)

    all_params = target_node.getAllParameterInfo()
    folder_param = all_params[1]
    detail_data = [he_instance.getParamDetailData(p) for p in all_params if p.parentId == folder_param.id]
    float_params = [p for p in detail_data if p['param_type'] == 4]
    bool_params = [p for p in detail_data if p['param_type'] == 2]
    choice_params = [p for p in detail_data if 'choices' in p]
    out_params = {
        'float': float_params,
        'bool': bool_params,
        'choice': choice_params
    }
    out_path = os.path.join(folder, name) + "_param_def.json"
    with open(out_path, "w") as f:
        json.dump(out_params, f, indent=4)
        f.close()
    print('Parameter definition saved to', out_path)


def main():
    hda_folder = "../hdas"
    hda_name = "primitive"
    generate_param_def(hda_folder, hda_name)

if __name__ == '__main__':
    main()