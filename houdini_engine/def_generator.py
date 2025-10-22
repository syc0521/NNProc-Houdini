import he_init
import json, os
from he_manager import he_instance, SessionType

def create_session():
    session_type = SessionType.InProcess.value
    named_pipe = he_instance.DEFAULT_NAMED_PIPE
    tcp_port = he_instance.DEFAULT_TCP_PORT
    use_cooking_thread = True

    session = he_instance.startSession(session_type, named_pipe, tcp_port)
    he_instance.initializeHAPI(use_cooking_thread)
    return session

def load_hda(hda_path):
    hda_loaded, asset_name = he_instance.loadAsset(hda_path)
    return asset_name

def generate_param_def(folder, name):
    session = create_session()
    if session is None:
        print("Failed to create the Houdini Engine session.")
        return

    hda_node_id = 0
    hda_loaded, asset_name = he_instance.loadAsset(os.path.join(folder, name) + ".hda")
    if not hda_loaded:
        print("Failed to load the HDA.")
        return

    hda_cooked = he_instance.createAndCookNode(asset_name, hda_node_id)
    if not hda_cooked:
        print("Failed to create and cook the HDA node.")
        return

    all_params = he_instance.getAllParameterInfo(hda_node_id)
    folder_param = all_params[1]
    output_params = [he_instance.getParamDetailData(p) for p in all_params if p.parentId == folder_param.id]
    out_path = os.path.join(folder, name) + "_param_def.json"
    with open(out_path, "w") as f:
        json.dump(output_params, f, indent=4)
        f.close()
    print('Parameter definition saved to', out_path)


def main():
    hda_folder = "../hdas"
    hda_name = "table"
    generate_param_def(hda_folder, hda_name)

if __name__ == '__main__':
    main()