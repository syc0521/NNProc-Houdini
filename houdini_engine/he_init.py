import os, sys
hfs = os.environ.get('HFS')
path = os.path.join(hfs, 'houdini\python3.11libs')
os.add_dll_directory(os.path.join(hfs, 'bin'))
os.add_dll_directory(os.path.join(hfs, 'houdini\python3.11libs'))
sys.path.append(path)

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