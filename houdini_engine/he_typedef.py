from enum import Enum
class SessionType(Enum):
    InProcess = 1
    NewNamedPipe = 2
    NewTCPSocket = 3
    ExistingNamedPipe = 4
    ExistingTCPSocket = 5
    ExistingSharedMemory = 6

class ParamType(Enum):
    Int = 0
    MultiParmInt = 1
    Toggle = 2
    Button = 3
    Float = 4
    Color = 5
    String = 6
    PathFile = 7
    PathFileGeo = 8
    PathFileImage = 9
    Node = 10
    FolderList = 11
    FolderListRadio = 12
    Folder = 13
    Label = 14
    Separator = 15
    PathFileDir = 16
    Max = 17

class ChoiceListType(Enum):
    Null = 0
    Normal = 1
    Mini = 2
    Replace = 3
    Toggle = 4