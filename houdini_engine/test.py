
if __name__ == '__main__':
    import os, sys

    hfs = os.environ.get('HFS')
    path = os.path.join(hfs, 'houdini\python3.11libs')
    os.add_dll_directory(os.path.join(hfs, 'bin'))
    os.add_dll_directory(os.path.join(hfs, 'houdini\python3.11libs'))
    sys.path.append(path)
    print(path)

    import he_sample

    he_sample.main()
