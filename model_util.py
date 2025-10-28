import bpy
import bmesh
import mathutils

def get_transform(loc, scl):
    mat_loc = mathutils.Matrix.Translation(loc)
    mat_scx = mathutils.Matrix.Scale(scl[0], 4, mathutils.Vector((1, 0, 0)))
    mat_scy = mathutils.Matrix.Scale(scl[1], 4, mathutils.Vector((0, 1, 0)))
    mat_scz = mathutils.Matrix.Scale(scl[2], 4, mathutils.Vector((0, 0, 1)))
    mat = mat_loc @ mat_scx @ mat_scy @ mat_scz
    return mat

def cube(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=get_transform(loc, scl))
    return bm

def cube_new(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0)):
    import open3d as o3d
    cube = o3d.geometry.TriangleMesh.create_box(width=scl[0], height=scl[1], depth=scl[2], create_uv_map=True)
    return cube.translate(loc)

def visualize_omesh(meshes):
    import open3d as o3d
    o3d.visualization.draw_geometries(meshes)

def cylinder(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0), seg=30):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=0.5, radius2=0.5, depth=1.0, matrix=get_transform(loc, scl))
    return bm

def merge_bmeshes(bmeshes):
    bm_final = bmesh.new()
    for bm in bmeshes:
        offset = len(bm_final.verts)
        for v in bm.verts:
            bm_final.verts.new(v.co)
        bm_final.verts.index_update()
        bm_final.verts.ensure_lookup_table()
        if bm.faces:
            for face in bm.faces:
                bm_final.faces.new(tuple(bm_final.verts[i.index+offset] for i in face.verts))
            bm_final.faces.index_update()
        if bm.edges:
            for edge in bm.edges:
                try:
                    bm_final.edges.new(tuple(bm_final.verts[i.index+offset] for i in edge.verts))
                except ValueError:
                    pass
            bm_final.edges.index_update()
    bm_final.normal_update()
    return bm_final