import hapi
import he_utility

# Houdini Node C++ API
class HoudiniNode:
    def __init__(self, node_id, session):
        self.node_id = node_id
        self.session = session
        self.cook_options = None

    def getParmFloatValues(self, start, length):
        return hapi.getParmFloatValues(self.session, self.node_id, start, length)

    def setParmFloatValues(self, values_array, start, length):
        return hapi.setParmFloatValues(self.session, self.node_id, values_array, start, length)

    def setParmBoolValue(self, name, value):
        return hapi.setParmIntValue(self.session, self.node_id, name, 0, 1 if value == True else 0)

    def setParmIntValue(self, name, value):
        return hapi.setParmIntValue(self.session, self.node_id, name, 0, 1 if value == True else 0)

    def getAttributeNames(self, part_id, owner, count):
        part_info = hapi.getPartInfo(self.session, self.node_id, part_id)
        return hapi.getAttributeNames(self.session, self.node_id, part_id, owner, count)

    def getAttributeInfo(self, part_id, attr_name):
        return hapi.getAttributeInfo(self.session, self.node_id, part_id, attr_name, 0)

    def readGeometry(self):
        hapi.cookNode(self.session, self.node_id, None)

        # Check the cook status
        status = hapi.getStatus(self.session, hapi.statusType.CookState)
        while (status > hapi.state.Ready):
            status = hapi.getStatus(self.session, hapi.statusType.CookState)

        # Get mesh geo info.
        print("\nGetting mesh geometry info:")
        mesh_geo_info = hapi.getDisplayGeoInfo(self.session, self.node_id)

        # Get mesh part info.
        mesh_part_info = hapi.getPartInfo(self.session, mesh_geo_info.nodeId, 0)

        # Get mesh face counts.
        mesh_face_counts = hapi.getFaceCounts(
            self.session,
            mesh_geo_info.nodeId,
            mesh_part_info.id,
            0, mesh_part_info.faceCount
        )
        print("  Face count: {}".format(len(mesh_face_counts)))

        # Get mesh vertex list.
        mesh_vertex_list = hapi.getVertexList(
            self.session,
            mesh_geo_info.nodeId,
            mesh_part_info.id,
            0, mesh_part_info.vertexCount
        )

        print("  Vertex count: {}".format(len(mesh_vertex_list)))

        def _fetchPointAttrib(owner, attrib_name):
            mesh_attrib_info = hapi.getAttributeInfo(
                self.session,
                mesh_geo_info.nodeId,
                mesh_part_info.id,
                attrib_name, owner
            )

            mesh_attrib_data = hapi.getAttributeFloatData(
                self.session,
                mesh_geo_info.nodeId,
                mesh_part_info.id,
                attrib_name,
                mesh_attrib_info, -1,
                0, mesh_attrib_info.count
            )

            print("  {} attribute count: {}".format(
                attrib_name, len(mesh_attrib_data)))
            return mesh_attrib_data

        mesh_p_attrib_info = _fetchPointAttrib(hapi.attributeOwner.Point, "P")

        return mesh_p_attrib_info, mesh_vertex_list

    def getAllParameterInfo(self):
        node_info = hapi.getNodeInfo(self.session, self.node_id)

        parm_infos = hapi.getParameters(self.session, self.node_id, 0, node_info.parmCount)
        return parm_infos

    def getParameters(self):
        '''Query and list the paramters of the given node'''
        node_info = hapi.getNodeInfo(self.session, self.node_id)
        parm_infos = hapi.getParameters(self.session, self.node_id, 0, node_info.parmCount)

        print("\nParameters: ")
        print("==========")
        for i in range(node_info.parmCount - 1):
            print("  Name: ", end='')
            print(he_utility.getString(self.session, parm_infos[i].nameSH))
            print("  Label: ", end='')
            print(he_utility.getString(self.session, parm_infos[i].labelSH))
            print("  Values: (", end='')

            if parm_infos[i].type == hapi.parmType.Int:
                parm_int_count = parm_infos[i].size

                parm_int_values = hapi.getParmIntValues(
                    self.session,
                    self.node_id,
                    parm_infos[i].intValuesIndex,
                    parm_int_count
                )

                for v in range(parm_int_count):
                    if v != 0:
                        print(", ", end='')
                    print(parm_int_values[v], end='')
            elif parm_infos[i].type == hapi.parmType.Float:
                parm_float_count = parm_infos[i].size

                parm_float_values = hapi.getParmFloatValues(
                    self.session,
                    self.node_id,
                    parm_infos[i].floatValuesIndex,
                    parm_float_count
                )

                for v in range(parm_float_count):
                    if v != 0:
                        print(", ", end='')
                    print(parm_float_values[v], end='')
            elif parm_infos[i].type == hapi.parmType.String:
                parm_string_count = parm_infos[i].size

                parmSH_values = hapi.getParmStringValues(
                    self.session,
                    self.node_id,
                    True,
                    parm_infos[i].stringValuesIndex,
                    parm_string_count
                )

                for v in range(parm_string_count):
                    if v != 0:
                        print(", ", end='')
                    print(he_utility.getString(
                        self.session, parmSH_values[v]), end='')
            print(")")

        return True
