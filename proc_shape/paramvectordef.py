import random
import numpy as np
from enum import Enum

class ParamType(Enum):
    SCALAR = 1
    BINARY = 2
    INTEGER = 3
    TYPE = 4


class Parameter:
    def __init__(self, paramtype, data):
        self.paramtype = paramtype
        self.data = data

class ParamVectorDef:
    def __init__(self):
        self.params = []

    def append_param(self, paramtype, data):
        self.params.append(Parameter(paramtype, data))

    def get_random_vectors(self, length):
        param_vectors = []
        for i in range(length):
            vector = []
            for param in self.params:
                paramtype = param.paramtype
                if paramtype == ParamType.SCALAR:
                    for dim in param.data:
                        vector.append(dim[0] + random.random() * dim[1])
                elif paramtype == ParamType.BINARY:
                    vector.append(random.choice([True, False]))
                elif paramtype == ParamType.INTEGER:
                    vector.append(random.randint(param.data[0], param.data[1]))
                elif paramtype == ParamType.TYPE:
                    vector.append(random.choice(param.data))
            param_vectors.append(vector)
        return param_vectors

    def encode(self, vectors):
        encoded_vectors = []
        index = 0
        for param in self.params:
            paramtype = param.paramtype
            if paramtype == ParamType.SCALAR:
                col_length = len(param.data)
            else:
                col_length = 1
            if paramtype == ParamType.SCALAR or paramtype == ParamType.BINARY:
                data = np.array([x[index:index+col_length] for x in vectors], dtype=float)
                if paramtype == ParamType.SCALAR:
                    min_vals = np.array([x[0] for x in param.data], ndmin=2)
                    max_vals = np.array([x[-1] for x in param.data], ndmin=2)
                    data = (data - min_vals) / (max_vals - min_vals)
                encoded_vectors.append(data)
            elif paramtype == ParamType.INTEGER:
                ohe_coded = np.array(
                    [[0 if i != (x[index] - param.data[0]) else 1 for i in range(param.data[1] - param.data[0] + 1)] for
                     x in vectors]
                )
                encoded_vectors.append(ohe_coded)
            elif paramtype == ParamType.TYPE:
                ohe_coded = np.array(
                    [[0 if x[index] != y else 1 for y in param.data] for x in vectors], dtype=float
                )
                encoded_vectors.append(ohe_coded)
            index = index + col_length
        return encoded_vectors

    def decode(self, vectors):
        collector = []
        for i, param in enumerate(self.params):
            paramtype = param.paramtype
            data = vectors[i]
            if paramtype == ParamType.SCALAR:
                min_vals = np.array([x[0] for x in param.data], ndmin=2)
                max_vals = np.array([x[-1] for x in param.data], ndmin=2)
                data = data * (max_vals - min_vals) + min_vals
            elif paramtype == ParamType.BINARY:
                data = np.isclose(data, 1.0)
            elif paramtype == ParamType.INTEGER:
                data = np.expand_dims(param.data[0]+np.argmax(data, axis=1), axis=1)
            elif paramtype == ParamType.TYPE:
                data = np.expand_dims(np.array(param.data)[np.argmax(data, axis=1)], axis=1)
            collector.append(data.tolist())
        decoded_vectors = []
        for i in range(len(collector[0])):
            vector = []
            for j in range(len(self.params)):
                vector.extend(collector[j][i])
            decoded_vectors.append(vector)
        return decoded_vectors

