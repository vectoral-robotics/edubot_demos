#!/usr/bin/env python3
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def convert_type(type_proto):
    if type_proto.HasField("tensor_type"):
        tensor_type = type_proto.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT16:
            tensor_type.elem_type = TensorProto.FLOAT


def convert_tensor(tensor):
    if tensor.data_type != TensorProto.FLOAT16:
        return tensor

    array = numpy_helper.to_array(tensor).astype(np.float32)
    return numpy_helper.from_array(array, tensor.name)


def convert_graph(graph):
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        convert_type(value_info.type)

    for index, initializer in enumerate(graph.initializer):
        graph.initializer[index].CopyFrom(convert_tensor(initializer))

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                attr.t.CopyFrom(convert_tensor(attr.t))
            elif attr.type == onnx.AttributeProto.GRAPH:
                convert_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    convert_graph(subgraph)
            elif attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT


def main():
    if len(sys.argv) != 3:
        print("usage: onnx_fp16_to_fp32.py input.onnx output.onnx", file=sys.stderr)
        return 2

    model = onnx.load(sys.argv[1])
    convert_graph(model.graph)
    onnx.checker.check_model(model)
    onnx.save(model, sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
