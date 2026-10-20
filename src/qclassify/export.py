"""Direct ONNX graph export: no heavyweight training framework required."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qclassify.features import FEATURE_VERSION, INPUT_DIM

SIZE_LIMIT = 100_000  # Decimal KB, stricter than 100 KiB.


def export_onnx(params: dict, path: Path) -> dict:
    nodes = [
        helper.make_node("Gemm", ["features", "w1", "b1"], ["pre_hidden"]),
        helper.make_node("Relu", ["pre_hidden"], ["hidden"]),
        helper.make_node("Gemm", ["hidden", "w2", "b2"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["person_probability"]),
    ]
    graph = helper.make_graph(nodes, "qclassify", [
        helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, INPUT_DIM]),
    ], [
        helper.make_tensor_value_info("person_probability", TensorProto.FLOAT, [None, 1]),
    ], [numpy_helper.from_array(v, k) for k, v in params.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="qclassify", ir_version=9)
    helper.set_model_props(model, {"feature_version": FEATURE_VERSION,
                                  "labels": "0=search_query,1=person_name"})
    onnx.checker.check_model(model)
    serialized = model.SerializeToString()
    if len(serialized) >= SIZE_LIMIT:
        raise ValueError(f"ONNX is {len(serialized)} bytes; must be < {SIZE_LIMIT}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialized)
    return {"onnx_bytes": len(serialized), "weight_bytes": sum(v.nbytes for v in params.values()),
            "parameters": sum(v.size for v in params.values()), "limit_bytes": SIZE_LIMIT}
