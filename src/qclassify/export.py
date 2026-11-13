"""Direct ONNX graph export of the per-word model: no heavyweight training framework required.

Inputs are padded: words [N, MAX_WORDS, WORD_DIM], word_mask [N, MAX_WORDS] (1 = real word), global [N, GLOBAL_DIM].
"""
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qclassify.features import FEATURE_VERSION, GLOBAL_BINS, GLOBAL_DIM, LEXICON_PATH, MAX_WORDS, WORD_DIM

SIZE_LIMIT = 134_000  # Decimal bytes: 2x the 67,336-byte 0.2.x model.


def export_onnx(params: dict, path: Path) -> dict:
    p = {k: np.asarray(v, np.float32) for k, v in params.items()}
    hidden = p["b1"].size
    consts = {
        "w1": p["w1"], "b1": p["b1"], "w2": p["w2"].reshape(hidden, 1), "b2": p["b2"].reshape(1),
        "mix0": p["mix"][:1], "mix1": p["mix"][1:], "w3": p["w3"], "b3": p["b3"],
        "w4": p["w4"].reshape(-1, 1), "vs": p["vs"].reshape(-1, 1), "c": p["c"].reshape(1),
        "lo": np.array(-30, np.float32), "hi": np.array(30, np.float32),
        "axis1": np.array([1], np.int64), "axis2": np.array([2], np.int64),
        "shape_start": np.array([GLOBAL_BINS], np.int64), "shape_end": np.array([GLOBAL_DIM], np.int64),
    }
    n = helper.make_node
    nodes = [
        n("MatMul", ["words", "w1"], ["h0"]), n("Add", ["h0", "b1"], ["h1"]), n("Relu", ["h1"], ["h"]),
        n("MatMul", ["h", "w2"], ["s0"]), n("Squeeze", ["s0", "axis2"], ["s1"]), n("Add", ["s1", "b2"], ["s2"]),
        n("Clip", ["s2", "lo", "hi"], ["s"]),
        n("Neg", ["s"], ["ns"]), n("Exp", ["ns"], ["e0"]), n("Mul", ["e0", "word_mask"], ["e"]),
        n("ReduceSum", ["e", "axis1"], ["esum"]), n("Log", ["esum"], ["lsum"]), n("Neg", ["lsum"], ["softmin"]),
        n("Mul", ["s", "word_mask"], ["sm"]), n("ReduceSum", ["sm", "axis1"], ["ssum"]),
        n("ReduceSum", ["word_mask", "axis1"], ["count"]), n("Div", ["ssum", "count"], ["mean"]),
        n("Mul", ["softmin", "mix0"], ["a"]), n("Mul", ["mean", "mix1"], ["b"]),
        n("MatMul", ["global", "w3"], ["g0"]), n("Add", ["g0", "b3"], ["g1"]), n("Relu", ["g1"], ["g2"]),
        n("MatMul", ["g2", "w4"], ["g"]),
        n("Slice", ["global", "shape_start", "shape_end", "axis1"], ["shape"]), n("MatMul", ["shape", "vs"], ["v"]),
        n("Add", ["a", "b"], ["t0"]), n("Add", ["t0", "g"], ["t1"]), n("Add", ["t1", "v"], ["t2"]),
        n("Add", ["t2", "c"], ["logits"]), n("Sigmoid", ["logits"], ["person_probability"]),
    ]
    graph = helper.make_graph(nodes, "qclassify", [
        helper.make_tensor_value_info("words", TensorProto.FLOAT, [None, MAX_WORDS, WORD_DIM]),
        helper.make_tensor_value_info("word_mask", TensorProto.FLOAT, [None, MAX_WORDS]),
        helper.make_tensor_value_info("global", TensorProto.FLOAT, [None, GLOBAL_DIM]),
    ], [helper.make_tensor_value_info("person_probability", TensorProto.FLOAT, [None, 1])],
        [numpy_helper.from_array(v, k) for k, v in consts.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="qclassify", ir_version=9)
    helper.set_model_props(model, {"feature_version": FEATURE_VERSION, "labels": "0=search_query,1=person_name",
                                   "lexicon_sha256": hashlib.sha256(LEXICON_PATH.read_bytes()).hexdigest()})
    onnx.checker.check_model(model)
    serialized = model.SerializeToString()
    if len(serialized) >= SIZE_LIMIT:
        raise ValueError(f"ONNX is {len(serialized)} bytes; must be < {SIZE_LIMIT}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialized)
    return {"onnx_bytes": len(serialized), "weight_bytes": sum(v.nbytes for v in p.values()),
            "parameters": sum(v.size for v in p.values()), "limit_bytes": SIZE_LIMIT}


def load_params(path: Path) -> dict:
    model = onnx.load(path)
    t = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    return {"w1": t["w1"], "b1": t["b1"], "w2": t["w2"].reshape(-1), "b2": t["b2"].reshape(1),
            "mix": np.concatenate([t["mix0"], t["mix1"]]), "w3": t["w3"], "b3": t["b3"],
            "w4": t["w4"].reshape(-1), "vs": t["vs"].reshape(-1), "c": t["c"].reshape(1)}
