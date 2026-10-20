import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import pytest

from qclassify.export import SIZE_LIMIT, export_onnx
from qclassify.features import INPUT_DIM, canonicalize, featurize, transform
from qclassify.model import fit, forward, sigmoid
from qclassify.packing import pack, unpack
from qclassify.predict import Classifier
from qclassify.train import load_dataset, split_indices

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_is_literal_balanced_and_unique():
    rows = load_dataset(ROOT / 'data/examples.json')
    assert len(rows) == 5226
    for label, source in [('person_name', 'names.txt'), ('search_query', 'queries.txt')]:
        actual = [r['input'] for r in rows if r['output'] == label]
        assert len(actual) == (2550 if label == "person_name" else 2676)
        assert actual == (ROOT / 'data' / source).read_text().splitlines()
    parts = split_indices(rows, 42)
    assert [len(part) for part in parts] == [4180, 523, 523]
    assert len(set(np.concatenate(parts))) == len(rows)
    keys = [{canonicalize(rows[i]['input']) for i in part} for part in parts]
    assert not (keys[0] & keys[1] or keys[1] & keys[2] or keys[0] & keys[2])
    # Reordering the authored examples cannot move inputs between partitions.
    reverse = rows[::-1]
    assert keys == [{canonicalize(reverse[i]['input']) for i in p} for p in split_indices(reverse, 42)]


def test_input_contract():
    for text in ['', ' ', '\n\t', '\u0085']:
        with pytest.raises(ValueError):
            featurize(text)
    assert transform([]).shape == (0, INPUT_DIM)
    assert np.isfinite(featurize('😀' * 300)).all()
    assert np.array_equal(featurize('A' * 256), featurize('A' * 300))


def test_duplicate_data_is_rejected(tmp_path):
    path = tmp_path / 'examples.json'
    path.write_text(json.dumps([{'input': 'Alice Smith', 'output': 'person_name'},
                                {'input': '  ALICE  SMITH ', 'output': 'search_query'}]))
    with pytest.raises(ValueError, match='Duplicate'):
        load_dataset(path)


def test_real_training_learns_a_separable_problem():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((100, 4)).astype(np.float32)
    y = (x[:, :1] > 0).astype(np.float32)
    params, info = fit(x, y, x, y, hidden=4, seed=8, epochs=80, device='cpu', learning_rate=0.03)
    assert np.mean((sigmoid(forward(params, x)[1]) >= 0.5) == y) > 0.97
    assert info['best_epoch'] > 0


def test_original_onnx_size_and_numerical_equivalence():
    path = ROOT / 'artifacts/classifier.onnx'
    assert path.stat().st_size < SIZE_LIMIT
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    params = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    rows = load_dataset(ROOT / 'data/examples.json')
    x = transform([r['input'] for r in rows])
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None, {'features': x})[0], sigmoid(forward(params, x)[1]),
                               atol=1e-6, rtol=1e-5)
    assert Classifier(path).predict([]) == []
    assert len(Classifier(path).predict(['Jane Doe', 'browser inference projects'])) == 2
    metrics = json.loads((ROOT / 'artifacts/metrics.json').read_text())
    assert metrics['dataset_sha256'] == hashlib.sha256((ROOT / 'data/examples.json').read_bytes()).hexdigest()
    assert metrics['training']['device'] == 'mps'


@pytest.mark.parametrize('bits', [2, 3, 4, 5, 6, 7, 8])
def test_pack_roundtrip_with_non_aligned_dimensions(bits):
    rng = np.random.default_rng(15)
    params = {'w1': rng.uniform(-0.3, 0.3, (37, 3)).astype(np.float32),
              'b1': np.array([0.25, -0.25, 0], np.float32),
              'w2': np.array([[0.125], [-0.25], [1]], np.float32),
              'b2': np.array([0.5], np.float32)}
    blob = pack(params, bits)
    actual = unpack(blob, 37, 3, bits)
    levels = (1 << (bits - 1)) - 1
    assert np.max(np.abs(actual['w1'] - params['w1'])) <= 0.3 / levels / 2 + 0.001
    for key in ['b1', 'w2', 'b2']:
        np.testing.assert_array_equal(actual[key], params[key])


def test_export_rejects_oversized_weights(tmp_path):
    params = {'w1': np.zeros((INPUT_DIM, 32), np.float32), 'b1': np.zeros(32, np.float32),
              'w2': np.zeros((32, 1), np.float32), 'b2': np.zeros(1, np.float32)}
    with pytest.raises(ValueError, match='must be <'):
        export_onnx(params, tmp_path / 'too_big.onnx')
    assert not (tmp_path / 'too_big.onnx').exists()
