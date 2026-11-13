import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import pytest

from qclassify.export import SIZE_LIMIT, export_onnx, load_params
from qclassify.features import (GLOBAL_DIM, LEX_BINS, LEXICON_PATH, MAX_WORDS, WORD_DIM, canonicalize, decode_lexicon,
                                featurize, transform)
from qclassify.model import fit, logits, sigmoid
from qclassify.packing import blob_bytes, pack, pack_matrix, unpack, unpack_matrix
from qclassify.train import rows_for
from qclassify.predict import Classifier
from qclassify.train import load_dataset, split_indices

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_is_literal_balanced_and_unique():
    rows = load_dataset(ROOT / 'data/examples.json')
    assert len(rows) == 24271
    for label, source in [('person_name', 'names.txt'), ('search_query', 'queries.txt')]:
        actual = [r['input'] for r in rows if r['output'] == label]
        assert len(actual) == (14508 if label == "person_name" else 9763)
        assert actual == (ROOT / 'data' / source).read_text().splitlines()
    parts = split_indices(rows, 42)
    assert [len(part) for part in parts] == [19416, 2427, 2428]
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
    words, mask, glob = transform([])
    assert words.shape == (0, MAX_WORDS, WORD_DIM) and glob.shape == (0, GLOBAL_DIM)
    for text in ['😀' * 300, 'long ' * 100]:
        w, g = featurize(text)
        assert 1 <= len(w) <= MAX_WORDS and np.isfinite(w).all() and np.isfinite(g).all()
    for a, b in [('A' * 256, 'A' * 300)]:
        assert all(np.array_equal(x, y) for x, y in zip(featurize(a), featurize(b)))
    assert all(np.array_equal(x, y) for x, y in zip(featurize('Solar Energy'), featurize('solar energy')))


def test_lexicon_is_well_formed():
    table = decode_lexicon(LEXICON_PATH.read_bytes())
    assert table.shape == (LEX_BINS,) and table.min() >= -1 and table.max() <= 1
    assert (table != 0).mean() > 0.9


def test_duplicate_data_is_rejected(tmp_path):
    path = tmp_path / 'examples.json'
    path.write_text(json.dumps([{'input': 'Alice Smith', 'output': 'person_name'},
                                {'input': '  ALICE  SMITH ', 'output': 'search_query'}]))
    with pytest.raises(ValueError, match='Duplicate'):
        load_dataset(path)


def test_real_training_learns_a_separable_problem():
    texts = [f"alice{i} smith" for i in range(40)] + [f"how to build {i} apps" for i in range(40)]
    y = np.array([1] * 40 + [0] * 40, np.float32)
    rows = rows_for(texts)
    params, info = fit(rows, y, rows, y, hidden=4, global_hidden=2, seed=8, epochs=60, learning_rate=0.03)
    words, mask, glob = transform(texts)
    assert np.mean((sigmoid(logits(params, words, mask, glob)) >= 0.5) == y) > 0.97
    assert info['best_epoch'] > 0


def test_original_onnx_size_and_numerical_equivalence():
    path = ROOT / 'artifacts/classifier.onnx'
    assert path.stat().st_size < SIZE_LIMIT
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    props = {p.key: p.value for p in model.metadata_props}
    assert props['lexicon_sha256'] == hashlib.sha256(LEXICON_PATH.read_bytes()).hexdigest()
    params = load_params(path)
    rows = load_dataset(ROOT / 'data/examples.json')
    texts = [r['input'] for r in rows][::7]
    words, mask, glob = transform(texts)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None, {'words': words, 'word_mask': mask, 'global': glob})[0].reshape(-1),
                               sigmoid(logits(params, words, mask, glob)), atol=1e-5, rtol=1e-4)
    assert Classifier(path).predict([]) == []
    assert len(Classifier(path).predict(['Jane Doe', 'browser inference projects'])) == 2
    metrics = json.loads((ROOT / 'artifacts/metrics.json').read_text())
    assert metrics['dataset_sha256'] == hashlib.sha256((ROOT / 'data/examples.json').read_bytes()).hexdigest()


def test_packed_weights_fit_the_budget_and_roundtrip():
    manifest = json.loads((ROOT / 'artifacts/web/model.json').read_text())
    blob = (ROOT / 'artifacts/web/weights.bin').read_bytes()
    assert len(blob) == manifest['weightBytes'] <= 25_000
    assert blob[:LEX_BINS // 2] == LEXICON_PATH.read_bytes()
    params, _ = unpack(blob, manifest['hiddenDim'], manifest['globalHiddenDim'], manifest['bits'])
    assert pack(params, manifest['bits'], LEXICON_PATH.read_bytes()) == blob


@pytest.mark.parametrize('bits', [2, 3, 4, 5, 6, 7, 8])
def test_pack_roundtrip_with_non_aligned_dimensions(bits):
    rng = np.random.default_rng(15)
    flat = rng.uniform(-0.3, 0.3, 37 * 3).astype(np.float32)
    blob = pack_matrix(flat, bits)
    actual = unpack_matrix(blob, 0, len(flat), bits)
    levels = (1 << (bits - 1)) - 1
    assert np.max(np.abs(actual - flat)) <= 0.3 / levels / 2 + 0.001
    assert blob_bytes(13, 4, bits) > 0


def test_export_rejects_oversized_weights(tmp_path):
    h, g = 32, 4
    params = {'w1': np.zeros((WORD_DIM, h), np.float32), 'b1': np.zeros(h, np.float32), 'w2': np.zeros(h, np.float32),
              'b2': np.zeros(1, np.float32), 'mix': np.zeros(2, np.float32), 'w3': np.zeros((GLOBAL_DIM, g), np.float32),
              'b3': np.zeros(g, np.float32), 'w4': np.zeros(g, np.float32), 'vs': np.zeros(20, np.float32),
              'c': np.zeros(1, np.float32)}
    with pytest.raises(ValueError, match='must be <'):
        export_onnx(params, tmp_path / 'too_big.onnx')
    assert not (tmp_path / 'too_big.onnx').exists()
