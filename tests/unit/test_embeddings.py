import subprocess
import sys

import numpy as np
import pytest

from sentineledu.config import PROJECT_ROOT
from sentineledu.services import embeddings

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_import_does_not_load_model():
    code = ("import sys; import sentineledu.services.embeddings as e; "
            "print('sentence_transformers' in sys.modules, e.load_count)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(PROJECT_ROOT)).stdout.strip()
    assert out == "False 0"


def test_model_loads_once(offline_settings):
    embeddings.encode(["warm up"], offline_settings)
    before = embeddings.load_count
    embeddings.encode(["again"], offline_settings)
    embeddings.encode("and again", offline_settings)
    embeddings.dimension(offline_settings)
    assert embeddings.load_count == before
    assert before >= 1


def test_vectors_are_normalised(offline_settings):
    v = embeddings.encode(["Plants make food.", "Water is wet."], offline_settings)
    assert v.shape == (2, embeddings.dimension(offline_settings))
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_cosine_identical_is_one(offline_settings):
    v = embeddings.encode("Photosynthesis makes glucose.", offline_settings)[0]
    assert embeddings.cosine(v, v) == pytest.approx(1.0, abs=1e-5)
    assert embeddings.similarity("Photosynthesis makes glucose.", "Photosynthesis makes glucose.",
                                 offline_settings) == pytest.approx(1.0, abs=1e-5)


def test_related_text_scores_higher_than_unrelated(offline_settings):
    near = embeddings.similarity("Plants make food using sunlight.", "Photosynthesis is how plants make food.", offline_settings)
    far = embeddings.similarity("Plants make food using sunlight.", "The train leaves at nine o'clock.", offline_settings)
    assert near > far


def test_blank_and_zero_vectors():
    assert embeddings.cosine(np.zeros(3), np.ones(3)) == 0.0
    assert embeddings.similarity("", "anything") == 0.0
    assert embeddings.similarity("   ", "anything") == 0.0


def test_empty_batch(offline_settings):
    assert embeddings.encode([], offline_settings).shape == (0, embeddings.dimension(offline_settings))
