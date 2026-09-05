"""fusion 模块的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import (
    FusedFeatures,
    adaptive_weights,
    fuse_feature_groups,
    normalize_group,
    save_fusion_outputs,
)


class NormalizeGroupTest(unittest.TestCase):
    def test_minmax_expected_range(self) -> None:
        array = np.array([[0.0], [4.0], [8.0]], dtype=np.float32)
        normalized = normalize_group(
            array,
            {"normalize": "minmax", "expected_min": 0.0, "expected_max": 8.0},
        )
        np.testing.assert_allclose(normalized, [[0.0], [0.5], [1.0]])

    def test_scale(self) -> None:
        array = np.array([[0.0], [128.0], [255.0]], dtype=np.float32)
        normalized = normalize_group(array, {"normalize": "scale", "scale": 255.0})
        np.testing.assert_allclose(normalized, [[0.0], [128.0 / 255.0], [1.0]])


class AdaptiveWeightsTest(unittest.TestCase):
    def test_weights_are_positive_and_sum_to_one(self) -> None:
        arrays = {
            "a": np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32),
            "b": np.random.default_rng(1).normal(size=(20, 16)).astype(np.float32),
        }
        weights = adaptive_weights(arrays, {"method": "variance_softmax"})
        self.assertEqual(set(weights), {"a", "b"})
        self.assertTrue(all(value > 0 for value in weights.values()))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_uniform_weights(self) -> None:
        arrays = {"a": np.ones((10, 2), dtype=np.float32)}
        weights = adaptive_weights(arrays, {"method": "uniform"})
        self.assertEqual(weights, {"a": 1.0})


class FuseFeatureGroupsTest(unittest.TestCase):
    def test_fusion_shape_and_slices(self) -> None:
        rng = np.random.default_rng(42)
        byte_seq = rng.integers(0, 256, size=(5, 8), dtype=np.uint8)
        entropy = rng.uniform(0, 8, size=(5, 1)).astype(np.float32)
        ngram_features = {
            1: rng.uniform(0, 1, size=(5, 4)).astype(np.float32),
            2: rng.uniform(0, 1, size=(5, 6)).astype(np.float32),
        }

        fused = fuse_feature_groups(
            byte_seq=byte_seq,
            entropy=entropy,
            ngram_features=ngram_features,
        )

        self.assertIsInstance(fused, FusedFeatures)
        self.assertEqual(fused.fused_vector.shape, (5, 8 + 1 + 4 + 6))
        self.assertEqual(fused.slices["byte_seq"], (0, 8))
        self.assertEqual(fused.slices["entropy"], (8, 9))
        self.assertEqual(fused.slices["ngram"], (9, 19))
        self.assertAlmostEqual(sum(fused.weights.values()), 1.0, places=6)

    def test_ngram_can_be_disabled(self) -> None:
        byte_seq = np.ones((3, 4), dtype=np.uint8)
        entropy = np.ones((3, 1), dtype=np.float32)
        config = {
            "feature_groups": {
                "byte_seq": {"enabled": True, "normalize": "scale", "scale": 255.0},
                "entropy": {"enabled": True, "normalize": "minmax", "expected_min": 0.0, "expected_max": 8.0},
                "ngram": {"enabled": False},
            }
        }
        fused = fuse_feature_groups(byte_seq=byte_seq, entropy=entropy, config=config)
        self.assertEqual(fused.fused_vector.shape, (3, 5))
        self.assertNotIn("ngram", fused.slices)

    def test_sample_count_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            fuse_feature_groups(
                byte_seq=np.ones((3, 4), dtype=np.uint8),
                entropy=np.ones((4, 1), dtype=np.float32),
            )


class SaveFusionOutputsTest(unittest.TestCase):
    def test_save_outputs(self) -> None:
        rng = np.random.default_rng(7)
        fused = fuse_feature_groups(
            byte_seq=rng.integers(0, 256, size=(4, 6), dtype=np.uint8),
            entropy=rng.uniform(0, 8, size=(4, 1)).astype(np.float32),
            ngram_features={1: rng.uniform(0, 1, size=(4, 3)).astype(np.float32)},
            config={
                "feature_groups": {
                    "ngram": {"enabled": True, "ns": [1], "normalize": "none"},
                }
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = save_fusion_outputs(fused, output_dir)
            self.assertTrue((output_dir / "fused_vector.npy").is_file())
            self.assertTrue((output_dir / "fusion_weights.json").is_file())
            self.assertTrue((output_dir / "fusion_summary.json").is_file())
            self.assertEqual(summary["sample_count"], 4)
            loaded = np.load(output_dir / "fused_vector.npy")
            np.testing.assert_allclose(loaded, fused.fused_vector)
            weights = json.loads(
                (output_dir / "fusion_weights.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(weights), {"byte_seq", "entropy", "ngram"})


if __name__ == "__main__":
    unittest.main()
