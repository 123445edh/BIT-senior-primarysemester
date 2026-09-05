"""feature_extractor 模块的单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_extractor import (
    build_features,
    count_ngrams,
    count_selected_ngrams,
    make_byte_sequence,
    parse_bytes_file,
    save_outputs,
    select_top_k,
    shannon_entropy_from_counts,
)


class ParseBytesFileTest(unittest.TestCase):
    def test_parse_known_unknown_and_invalid_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bytes"
            path.write_bytes(
                b"00401000 00 01 ?? 02\n"
                b"00401010 03 04 05 06\n"
                b"00401020 zz 07 08\n"
            )
            values, known, total, known_tokens, unknown_tokens, invalid_tokens = (
                parse_bytes_file(path)
            )

            self.assertEqual(total, 11)
            self.assertEqual(known_tokens, 9)
            self.assertEqual(unknown_tokens, 1)
            self.assertEqual(invalid_tokens, 1)
            self.assertEqual(values.tolist(), [0, 1, 0, 2, 3, 4, 5, 6, 0, 7, 8])
            self.assertEqual(
                known.tolist(),
                [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            )


class MakeByteSequenceTest(unittest.TestCase):
    def test_keep_head_truncates(self) -> None:
        values = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        sequence, mask, length, truncated, padded = make_byte_sequence(
            values, max_length=3, truncate="keep_head"
        )
        np.testing.assert_array_equal(sequence, [1, 2, 3])
        np.testing.assert_array_equal(mask, [1, 1, 1])
        self.assertEqual(length, 5)
        self.assertEqual(truncated, 2)
        self.assertEqual(padded, 0)

    def test_keep_tail_pads_on_left(self) -> None:
        values = np.array([1, 2, 3], dtype=np.uint8)
        sequence, mask, length, truncated, padded = make_byte_sequence(
            values, max_length=5, pad_value=9, truncate="keep_tail"
        )
        np.testing.assert_array_equal(sequence, [9, 9, 1, 2, 3])
        np.testing.assert_array_equal(mask, [0, 0, 1, 1, 1])
        self.assertEqual(padded, 2)


class EntropyTest(unittest.TestCase):
    def test_uniform_distribution(self) -> None:
        self.assertAlmostEqual(
            shannon_entropy_from_counts(np.array([1, 1, 1, 1])), 2.0
        )

    def test_single_symbol_is_zero(self) -> None:
        self.assertAlmostEqual(shannon_entropy_from_counts(np.array([4])), 0.0)

    def test_empty_is_zero(self) -> None:
        self.assertAlmostEqual(shannon_entropy_from_counts(np.array([])), 0.0)


class NgramTest(unittest.TestCase):
    def test_known_only_ngrams_do_not_cross_unknown(self) -> None:
        values = np.array([0, 1, 0, 2, 3], dtype=np.uint8)
        known = np.array([1, 1, 0, 1, 1], dtype=bool)

        bigram_counts = count_ngrams(values, known, 2)
        self.assertEqual(int(bigram_counts[0x0001]), 1)
        self.assertEqual(int(bigram_counts[0x0203]), 1)
        self.assertEqual(int(bigram_counts.sum()), 2)

        trigram_counts = count_ngrams(values, known, 3)
        self.assertEqual(int(trigram_counts.sum()), 0)

    def test_trigram_counts(self) -> None:
        values = np.array([0, 1, 2, 3], dtype=np.uint8)
        known = np.ones(4, dtype=bool)
        counts = count_ngrams(values, known, 3)
        self.assertEqual(int(counts[0x000102]), 1)
        self.assertEqual(int(counts[0x010203]), 1)

    def test_count_selected_ngrams(self) -> None:
        values = np.array([0, 1, 0, 1, 2], dtype=np.uint8)
        known = np.ones(5, dtype=bool)
        selected = np.array([0x0001, 0x0102], dtype=np.int64)
        counts, total = count_selected_ngrams(values, known, 2, selected)
        self.assertEqual(total, 4)
        np.testing.assert_array_equal(counts, [2, 1])


class SelectTopKTest(unittest.TestCase):
    def test_select_top_k_sorted_and_ties_by_index(self) -> None:
        counts = np.array([3, 5, 5, 1], dtype=np.int64)
        selected = select_top_k(counts, 2)
        np.testing.assert_array_equal(selected, [1, 2])

    def test_select_all_when_k_exceeds_size(self) -> None:
        counts = np.array([1, 2, 3], dtype=np.int64)
        selected = select_top_k(counts, 10)
        np.testing.assert_array_equal(selected, [0, 1, 2])


class BuildFeaturesTest(unittest.TestCase):
    def test_build_and_save_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bytes_dir = root / "bytes"
            bytes_dir.mkdir()
            labels = root / "labels.csv"
            manifest = root / "manifest.csv"
            output_dir = root / "outputs"

            (bytes_dir / "a.bytes").write_bytes(
                b"00401000 00 01 02 03\n"
                b"00401010 04 05 06 07\n"
            )
            (bytes_dir / "b.bytes").write_bytes(
                b"00401000 00 00 01 01\n"
            )
            labels.write_text("Id,Class\na,1\nb,2\n", encoding="utf-8")
            manifest.write_text(
                "Id,Class,split,image,unknown_ratio,quality\n"
                "a,1,train,,,\n"
                "b,2,test,,,\n",
                encoding="utf-8",
            )

            config = {
                "input": {
                    "bytes_dir": str(bytes_dir),
                    "labels_csv": str(labels),
                    "manifest_csv": str(manifest),
                },
                "output_dir": str(output_dir),
                "max_samples": 0,
                "byte_sequence": {
                    "max_length": 8,
                    "pad_value": 0,
                    "unknown_value": 0,
                    "truncate": "keep_head",
                },
                "entropy": {"enabled": True},
                "ngram": {
                    "ns": [1, 2],
                    "top_k": {"1": 256, "2": 8},
                    "normalize": "frequency",
                },
            }

            batch = build_features(config, base_dir=root)
            self.assertEqual(batch.ids, ["a", "b"])
            self.assertEqual(batch.byte_seq.shape, (2, 8))
            self.assertEqual(batch.byte_seq_mask.shape, (2, 8))
            self.assertEqual(batch.entropy.shape, (2, 1))
            self.assertEqual(batch.ngram_features[1].shape, (2, 256))
            self.assertEqual(batch.ngram_features[2].shape, (2, 8))
            self.assertAlmostEqual(float(batch.ngram_features[1][0].sum()), 1.0, places=5)
            self.assertAlmostEqual(float(batch.ngram_features[2][0].sum()), 1.0, places=5)

            summary = save_outputs(batch, output_dir)
            self.assertEqual(summary["sample_count"], 2)
            self.assertTrue((output_dir / "samples.csv").is_file())
            self.assertTrue((output_dir / "byte_seq.npy").is_file())
            self.assertTrue((output_dir / "ngram_1_features.npy").is_file())
            self.assertTrue((output_dir / "ngram_2_vocab.csv").is_file())


if __name__ == "__main__":
    unittest.main()
