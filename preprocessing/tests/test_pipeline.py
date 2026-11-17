"""Local verification of the D-Vlog preprocessing pipeline.

Pure-logic tests need no deps. The Rust-bridge and end-to-end tests need the
built CLI; they skip cleanly when it is absent. Run:

  python -m unittest discover -s preprocessing/tests -v
"""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from preprocessing import assemble, config, dvlog, encoders, pipeline, sampling, signals
from preprocessing.manifest import Manifest
from preprocessing.schema_catalog import CATALOG


def _cli_available():
    from preprocessing import rust_bridge
    try:
        rust_bridge.find_cli()
        return True
    except FileNotFoundError:
        return False


class TestCatalog(unittest.TestCase):
    def test_catalog_size(self):
        self.assertEqual(len(CATALOG), 252)
        ids = [e[0] for e in CATALOG]
        self.assertEqual(len(set(ids)), 252)  # collision-free
        ho = [e for e in CATALOG if e[0].startswith("HO")]
        self.assertEqual(len(ho), 52)  # 2/3/4/5-way block


class TestDvlog(unittest.TestCase):
    def test_load_counts(self):
        recs = dvlog.load_records(config.DVLOG_CSV)
        self.assertEqual(len(recs), 961)
        counts = dvlog.label_counts(recs)
        self.assertEqual(counts, {0: 406, 1: 555})
        self.assertTrue(all(r["label"] in (0, 1) for r in recs))
        self.assertTrue(all(r["key"] for r in recs))


class TestManifest(unittest.TestCase):
    def test_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"
            m = Manifest(path)
            self.assertFalse(m.is_done("7"))
            m.mark("7", {"label": 1})
            self.assertTrue(m.is_done("7"))
            again = Manifest(path)  # reload from disk
            self.assertTrue(again.is_done("7"))
            self.assertEqual(len(again), 1)


class TestSignals(unittest.TestCase):
    def test_synthetic_deterministic(self):
        a = signals.synthetic_signals(3, True)
        b = signals.synthetic_signals(3, True)
        self.assertEqual(a, b)

    def test_class_separation(self):
        dep = signals.synthetic_signals(1, True)
        nor = signals.synthetic_signals(1, False)
        self.assertLess(dep["unique"], nor["unique"])
        self.assertLess(max(dep["au12"]), max(nor["au12"]))

    def test_token_counts(self):
        from preprocessing.asr import token_counts
        c = token_counts("a a b")
        self.assertEqual(c, {"unique": 2, "total": 3})


class TestEncoders(unittest.TestCase):
    def test_synthetic_dim_and_determinism(self):
        v = encoders.synthetic_embedding(5, "audio")
        self.assertEqual(len(v), 768)
        self.assertEqual(v, encoders.synthetic_embedding(5, "audio"))
        self.assertNotEqual(v, encoders.synthetic_embedding(5, "text"))

    def test_save_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            emb = {m: encoders.synthetic_embedding(1, m) for m in encoders.MODALITIES}
            p = encoders.save_embeddings(d, "42", emb)
            self.assertTrue(p.exists())
            self.assertFalse(p.with_suffix(".tmp").exists())


class TestAssemble(unittest.TestCase):
    def test_fill_200_absent(self):
        recs = [{"video_id": "0", "label": 1}]
        wide, rel = assemble.assemble_rows(recs, {})
        self.assertEqual(len(wide), 1)
        self.assertEqual(len(wide[0]), 2 + len(assemble.FEATURE_IDS))
        self.assertTrue(all(math.isnan(wide[0][fid]) for fid in assemble.FEATURE_IDS))
        self.assertEqual(len(rel), len(assemble.FEATURE_IDS))
        self.assertTrue(all(r[2] == "Absent" for r in rel))


@unittest.skipUnless(_cli_available(), "Rust CLI not built")
class TestRustBridge(unittest.TestCase):
    def test_compute_features(self):
        from preprocessing import rust_bridge
        sig = {"0": signals.synthetic_signals(0, True)}
        with tempfile.TemporaryDirectory() as d:
            feats, _ = rust_bridge.compute_features(sig, d)
        self.assertIn("0", feats)
        self.assertIn("AUD-005", feats["0"])
        val, reliability = feats["0"]["AUD-005"]
        self.assertFalse(math.isnan(val))
        self.assertEqual(reliability, "Exact")
        # Higher-order 2-way feature is computed as a Proxy.
        self.assertIn("HO2-AT-C", feats["0"])
        ho_val, ho_rel = feats["0"]["HO2-AT-C"]
        self.assertFalse(math.isnan(ho_val))
        self.assertEqual(ho_rel, "Proxy")

    def test_derive_signals(self):
        import struct
        from preprocessing import rust_bridge
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sr = 16000
            n = sr // 2  # 0.5 s of 200 Hz sine, 16-bit mono PCM
            samples = [int(0.5 * 32767 * math.sin(2 * math.pi * 200 * i / sr))
                       for i in range(n)]
            data = b"".join(struct.pack("<h", s) for s in samples)
            hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            hdr += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            hdr += b"data" + struct.pack("<I", len(data))
            wav = d / "tone.wav"
            wav.write_bytes(hdr + data)
            # 3 raw 4x4 gray frames: 0, 10, 20 -> motion [10, 10]
            raw = d / "motion.gray"
            raw.write_bytes(bytes([0] * 16 + [10] * 16 + [20] * 16))
            out = rust_bridge.derive_signals(wav, raw, d, "7", size=4)
            self.assertTrue(out["f0"])
            mean_f0 = sum(out["f0"]) / len(out["f0"])
            self.assertAlmostEqual(mean_f0, 200.0, delta=5.0)
            self.assertEqual(out["motion"], [10.0, 10.0])


@unittest.skipUnless(_cli_available(), "Rust CLI not built")
class TestPipelineEndToEnd(unittest.TestCase):
    def test_sample_run_and_resume(self):
        recs = dvlog.load_records(config.DVLOG_CSV)
        with tempfile.TemporaryDirectory() as d:
            summary = pipeline.run(recs, d, mode="sample", batch_size=2, limit=4)
            self.assertEqual(summary["processed"], 4)
            self.assertEqual(summary["merged_rows"], 4)
            wide_path = Path(d) / "features_wide.csv"
            self.assertTrue(wide_path.exists())
            with open(wide_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(rows[0]), 2 + len(assemble.FEATURE_IDS))
            # Embeddings: one JSON per video, 4 modalities x 768.
            embs = list((Path(d) / "batches").glob("batch_*/embeddings/*.json"))
            self.assertEqual(len(embs), 4)
            # Resume: a second run skips all finished videos.
            again = pipeline.run(recs, d, mode="sample", batch_size=2, limit=4)
            self.assertEqual(again["processed"], 0)
            self.assertEqual(again["skipped_done"], 4)

    def test_flush_fires_after_each_batch_and_merge(self):
        recs = dvlog.load_records(config.DVLOG_CSV)
        calls = []

        def flush(out_root):
            # The manifest must already be saved when the flush runs, so a
            # cloud sync never trails the record of finished work.
            self.assertTrue((Path(out_root) / "manifest.json").exists())
            calls.append(Path(out_root))

        with tempfile.TemporaryDirectory() as d:
            pipeline.run(recs, d, mode="sample", batch_size=2, limit=4,
                         flush=flush)
            # 2 batches + 1 final merge = 3 flushes.
            self.assertEqual(len(calls), 3)


class TestSampling(unittest.TestCase):
    def _recs(self):
        # 6 of label 1 then 6 of label 0: a front-loaded order.
        return [{"video_id": str(i), "label": 1 if i < 6 else 0}
                for i in range(12)]

    def test_balanced(self):
        sub = sampling.stratified_subset(self._recs(), 4)
        self.assertEqual(len(sub), 4)
        self.assertEqual(sum(r["label"] for r in sub), 2)  # 2 of each class

    def test_limit_none_returns_all(self):
        recs = self._recs()
        self.assertEqual(sampling.stratified_subset(recs, None), recs)
        self.assertEqual(sampling.stratified_subset(recs, 99), recs)

    def test_deterministic_and_spread(self):
        recs = self._recs()
        a = sampling.stratified_subset(recs, 4)
        b = sampling.stratified_subset(recs, 4)
        self.assertEqual(a, b)
        ids = {int(r["video_id"]) for r in a}
        self.assertTrue(any(i >= 6 for i in ids))  # reaches the tail class
        self.assertTrue(any(i > 0 for i in ids if i < 6))  # not just the head

    def test_head_slice_was_imbalanced(self):
        recs = self._recs()
        self.assertEqual(sum(r["label"] for r in recs[:4]), 4)  # the old bug


def _cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_cv2_available(), "cv2 not installed (Colab only)")
class TestFaceCrop(unittest.TestCase):
    def test_crop_differs_from_full_frame(self):
        import cv2
        import numpy as np
        from preprocessing import face
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "frame_00001.jpg"
            cv2.imwrite(str(src), np.full((200, 200, 3), 127, np.uint8))
            crops = face.face_crops([src], Path(d) / "face_crops")
            self.assertEqual(len(crops), 1)
            img = cv2.imread(str(crops[0]))
            # No face in a flat image -> center crop, smaller than the frame.
            self.assertLess(img.shape[0], 200)
            self.assertLess(img.shape[1], 200)


class TestStreamingBatch(unittest.TestCase):
    """The producer-consumer consumer side: pull-and-process per item."""

    def _items(self, n):
        return iter([({"video_id": str(i), "label": i % 2, "key": str(i)},
                      None, None) for i in range(n)])

    def test_full_batch_suspends_iterator(self):
        with tempfile.TemporaryDirectory() as d:
            it = self._items(5)
            wd, ok, failed, exhausted = pipeline._process_batch(
                it, "sample", d, d, 0, 2)
            self.assertEqual(len(ok), 2)
            self.assertFalse(exhausted)
            self.assertEqual(wd.name, "batch_0000")
            self.assertEqual(len(list(it)), 3)  # pulled exactly batch_size

    def test_partial_batch_exhausted(self):
        with tempfile.TemporaryDirectory() as d:
            wd, ok, failed, exhausted = pipeline._process_batch(
                self._items(3), "sample", d, d, 0, 8)
            self.assertEqual(len(ok), 3)
            self.assertTrue(exhausted)

    def test_empty_iterator_no_workdir(self):
        with tempfile.TemporaryDirectory() as d:
            wd, ok, failed, exhausted = pipeline._process_batch(
                iter([]), "sample", d, d, 0, 8)
            self.assertIsNone(wd)
            self.assertTrue(exhausted)
            self.assertEqual(ok, [])

    def test_download_error_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            items = iter([
                ({"video_id": "0", "label": 1, "key": "0"}, None, "Boom: x"),
                ({"video_id": "1", "label": 0, "key": "1"}, None, None),
            ])
            wd, ok, failed, exhausted = pipeline._process_batch(
                items, "sample", d, d, 0, 8)
            self.assertEqual(len(ok), 1)
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0][0], "0")


if __name__ == "__main__":
    unittest.main()