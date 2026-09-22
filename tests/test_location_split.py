"""Split reproducibility and leakage guards, using the existing v2 pool."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import split_location_dataset as split


class GeographicSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = split.ROOT / "datasets/terrestrial_location_v2/campaign"
        if not (root/"incident_training/selection.json").is_file() or not (
                split.ROOT/"datasets/terrestrial_location_v3/verification.json").is_file():
            raise unittest.SkipTest("Generate the v3 pool and split to run data integration checks")
        cls.inputs, cls.labels, cls.truth = [split.read_table(root / name) for name in
            ("inputs.jsonl", "labels.jsonl", "ground_truth.jsonl")]
        cls.old = set(json.loads((root / "incident_training/selection.json").read_text())["sample_ids"])
        cls.selected, cls.audit, cls.roles, _, _ = split.select_samples(cls.inputs, cls.truth, cls.old, 20260922)

    def test_reproducible_despite_source_row_order(self):
        selected, _, roles, _, _ = split.select_samples(
            dict(reversed(list(self.inputs.items()))), dict(reversed(list(self.truth.items()))), self.old, 20260922)
        self.assertEqual(selected, self.selected)
        self.assertEqual(roles, self.roles)

    def test_geographic_and_historical_isolation(self):
        distances = split.verify_split(self.selected, self.roles, self.inputs, self.labels, self.truth, self.old)
        self.assertTrue(all(d >= 500 for d in distances.values()))
        self.assertEqual([len(self.selected[k]) for k in split.SIZES], [1000, 200, 200])

    def test_prior_evaluation_id_rejected(self):
        contaminated_history = self.old | {self.selected["test"][0]}
        with self.assertRaisesRegex(ValueError, "Previously evaluated"):
            split.verify_split(self.selected, self.roles, self.inputs, self.labels, self.truth, contaminated_history)

    def test_label_mismatch_rejected(self):
        labels = dict(self.labels)
        sid = self.selected["train"][0]
        labels[sid] = {**labels[sid], "latitude_deg": labels[sid]["latitude_deg"] + 1.0}
        with self.assertRaisesRegex(ValueError, "Label/truth"):
            split.verify_split(self.selected, self.roles, self.inputs, labels, self.truth, self.old)

    def test_cross_region_swap_rejected(self):
        changed = copy.deepcopy(self.selected)
        changed["train"][0], changed["test"][0] = changed["test"][0], changed["train"][0]
        with self.assertRaises(ValueError):
            split.verify_split(changed, self.roles, self.inputs, self.labels, self.truth, self.old)

    def test_block_boundaries_and_buffer(self):
        self.assertEqual(split.block_and_margin(-6000, -8000), (1, 0.0))
        self.assertEqual(split.block_and_margin(-5750, -8000), (1, 250.0))
        for sid, row in self.audit.items():
            if row["boundary_margin_m"] < 250:
                self.assertFalse(row["selected_split"])

    def test_export_preserves_whole_receiver_groups(self):
        dest = split.ROOT / "datasets/terrestrial_location_v3"
        for role, ids in self.selected.items():
            exported = split.read_table(dest / role / "inputs.jsonl")
            self.assertEqual(exported, {sid: self.inputs[sid] for sid in ids})


if __name__ == "__main__":
    unittest.main()
