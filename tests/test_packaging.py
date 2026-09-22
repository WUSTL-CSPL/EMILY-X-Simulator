"""Local data links must preserve existing files; reference checks must fail closed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_link_data_is_repeatable(self):
        link = load_script("link_local_data")
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp)/"source", Path(temp)/"dest"
            dest.mkdir()
            for name in ("EMILY-X", "srtm_data"):
                (source/name).mkdir(parents=True)
            link.link_data(source, dest)
            link.link_data(source, dest)
            for name in ("EMILY-X", "srtm_data"):
                self.assertTrue((dest/name).is_symlink())
                self.assertEqual((dest/name).resolve(), source/name)

    def test_existing_data_are_untouched_on_failure(self):
        link = load_script("link_local_data")
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp)/"source", Path(temp)/"dest"
            dest.mkdir()
            for name in ("EMILY-X", "srtm_data"):
                (source/name).mkdir(parents=True)
            (dest/"srtm_data").write_text("keep me")
            with self.assertRaises(ValueError):
                link.link_data(source, dest)
            self.assertFalse((dest/"EMILY-X").exists())
            self.assertEqual((dest/"srtm_data").read_text(), "keep me")

    def test_missing_or_changed_reference_file_fails(self):
        verify = load_script("verify_reference")
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = {"data.jsonl": hashlib.sha256(b"example\n").hexdigest()}
            self.assertFalse(verify.check_files(root, expected)["data.jsonl"]["matches"])
            (root/"data.jsonl").write_bytes(b"example\n")
            self.assertTrue(verify.check_files(root, expected)["data.jsonl"]["matches"])
            (root/"data.jsonl").write_bytes(b"changed\n")
            self.assertFalse(verify.check_files(root, expected)["data.jsonl"]["matches"])


if __name__ == "__main__":
    unittest.main()
