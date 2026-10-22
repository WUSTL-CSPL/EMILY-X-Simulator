"""Portable metadata bootstrap must be repeatable and preserve existing files."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_v8',ROOT/'scripts/prepare_v8_inputs.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def test_repeated_preparation_keeps_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copytree(ROOT/'reference',root/'reference')
            paths=module.prepare(root);before={p:Path(p).read_bytes() for p in paths}
            module.prepare(root)
            self.assertEqual(before,{p:Path(p).read_bytes() for p in paths})
            config=json.loads(Path(paths[0]).read_text());self.assertEqual(config['srtm_dir'],str(root/'srtm_data'))

    def test_conflicting_existing_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copytree(ROOT/'reference',root/'reference')
            paths=module.prepare(root);path=Path(paths[0]);data=json.loads(path.read_text());data['seed']+=1;path.write_text(json.dumps(data));before=path.read_bytes()
            with self.assertRaises(ValueError):module.prepare(root)
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
