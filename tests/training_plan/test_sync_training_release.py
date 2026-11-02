import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PLAN = Path(__file__).resolve().parents[2] / "training_plan"
sys.path.insert(0, str(PLAN))
spec = importlib.util.spec_from_file_location("sync_training_release", PLAN / "sync_training_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ParallelUploadTests(unittest.TestCase):
    def test_every_file_has_exactly_one_sender(self):
        files = [f"videos/episode-{i:06d}.mp4" for i in range(101)]
        for workers in (1, 4, 16, 32):
            with self.subTest(workers=workers), patch.object(module.subprocess, "run") as run, \
                    patch.object(module.time, "sleep"):
                module.upload_files(["rsync"], files, {}, workers)
                received = [name for call in run.call_args_list
                            for name in call.kwargs["input"].splitlines()]
                self.assertCountEqual(received, files)
                self.assertEqual(len(received), len(set(received)))
                self.assertTrue(all(c.kwargs["check"] for c in run.call_args_list))

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            module.upload_files(["rsync"], ["x", "x"], {}, 4)

    def test_transfer_failure_propagates(self):
        with patch.object(module.subprocess, "run", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                module.upload_files(["rsync"], ["x", "y"], {}, 2)

    def test_transient_connection_failure_is_retried(self):
        failure = module.subprocess.CalledProcessError(255, ["rsync"])
        with patch.object(module.subprocess, "run", side_effect=[failure, None]) as run, \
                patch.object(module.time, "sleep"):
            module.upload_files(["rsync"], ["x"], {}, 1)
            self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
