import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import inspect
import shutil
from unittest.mock import patch

PLAN = Path(__file__).resolve().parents[2] / "training_plan"
sys.path.insert(0, str(PLAN))
spec = importlib.util.spec_from_file_location("sync_training_release", PLAN / "sync_training_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ParallelUploadTests(unittest.TestCase):
    def test_sender_avoids_full_video_checksum_prescan_but_forces_selected_files(self):
        text = inspect.getsource(module.main)
        self.assertNotIn('"--checksum"', text)
        self.assertIn('"--ignore-times"', text)

    @unittest.skipUnless(shutil.which("rsync"), "requires the host rsync executable")
    def test_release_symlink_upload_copies_content_not_local_only_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "release", root / "remote"
            source.mkdir(); destination.mkdir()
            payload = root / "accepted-video.mp4"
            payload.write_bytes(b"accepted video content")
            (source / "video.mp4").symlink_to(payload)
            (destination / "video.mp4").symlink_to("/nonexistent/local-only/video.mp4")
            text = inspect.getsource(module.main)
            assert '"--copy-links"' in text
            module.upload_files(["rsync", "-a", "--copy-links", "--files-from=-",
                                 str(source) + "/", str(destination) + "/"],
                                ["video.mp4"], {}, 1)
            self.assertFalse((destination / "video.mp4").is_symlink())
            self.assertEqual((destination / "video.mp4").read_bytes(), payload.read_bytes())

    def test_expected_digest_skips_local_rehash_but_never_accepts_invalid_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(b"video")
            expected = {"size": 5, "sha256": module.digest_file(path)}
            with patch.object(module, "digest_file", side_effect=AssertionError("must reuse proven digest")):
                self.assertEqual(module.expected_video_identity(path, 5, expected), expected)
                for invalid in ({"size": 4, "sha256": expected["sha256"]},
                                {"size": 5, "sha256": "not-a-digest"}):
                    with self.assertRaises(ValueError):
                        module.expected_video_identity(path, 5, invalid)

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
