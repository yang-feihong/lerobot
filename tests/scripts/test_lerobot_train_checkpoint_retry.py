from pathlib import Path

from lerobot.scripts.lerobot_train import _run_checkpoint_save_with_retry


def test_checkpoint_save_retries_forever_until_success_and_cleans_partial_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "005000"
    attempts = 0
    sleep_calls: list[float] = []

    def save_once() -> None:
        nonlocal attempts
        attempts += 1
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / f"attempt_{attempts}.partial").write_text("partial")
        if attempts < 3:
            raise OSError("simulated checkpoint write failure")
        (checkpoint_dir / "complete").write_text("ok")

    _run_checkpoint_save_with_retry(
        checkpoint_dir,
        save_once,
        retry_interval_s=600,
        sleep_fn=sleep_calls.append,
    )

    assert attempts == 3
    assert sleep_calls == [600, 600]
    assert (checkpoint_dir / "complete").read_text() == "ok"
    assert not (checkpoint_dir / "attempt_1.partial").exists()
    assert not (checkpoint_dir / "attempt_2.partial").exists()


def test_checkpoint_save_replaces_stale_partial_directory_before_first_attempt(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "010000"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "stale.partial").write_text("stale")

    def save_once() -> None:
        assert not checkpoint_dir.exists()
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "complete").write_text("ok")

    _run_checkpoint_save_with_retry(checkpoint_dir, save_once, sleep_fn=lambda _: None)

    assert not (checkpoint_dir / "stale.partial").exists()
    assert (checkpoint_dir / "complete").read_text() == "ok"
