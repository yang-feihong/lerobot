#!/usr/bin/env python3
"""Sync an immutable release's training assets, reusing verified remote video inodes."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from dataset_release import RECEIPT, verify_release


def upload_files(command, files, env, parallel):
    """Each path belongs to one sender; all senders finish before verification."""
    if len(files) != len(set(files)):
        raise ValueError("Duplicate upload paths")
    groups = [files[index::parallel] for index in range(min(parallel, len(files)))]

    def send(index, group):
        time.sleep(index * 0.5)
        for attempt in range(5):
            try:
                subprocess.run(command, input="\n".join(group) + "\n", text=True, check=True, env=env)
                return
            except subprocess.CalledProcessError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(send, index, group) for index, group in enumerate(groups)]
        for future in futures:
            future.result()


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_video_identity(path, size, expected=None):
    """A proven expected digest is verified against received bytes, never size alone."""
    if expected is None:
        return {"size": size, "sha256": digest_file(path)}
    if (set(expected) != {"size", "sha256"} or expected["size"] != size
            or path.stat().st_size != size
            or not isinstance(expected["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
        raise ValueError(f"Invalid expected video identity: {path}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ssh-config", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-training-root", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parallel-transfers", type=int, default=1)
    parser.add_argument("--expected-video-identities", type=Path,
                        help="Optional relative-path identities from a previously verified SHA256 receipt; received bytes still fully hashed")
    args = parser.parse_args()
    if not 1 <= args.parallel_transfers <= 32:
        parser.error("--parallel-transfers must be between 1 and 32")
    source = args.source.resolve()
    receipt = verify_release(source)
    recipe=receipt['inputs']['request']['recipe']
    logical=receipt.get('logical_recipe',recipe)
    if any(not name or Path(name).name!=name or name.startswith('.') for name in (logical,recipe)):
        raise ValueError('Unsafe training recipe identity')
    remote_root = str(Path(args.remote_training_root).parent/'release_history'/logical/receipt['artifact_id'])
    if not Path(remote_root).is_absolute():
        raise ValueError("Remote release path must be explicit and absolute")
    env = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    ssh = ["ssh", "-F", args.ssh_config, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", args.host]

    def remote(code, payload):
        result = subprocess.run(
            ssh + ["python3 -c " + shlex.quote(code)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if result.returncode:
            raise RuntimeError(f"Remote release operation failed on {args.host}: {result.stderr.strip()}")
        return json.loads(result.stdout)

    expected = json.loads(args.expected_video_identities.read_text()) if args.expected_video_identities else {}
    if not isinstance(expected, dict) or set(expected) - set(receipt["assets"]["videos"]):
        raise ValueError("Expected video identities must reference release video paths only")
    videos = {}
    for index, (relative, size) in enumerate(receipt["assets"]["videos"].items()):
        videos[relative] = expected_video_identity(source / relative, size, expected.get(relative))
        if index % 500 == 0:
            print(f"Local video identity {index}/{len(receipt['assets']['videos'])}", flush=True)
    result = remote(
        r"""
import hashlib, json, os, sys
from pathlib import Path
p = json.load(sys.stdin)
root, training = Path(p["root"]), Path(p["training"])
training.mkdir(parents=True, exist_ok=True)
root.mkdir(parents=True, exist_ok=True)
sizes = {v["size"] for v in p["videos"].values()}
by_size = {}
for dataset in training.iterdir():
    if dataset == root or not dataset.is_dir():
        continue
    for path in (dataset / "videos").rglob("*.mp4"):
        size = path.stat().st_size
        if size in sizes:
            by_size.setdefault(size, []).append(path)
cache = {}
def sha(path):
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        cache[key] = h.hexdigest()
    return cache[key]
linked, existing, missing, bytes_reused = 0, 0, [], 0
for relative, identity in p["videos"].items():
    target = root / relative
    # Releases may use absolute local symlinks into accepted simulation shards.
    # Remote training must receive the bytes, never a link to a local-only path.
    if target.is_symlink():
        target.unlink()
    if target.exists():
        if target.stat().st_size != identity["size"] or sha(target) != identity["sha256"]:
            raise ValueError(f"Existing destination video differs: {target}")
        existing += 1
        continue
    matched = next((candidate for candidate in by_size.get(identity["size"], [])
                    if candidate.stat().st_dev == root.stat().st_dev and sha(candidate) == identity["sha256"]), None)
    if matched is None:
        missing.append(relative)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(matched, target)
    if target.stat().st_ino != matched.stat().st_ino:
        raise ValueError("Hardlink inode verification failed")
    linked += 1
    bytes_reused += identity["size"]
print(json.dumps(dict(linked=linked, existing=existing, missing=missing, bytes_reused=bytes_reused)))
""",
        {"root": remote_root, "training": args.remote_training_root, "videos": videos},
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "missing"}
            | {"videos_to_upload": len(result["missing"])}
        ),
        flush=True,
    )
    transfer_assets = dict(receipt["assets"]["files"])
    if (source / "meta/ee_frames.json").is_file():
        transfer_assets["meta/ee_frames.json"] = digest_file(source / "meta/ee_frames.json")
    files = [*transfer_assets, RECEIPT, *result["missing"]]
    upload_files(
        [
            "rsync",
            "-a",
            "--copy-links",
            "--partial",
            "--partial-dir=.rsync-partial",
            # Final receiver SHA256 is authoritative. --checksum would pre-read
            # every large source video before rsync sends even its file list.
            # Force selected files through instead of trusting size/mtime.
            "--ignore-times",
            "--files-from=-",
            "--stats",
            "-e",
            shlex.join(ssh[:-1]),
            str(source) + "/",
            args.host + ":" + remote_root + "/",
        ],
        files,
        env,
        args.parallel_transfers,
    )
    verified = remote(
        r"""
import hashlib, json, sys
from pathlib import Path
p = json.load(sys.stdin)
root = Path(p["root"])
for relative, expected in p["files"].items():
    h = hashlib.sha256()
    with (root / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError(f"Training asset hash mismatch: {relative}")
for relative, expected in p["videos"].items():
    path = root / relative
    if path.stat().st_size != expected["size"]:
        raise ValueError(f"Video size mismatch: {relative}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != expected["sha256"]:
        raise ValueError(f"Video hash mismatch: {relative}")
# A resumed transfer may repartition paths between senders. Only after every
# final target is verified may its old rsync partial copy be discarded.
for relative in p["files"].keys() | p["videos"].keys():
    target = root / relative
    partial = target.parent / ".rsync-partial" / target.name
    if partial.is_file():
        partial.unlink()
print(json.dumps(dict(verified=True, training_files=len(p["files"]), videos=len(p["videos"]))))
""",
        {
            "root": remote_root,
            "files": transfer_assets | {RECEIPT: digest_file(source / RECEIPT)},
            "videos": videos,
        },
    )
    published = remote(
        r"""
import fcntl, json, os, sys
from pathlib import Path
p=json.load(sys.stdin)
training=Path(p['training']); root=Path(p['root'])
training.mkdir(parents=True,exist_ok=True)
with (training/'.publication.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    entry=training/p['logical']
    if entry.exists() and not entry.is_symlink():
        raise ValueError('Legacy physical training entry exists; migrate it before publishing')
    path=training/'releases.json'
    index=json.loads(path.read_text()) if path.exists() else dict(format='b2_z1.training_release',requests={})
    ready=dict(status='READY',path=entry.name,artifact_id=p['artifact_id'])
    index.setdefault('recipes',{})[p['recipe']]=ready
    index.setdefault('artifacts',{})[p['artifact_id']]=os.path.relpath(root,training)
    pending=training/('.'+p['logical']+'.link');pending.unlink(missing_ok=True)
    pending.symlink_to(os.path.relpath(root,training),target_is_directory=True)
    os.replace(pending,entry)
    pending=path.with_suffix('.json.tmp');pending.write_text(json.dumps(index,indent=2)+'\n')
    os.replace(pending,path)
print(json.dumps(dict(training_entry=str(entry))))
""",
        dict(training=args.remote_training_root,root=remote_root,logical=logical,recipe=recipe,
             artifact_id=receipt['artifact_id']),
    )
    report = {
        "source": str(source),
        "destination": remote_root,
        "artifact_id": receipt["artifact_id"],
        "parallel_transfers": args.parallel_transfers,
        "expected_video_identities": str(args.expected_video_identities) if args.expected_video_identities else None,
        "reused_expected_video_digests": len(expected),
        **result,
        **verified,
        **published,
        "video_bytes_uploaded": sum(videos[name]["size"] for name in result["missing"]),
        "excluded": ["raw_nonvisual", "diagnostic images", "unused non-training assets"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
