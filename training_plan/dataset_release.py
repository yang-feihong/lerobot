"""Dependency-free release protocol shared by preparation and training launch.

The data pipeline owns construction/physical acceptance. Training can only
consume an immutable release; it never guesses a cleaned path or sidecar name.
"""
import hashlib
import json
from pathlib import Path

FORMAT = "b2_z1.training_release"
RECEIPT = "meta/dataset_release.json"
# Bump only for an intentional, documented data-contract migration, not edits
# to validators or generators. Legacy accepted receipts implement this contract.
SEMANTIC_CONTRACT = "b2_z1.canonical_semantics.v1"


def compatible_contract(receipt):
    return receipt.get("semantic_contract", SEMANTIC_CONTRACT) == SEMANTIC_CONTRACT


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def asset_identity(root):
    """Hash numeric/schema/semantic assets; enumerate receiver-accepted videos.

    Video decoding/acceptance belongs to ingestion. This is deliberately not
    advertised as a fresh full-video checksum audit on every training launch.
    """
    root = Path(root)
    files = sorted(p for folder in ("data", "meta") for p in (root/folder).rglob("*")
                   if p.is_file() and str(p.relative_to(root)) != RECEIPT)
    provenance = root/"diagnostics/conversion_manifest.json"
    if provenance.is_file():
        files.append(provenance)
    return {"files": {str(p.relative_to(root)): sha256(p) for p in files},
            "videos": {str(p.relative_to(root)): p.stat().st_size
                       for p in sorted((root/"videos").rglob("*.mp4"))}}


def verify_release(root, request=None, *, allow_relation_subset=False):
    root = Path(root)
    path = root/RECEIPT
    if not path.is_file():
        raise ValueError(f"Not a published training release: {root}; prepare the data recipe first")
    receipt = read(path)
    if receipt.get("format") != FORMAT or receipt.get("status") != "READY":
        raise ValueError(f"Incomplete training release: {root}")
    if receipt["artifact_id"] != digest(receipt["inputs"]):
        raise ValueError("Invalid release identity")
    if not compatible_contract(receipt):
        raise ValueError("Training release requires an explicit semantic-contract migration")
    # Physical releases contain all accepted relations. Matrix/weight coverage
    # is checked by semantic_plan at training resolution, never by duplicating
    # the physical dataset for each supervision preset (including old releases).
    if request is not None and receipt['inputs']['request']['recipe'] != request['recipe']:
        raise ValueError("Training configuration differs from the locked release recipe")
    if not assets_match(root, receipt["assets"]):
        raise ValueError(f"Published training assets changed: {root}")
    return receipt


def assets_match(root, expected):
    """An audited all-legacy sidecar only makes the old implicit contract explicit.

    Preserve historical release identities; never exempt a physical/mixed map,
    nor an already-hashed sidecar, from asset integrity checks.
    """
    current = asset_identity(root)
    relative = "meta/ee_frames.json"
    if relative not in expected["files"] and relative in current["files"]:
        sidecar = read(Path(root) / relative)
        manifest = read(Path(root) / "diagnostics/conversion_manifest.json")
        records = {k: r for k, r in manifest["bags"].items() if r.get("status") == "done"}
        if (sidecar.get("version") == 1
                and sidecar.get("ee_frame_contract") == "same_tcp_tool_rx_minus_90_v1"
                and set(sidecar.get("recordings", {})) == set(records)
                and all(v == "legacy" for v in sidecar["recordings"].values())
                and all(r.get("ee_frame", "legacy") == "legacy" for r in records.values())):
            current["files"].pop(relative)
    return current == expected


def resolve_release(datasets_root, request, *, pinned_root=None, artifact_id=None):
    training = (Path(datasets_root)/"training").resolve()
    index = read(training/"releases.json") if (training/"releases.json").is_file() else {}
    if pinned_root is None and artifact_id is not None:
        location = index.get("artifacts", {}).get(artifact_id)
        root = (training/location if location else training/f"{request['recipe']}--{artifact_id}").resolve()
    elif pinned_root is None:
        entry = index.get("recipes", {}).get(request['recipe']) or index.get("requests", {}).get(digest(request))
        if entry is None or entry["status"] != "READY":
            raise ValueError(f"No READY release for recipe {request['recipe']}; run prepare_dataset.sh")
        root = (training/entry["path"]).resolve()
    else:
        original = Path(pinned_root).absolute()
        # Old checkpoint paths remain exact identities after recoverable archival.
        alias = index.get("aliases", {}).get(original.name) if original.parent == training else None
        root = (training/alias if alias else original).resolve()
    history = (training.parent/"release_history").resolve()
    if root.parent != training and not root.is_relative_to(history):
        raise ValueError("Training must use a registered release, never cleaned/")
    receipt = verify_release(root, request, allow_relation_subset=artifact_id is not None)
    if artifact_id is not None and receipt["artifact_id"] != artifact_id:
        raise ValueError("Explicit dataset artifact identity mismatch")
    return root


def same_release_root(saved, current):
    """Compare a checkpoint path after archival without adopting different data."""
    if not saved:return False
    current=Path(current).resolve()
    if current.parent.parent.name=='release_history':
        datasets=current.parents[2]
    elif current.parent.name=='training':
        datasets=current.parents[1]
    else:
        return False
    try:
        receipt=read(current/RECEIPT)
        return resolve_release(datasets,receipt['inputs']['request'],pinned_root=saved)==current
    except (OSError,ValueError,KeyError):
        return False
