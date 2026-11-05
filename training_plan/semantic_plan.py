#!/usr/bin/env python3
"""Resolve training jobs against maintained state-instruction presets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PRESETS_FORMAT = "lerobot.semantic_training_presets"
JOBS_FORMAT = "lerobot.training_jobs"
VERSION = 1
VALID_MEMORY_MODES = {"standard", "mem_vit", "image_mem", "full_mem"}
VALID_STATUSES = {"active", "complete", "blocked"}
VALID_ACTION_SUPERVISION = {"demonstrated", "hold"}
VALID_COMPLETION_BOUNDARIES = {"none", "state_end", "goal_end", "future_goal_end"}
VALID_NODES = {"f", "wb1", "wb2", "wb3", "wb4", "wb5", "wb6", "lerobot"}

# The launcher/preparation preflight also runs outside the training container.
# Only this dependency-free origin contract is imported, never datasets/torch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lerobot.data_domains import load_episode_domains, validate_domain_weights, validate_instruction_domain_weights


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_relation(state: str, intent: str, relation: Any) -> None:
    if not isinstance(state, str) or not state or not isinstance(intent, str) or not intent:
        raise ValueError("state and instruction names must be non-empty strings")
    if not isinstance(relation, dict):
        raise ValueError(f"relation {state} x {intent} must be an object")
    required = {"status", "action_supervision", "completion_boundary"}
    if set(relation) != required:
        raise ValueError(f"relation {state} x {intent} requires exactly {sorted(required)}")
    if relation["status"] not in VALID_STATUSES:
        raise ValueError(f"relation {state} x {intent} has invalid status")
    if relation["action_supervision"] not in VALID_ACTION_SUPERVISION:
        raise ValueError(f"relation {state} x {intent} has invalid action_supervision")
    if relation["completion_boundary"] not in VALID_COMPLETION_BOUNDARIES:
        raise ValueError(f"relation {state} x {intent} has invalid completion_boundary")


def _validate_semantic_definition(name: str, definition: Any) -> None:
    if not isinstance(definition, dict):
        raise ValueError(f"{name} must be an object")
    if set(definition) - {"domain_weights"} != {"state_weights", "state_instruction_matrix"}:
        raise ValueError(f"{name} requires state_weights and state_instruction_matrix")
    weights = definition["state_weights"]
    _validate_domain_weights(definition.get("domain_weights", {}))
    matrix = definition["state_instruction_matrix"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{name} requires explicit state_weights")
    if not isinstance(matrix, dict) or not matrix:
        raise ValueError(f"{name} requires an explicit state_instruction_matrix")
    if set(weights) != set(matrix):
        raise ValueError(f"{name} state weights and matrix rows must exactly match")
    if any(
        not isinstance(state, str) or not state or not isinstance(weight, (int, float)) or weight <= 0
        for state, weight in weights.items()
    ):
        raise ValueError(f"{name} has invalid state_weights")
    for state, instructions in matrix.items():
        if not isinstance(instructions, dict) or not instructions:
            raise ValueError(f"{name} state {state!r} requires instructions")
        for intent, relation in instructions.items():
            _validate_relation(state, intent, relation)


def _validate_domain_weights(weights):
    validate_domain_weights(weights)


def _validate_resources(task_id: str, resources: Any) -> None:
    if not isinstance(resources, dict) or set(resources) != {
        "node",
        "gpu_ids",
        "main_process_port",
    }:
        raise ValueError(f"task {task_id!r} resources require exactly node, gpu_ids and main_process_port")
    if resources["node"] not in VALID_NODES:
        raise ValueError(f"task {task_id!r} has invalid resource node")
    gpu_ids = resources["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError(f"task {task_id!r} must request one or more distinct GPU IDs")
    port = resources["main_process_port"]
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError(f"task {task_id!r} has invalid main_process_port")


def load_plan(presets_path: Path, jobs_path: Path) -> dict[str, Any]:
    profile = _load_json(Path(__file__).with_name("training_profile.json"))
    if (
        set(profile)
        != {
            "format",
            "version",
            "memory_mode",
            "global_batch_size",
            "reset_preset_selector_extension",
            "reference_run",
            "policy",
            "training",
            "dataset",
            "adapter_schema",
        }
        or profile["format"] != "lerobot.training_profile"
        or profile["version"] != VERSION
        or profile["memory_mode"] not in VALID_MEMORY_MODES
    ):
        raise ValueError("invalid shared training_profile.json")
    presets_payload = _load_json(presets_path)
    if presets_payload.get("format") != PRESETS_FORMAT or presets_payload.get("version") != VERSION:
        raise ValueError(f"unsupported semantic training presets: {presets_path}")
    presets = presets_payload.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError("semantic training presets require non-empty presets")
    for preset_name, preset in presets.items():
        if not isinstance(preset_name, str) or not preset_name:
            raise ValueError("training preset names and definitions must be non-empty")
        _validate_semantic_definition(f"training preset {preset_name!r}", preset)

    jobs_payload = _load_json(jobs_path)
    if jobs_payload.get("format") != JOBS_FORMAT or jobs_payload.get("version") != VERSION:
        raise ValueError(f"unsupported training jobs: {jobs_path}")
    tasks = jobs_payload.get("jobs")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("training jobs require non-empty jobs")
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            raise ValueError(f"task {task_id!r} must be an object")
        if "memory_mode" in task:
            raise ValueError("Per-task memory_mode is forbidden; use shared training_profile.json")
        if ("dataset_family" in task)==("data_recipe" in task):
            raise ValueError("Choose exactly one data_recipe (or legacy dataset_family alias)")
        common = {"data_recipe" if "data_recipe" in task else "dataset_family", "resources"}
        optional = {
            "domain_weights",
            "instruction_domain_weights",
            "dataset_artifact_id",
            "model_init_checkpoint",
            "reset_preset_sidecar",
            "reset_preset_unknown_target_policy",
            "train_steps",
            "semantic_views_file",
            "allow_resume_semantic_adoption",
            "allow_resume_artifact_adoption",
            "allow_resume_instruction_extension",
            "run_dir_name",
        }
        preset_keys = common | {"training_preset"}
        inline_keys = common | {"state_weights", "state_instruction_matrix"}
        task_keys = frozenset(task)
        if task_keys - optional not in {frozenset(preset_keys), frozenset(inline_keys)}:
            raise ValueError(
                f"task {task_id!r} must use exactly one semantic configuration form: "
                "training_preset, or inline state_weights plus state_instruction_matrix"
            )
        recipe=task.get("data_recipe",task.get("dataset_family"))
        _validate_domain_weights(task.get("domain_weights", {}))
        if task.get("reset_preset_unknown_target_policy", "mask_audited_unknown") not in {"error", "mask_audited_unknown"}:
            raise ValueError("Invalid reset_preset_unknown_target_policy")
        if "dataset_artifact_id" in task and not re.fullmatch(r"[0-9a-f]{64}", task["dataset_artifact_id"]):
            raise ValueError("dataset_artifact_id must be an explicit SHA256 release identity")
        for path_field in ("model_init_checkpoint", "reset_preset_sidecar"):
            if path_field in task and not Path(task[path_field]).is_absolute():
                raise ValueError(f"{path_field} must be an explicit absolute container path")
        if not isinstance(recipe,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",recipe):
            raise ValueError(f"task {task_id!r} has invalid data recipe")
        if not isinstance(task.get("allow_resume_semantic_adoption", False), bool):
            raise ValueError(f"task {task_id!r} allow_resume_semantic_adoption must be boolean")
        if not isinstance(task.get("allow_resume_artifact_adoption", False), bool):
            raise ValueError(f"task {task_id!r} allow_resume_artifact_adoption must be boolean")
        if not isinstance(task.get("allow_resume_instruction_extension", False), bool):
            raise ValueError(f"task {task_id!r} allow_resume_instruction_extension must be boolean")
        if "train_steps" in task and (not isinstance(task["train_steps"], int) or task["train_steps"] <= 0):
            raise ValueError(f"task {task_id!r} train_steps must be a positive integer")
        semantic_views_file = task.get("semantic_views_file", "semantic_views.json")
        if (
            not isinstance(semantic_views_file, str)
            or not semantic_views_file
            or Path(semantic_views_file).name != semantic_views_file
            or not semantic_views_file.endswith(".json")
        ):
            raise ValueError(
                f"task {task_id!r} semantic_views_file must be a JSON filename under dataset meta/"
            )
        run_dir_name = task.get("run_dir_name", "")
        if run_dir_name and (
            not isinstance(run_dir_name, str)
            or Path(run_dir_name).name != run_dir_name
            or "/" in run_dir_name
        ):
            raise ValueError(f"task {task_id!r} run_dir_name must be a single directory name")
        _validate_resources(task_id, task["resources"])
        if "training_preset" in task:
            if task["training_preset"] not in presets:
                raise ValueError(
                    f"task {task_id!r} references unknown training_preset {task['training_preset']!r}"
                )
        else:
            _validate_semantic_definition(
                f"task {task_id!r}",
                {
                    "state_weights": task["state_weights"],
                    "state_instruction_matrix": task["state_instruction_matrix"],
                },
            )
    payload = {"training_presets": presets, "tasks": tasks, "training_profile": profile}
    for task_id in tasks:
        instruction_domain_weights(payload, task_id)
    return payload


def is_instruction_matrix_extension(saved: dict, requested: dict) -> bool:
    """Permit only appended instructions over the same states and unchanged old relations."""
    if not saved or set(saved) != set(requested):
        return False
    added = False
    for state, relations in saved.items():
        if not relations or any(requested[state].get(goal) != rule for goal, rule in relations.items()):
            return False
        added |= bool(set(requested[state]) - set(relations))
    return added


def resolve_task(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    try:
        task=payload["tasks"][task_id]
        return {**task,"dataset_family":task.get("data_recipe",task.get("dataset_family")),
                "semantic_views_file":task.get("semantic_views_file","semantic_views.json"),
                "memory_mode": payload["training_profile"]["memory_mode"]}
    except KeyError as exc:
        raise ValueError(f"unknown task id: {task_id}") from exc


def selected_matrix(payload: dict[str, Any], task_id: str) -> dict[str, dict[str, Any]]:
    task = resolve_task(payload, task_id)
    if "training_preset" in task:
        return payload["training_presets"][task["training_preset"]]["state_instruction_matrix"]
    return task["state_instruction_matrix"]


def state_weights(payload: dict[str, Any], task_id: str) -> dict[str, float]:
    task = resolve_task(payload, task_id)
    if "training_preset" in task:
        return payload["training_presets"][task["training_preset"]]["state_weights"]
    return task["state_weights"]


def domain_weights(payload: dict[str, Any], task_id: str) -> dict[str, float]:
    task = resolve_task(payload, task_id)
    preset = payload.get("training_presets", {}).get(task.get("training_preset"), {})
    return task.get("domain_weights", preset.get("domain_weights", {}))


def instruction_domain_weights(payload: dict[str, Any], task_id: str) -> dict[str, dict[str, float]]:
    task = resolve_task(payload, task_id)
    return validate_instruction_domain_weights(
        task.get("instruction_domain_weights", {}), domain_weights(payload, task_id),
        selected_matrix(payload, task_id),
    )


def _relation_tuple(intent: str, relation: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        intent,
        relation["status"],
        relation["action_supervision"],
        relation["completion_boundary"],
    )


def validate_sidecar(payload: dict[str, Any], task_id: str, sidecar_path: Path) -> dict[str, int]:
    matrix = selected_matrix(payload, task_id)
    sidecar = _load_json(sidecar_path)
    if sidecar.get("format") != "lerobot.semantic_views" or sidecar.get("version") != 1:
        raise ValueError(f"unsupported semantic-view sidecar: {sidecar_path}")
    provenance = sidecar.get("provenance", {})
    if provenance.get("builder") == "build_goal_conditioned_semantic_views.py":
        if provenance.get("selection_policy") != "available_goals_per_valid_stage":
            raise ValueError(
                "Legacy goal sidecar required whole-episode goal coverage; rebuild it from the canonical training sidecar"
            )
        canonical = sidecar_path.parent / "semantic_views.json"
        if (
            canonical.is_file()
            and provenance["source_sha256"] != hashlib.sha256(canonical.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Goal sidecar was not built from the current canonical training sidecar; rebuild it"
            )
    intents = sidecar.get("intents")
    episodes = sidecar.get("episodes")
    if not isinstance(intents, dict) or not isinstance(episodes, dict):
        raise ValueError("semantic-view sidecar requires intents and episodes")

    selected_states = set(state_weights(payload, task_id))
    allowed = {
        state: {_relation_tuple(intent, relation) for intent, relation in matrix[state].items()}
        for state in selected_states
    }
    seen: dict[str, set[tuple[str, str, str, str]]] = {state: set() for state in selected_states}
    domains = domain_weights(payload, task_id)
    instruction_domains = instruction_domain_weights(payload, task_id)
    episode_domains = load_episode_domains(sidecar_path, episodes) if domains else {}
    domain_seen = {d: set() for d, w in domains.items() if w > 0}
    segment_count = 0
    frame_count = 0
    episode_count = 0
    for episode_key, episode in episodes.items():
        if domains:
            if int(episode_key) not in episode_domains:
                raise ValueError(f"Missing recording domain for episode {episode_key}")
            if episode_domains[int(episode_key)] not in domain_seen:
                continue
        episode_count += 1
        if not isinstance(episode, dict):
            raise ValueError(f"episode {episode_key} must be an object")
        views = episode.get("views")
        segments = episode.get("segments", [])
        if not isinstance(views, list) or not isinstance(segments, list):
            raise ValueError(f"episode {episode_key} requires list-valued views and segments")
        for segment in segments:
            state = segment.get("phase")
            if state not in selected_states:
                continue
            start = int(segment["frame_start"])
            stop = int(segment["frame_stop"])
            if not 0 <= start < stop:
                raise ValueError(f"episode {episode_key} state {state} has an invalid frame range")
            enabled: list[tuple[int, int, tuple[str, str, str, str]]] = []
            boundaries = {start, stop}
            for view in views:
                view_start = start if view.get("frame_start") is None else int(view["frame_start"])
                view_stop = stop if view.get("frame_stop") is None else int(view["frame_stop"])
                overlap_start = max(start, view_start)
                overlap_stop = min(stop, view_stop)
                if overlap_start >= overlap_stop:
                    continue
                intent_id = str(view.get("intent_id", ""))
                definition = intents.get(intent_id)
                if not isinstance(definition, dict):
                    raise ValueError(f"episode {episode_key} references unknown intent {intent_id!r}")
                relation_prefix = (
                    str(definition.get("canonical_intent", intent_id)),
                    str(view.get("status", "")),
                    str(view.get("action_supervision", "")),
                )
                completion_frame = view.get("completion_frame")
                matching_relations = [
                    relation
                    for relation in allowed[state]
                    if relation[:3] == relation_prefix
                    and (
                        (relation[3] == "none" and completion_frame is None)
                        or (relation[3] == "state_end" and completion_frame == stop)
                        or (
                            relation[3] == "goal_end"
                            and completion_frame is not None
                            and start < int(completion_frame) <= stop
                        )
                        or (
                            relation[3] == "future_goal_end"
                            and completion_frame is not None
                            and int(completion_frame) > stop
                        )
                    )
                ]
                if len(matching_relations) != 1:
                    if relation_prefix[0] in matrix[state]:
                        raise ValueError(
                            f"task {task_id} has inconsistent relation for episode {episode_key}, state {state}: {view}"
                        )
                    continue
                relation = matching_relations[0]
                enabled.append((overlap_start, overlap_stop, relation))
                boundaries.update((overlap_start, overlap_stop))
                seen[state].add(relation)
                if domains:
                    domain_seen[episode_domains[int(episode_key)]].add((state, relation[0]))
            ordered = sorted(boundaries)
            covered_frames = 0
            for left, right in zip(ordered, ordered[1:]):
                covering = [
                    relation
                    for view_start, view_stop, relation in enabled
                    if view_start <= left and right <= view_stop
                ]
                covering_set = set(covering)
                if len(covering) != len(covering_set):
                    raise ValueError(
                        f"task {task_id} has duplicate state-instruction supervision over "
                        f"episode {episode_key}, state {state}, frames [{left}, {right}); "
                        f"expected={sorted(allowed[state])}, got={sorted(covering)}"
                    )
                if covering:
                    covered_frames += right - left
            segment_count += int(covered_frames > 0)
            frame_count += covered_frames
    missing_states = [state for state in selected_states if not seen[state]]
    if missing_states:
        raise ValueError(f"task {task_id} has no sidecar coverage for states {sorted(missing_states)}")
    for state in selected_states:
        missing = allowed[state] - seen[state]
        if missing:
            raise ValueError(
                f"task {task_id} has configured cells with no sidecar supervision: {state}: {sorted(missing)}"
            )
    cells = {(s, i) for s, row in matrix.items() for i in row}
    for domain, found in domain_seen.items():
        required = {(s, i) for s, i in cells if instruction_domains.get(i, domains).get(domain, 0) > 0}
        if required - found:
            raise ValueError(f"Domain {domain} has no sidecar coverage for cells: {sorted(required - found)}")
    return {"episodes": episode_count, "segments": segment_count, "frames": frame_count}


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def command_resolve(args: argparse.Namespace) -> None:
    payload = load_plan(args.presets, args.jobs)
    task = resolve_task(payload, args.task_id)
    weights = state_weights(payload, args.task_id)
    resources = task["resources"]
    values = (
        task["dataset_family"],
        task["memory_mode"],
        _compact(list(weights)),
        _compact(weights),
        _compact(selected_matrix(payload, args.task_id)),
        resources["node"],
        ",".join(str(gpu_id) for gpu_id in resources["gpu_ids"]),
        str(resources["main_process_port"]),
        str(task.get("train_steps", "")),
        str(task.get("semantic_views_file", "semantic_views.json")),
        "true" if task.get("allow_resume_semantic_adoption", False) else "false",
        str(task.get("run_dir_name", "")),
        "true" if task.get("allow_resume_artifact_adoption", False) else "false",
        "true" if task.get("allow_resume_instruction_extension", False) else "false",
        _compact(domain_weights(payload, args.task_id)),
        _compact(instruction_domain_weights(payload, args.task_id)),
    )
    print("\n".join(values))


def command_assignments(args: argparse.Namespace) -> None:
    payload = load_plan(args.presets, args.jobs)
    for task_id, task in payload["tasks"].items():
        resources = task["resources"]
        gpu_ids = ",".join(str(gpu_id) for gpu_id in resources["gpu_ids"])
        print(f"{task_id}\t{resources['node']}\t{gpu_ids}\t{resources['main_process_port']}")


def command_describe(args: argparse.Namespace) -> None:
    payload = load_plan(args.presets, args.jobs)
    task = resolve_task(payload, args.task_id)
    matrix = selected_matrix(payload, args.task_id)
    weights = state_weights(payload, args.task_id)
    print(f"semantic_config={'preset:' + task['training_preset'] if 'training_preset' in task else 'inline'}")
    resources = task["resources"]
    print(
        f"resources=node:{resources['node']},gpus:"
        f"{','.join(str(gpu_id) for gpu_id in resources['gpu_ids'])},"
        f"port:{resources['main_process_port']}"
    )
    print(f"enabled_states={_compact(list(weights))}")
    print(f"state_weights={_compact(weights)}")
    print(f"domain_weights={_compact(domain_weights(payload, args.task_id))}")
    print(f"instruction_domain_weights={_compact(instruction_domain_weights(payload, args.task_id))}")
    print(f"semantic_views_file={task.get('semantic_views_file', 'semantic_views.json')}")
    print("state_instruction_matrix:")
    for state, instructions in matrix.items():
        for intent, relation in instructions.items():
            print(
                f"  {state} x {intent} -> status={relation['status']}, "
                f"action={relation['action_supervision']}, "
                f"completion={relation['completion_boundary']}"
            )
    if args.sidecar is not None:
        stats = validate_sidecar(payload, args.task_id, args.sidecar)
        print(
            "semantic_sidecar_validation=ok "
            f"episodes={stats['episodes']} selected_segments={stats['segments']} "
            f"selected_frames={stats['frames']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(
        handler=lambda args: print("\n".join(load_plan(args.presets, args.jobs)["tasks"]))
    )
    assignments_parser = subparsers.add_parser("assignments")
    assignments_parser.set_defaults(handler=command_assignments)
    for command, handler in (("resolve", command_resolve), ("describe", command_describe)):
        child = subparsers.add_parser(command)
        child.add_argument("task_id")
        if command == "describe":
            child.add_argument("--sidecar", type=Path)
        child.set_defaults(handler=handler)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
