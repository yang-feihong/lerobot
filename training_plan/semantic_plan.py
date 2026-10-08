#!/usr/bin/env python3
"""Resolve training jobs against maintained state-instruction presets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRESETS_FORMAT = "lerobot.semantic_training_presets"
JOBS_FORMAT = "lerobot.training_jobs"
VERSION = 1
VALID_DATASET_FAMILIES = {"staff1", "all_scenes"}
VALID_MEMORY_MODES = {"standard", "mem_vit", "full_mem"}
VALID_STATUSES = {"active", "complete", "blocked"}
VALID_ACTION_SUPERVISION = {"demonstrated", "hold"}
VALID_COMPLETION_BOUNDARIES = {"none", "state_end"}
VALID_NODES = {"f", "wb2", "wb3", "wb4", "wb5"}


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
    if set(definition) != {"state_weights", "state_instruction_matrix"}:
        raise ValueError(f"{name} requires state_weights and state_instruction_matrix")
    weights = definition["state_weights"]
    matrix = definition["state_instruction_matrix"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{name} requires explicit state_weights")
    if not isinstance(matrix, dict) or not matrix:
        raise ValueError(f"{name} requires an explicit state_instruction_matrix")
    if set(weights) != set(matrix):
        raise ValueError(f"{name} state weights and matrix rows must exactly match")
    if any(
        not isinstance(state, str)
        or not state
        or not isinstance(weight, (int, float))
        or weight <= 0
        for state, weight in weights.items()
    ):
        raise ValueError(f"{name} has invalid state_weights")
    for state, instructions in matrix.items():
        if not isinstance(instructions, dict) or not instructions:
            raise ValueError(f"{name} state {state!r} requires instructions")
        for intent, relation in instructions.items():
            _validate_relation(state, intent, relation)


def _validate_resources(task_id: str, resources: Any) -> None:
    if not isinstance(resources, dict) or set(resources) != {
        "node",
        "gpu_ids",
        "main_process_port",
    }:
        raise ValueError(
            f"task {task_id!r} resources require exactly node, gpu_ids and main_process_port"
        )
    if resources["node"] not in VALID_NODES:
        raise ValueError(f"task {task_id!r} has invalid resource node")
    gpu_ids = resources["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or len(gpu_ids) != 4
        or any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError(f"task {task_id!r} must request exactly four distinct GPU IDs")
    port = resources["main_process_port"]
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError(f"task {task_id!r} has invalid main_process_port")


def load_plan(presets_path: Path, jobs_path: Path) -> dict[str, Any]:
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
        common = {"dataset_family", "memory_mode", "resources"}
        preset_keys = common | {"training_preset"}
        inline_keys = common | {"state_weights", "state_instruction_matrix"}
        if frozenset(task) not in {frozenset(preset_keys), frozenset(inline_keys)}:
            raise ValueError(
                f"task {task_id!r} must use exactly one semantic configuration form: "
                "training_preset, or inline state_weights plus state_instruction_matrix"
            )
        if task["dataset_family"] not in VALID_DATASET_FAMILIES:
            raise ValueError(f"task {task_id!r} has invalid dataset_family")
        if task["memory_mode"] not in VALID_MEMORY_MODES:
            raise ValueError(f"task {task_id!r} has invalid memory_mode")
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
    return {"training_presets": presets, "tasks": tasks}


def resolve_task(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    try:
        return payload["tasks"][task_id]
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
    intents = sidecar.get("intents")
    episodes = sidecar.get("episodes")
    if not isinstance(intents, dict) or not isinstance(episodes, dict):
        raise ValueError("semantic-view sidecar requires intents and episodes")

    selected_states = set(state_weights(payload, task_id))
    allowed = {
        state: {_relation_tuple(intent, relation) for intent, relation in matrix[state].items()}
        for state in selected_states
    }
    seen: dict[str, set[tuple[str, str, str, str]]] = {
        state: set() for state in selected_states
    }
    segment_count = 0
    frame_count = 0
    for episode_key, episode in episodes.items():
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
                relation = (
                    str(definition.get("canonical_intent", intent_id)),
                    str(view.get("status", "")),
                    str(view.get("action_supervision", "")),
                    "state_end" if view.get("completion_frame") == stop else "none",
                )
                if relation not in allowed[state]:
                    continue
                enabled.append((overlap_start, overlap_stop, relation))
                boundaries.update((overlap_start, overlap_stop))
                seen[state].add(relation)
            ordered = sorted(boundaries)
            for left, right in zip(ordered, ordered[1:], strict=False):
                covering = [
                    relation
                    for view_start, view_stop, relation in enabled
                    if view_start <= left and right <= view_stop
                ]
                if len(covering) != 1:
                    raise ValueError(
                        f"task {task_id} requires exactly one configured state-instruction relation over "
                        f"episode {episode_key}, state {state}, frames [{left}, {right}); got {covering}"
                    )
            segment_count += 1
            frame_count += stop - start
    missing_states = [state for state in selected_states if not seen[state]]
    if missing_states:
        raise ValueError(f"task {task_id} has no sidecar coverage for states {sorted(missing_states)}")
    for state in selected_states:
        missing_relations = allowed[state] - seen[state]
        if missing_relations:
            raise ValueError(
                f"task {task_id} never observes configured relations for {state}: {sorted(missing_relations)}"
            )
    return {"episodes": len(episodes), "segments": segment_count, "frames": frame_count}


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
