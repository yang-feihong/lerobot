"""Instruction-relative semantic views over one physical LeRobot dataset.

The sidecar changes neither parquet nor video data.  It declares which
instructions are meaningful for an episode and whether their action target is
the recorded demonstration or a deterministic hold.  Sampling is hierarchical:
the dataloader first chooses a physical frame, this module chooses one view,
then it chooses one language realization.  Adding paraphrases therefore never
changes a physical sample's probability. Matrix-based training chooses the
state/instruction cell before the physical frame and carries that intent through
the dataloader; legacy frame-only sampling remains available without a matrix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.pi05.b2_action_transform import TASK_BLOCKED_KEY
from lerobot.utils.constants import ACTION

FORMAT = "lerobot.semantic_views"
VERSION = 1
VALID_STATUS = {"active", "complete", "blocked"}
VALID_ACTION_SUPERVISION = {"demonstrated", "hold"}


@dataclass(frozen=True)
class SemanticView:
    intent_id: str
    canonical_intent: str
    instructions: tuple[str, ...]
    status: str
    action_supervision: str
    weight: float = 1.0
    # For an active view, keep the physical dataset's terminal completion
    # labels only when they describe this exact instruction.
    completion_source: str = "zero"  # zero | recorded
    frame_start: int | None = None
    frame_stop: int | None = None
    completion_frame: int | None = None
    view_kind: str = "primary"


@dataclass(frozen=True)
class SemanticSegment:
    phase: str
    frame_start: int
    frame_stop: int
    source_group: str = "default"


@dataclass(frozen=True)
class SemanticViewCatalog:
    path: Path
    episodes: dict[int, tuple[SemanticView, ...]]
    sha256: str
    action_stride_dataset_frames: int
    semantic_names: tuple[str, ...]
    selected_episodes: tuple[int, ...] | None
    segments: dict[int, tuple[SemanticSegment, ...]]
    recording_ids: dict[int, str] = field(default_factory=dict)
    # Runtime train-split index only; never written into the dataset sidecar.
    static_first_action_frames: frozenset[tuple[int, int]] | None = None
    recorded_complete_frames: frozenset[tuple[int, int]] = frozenset()
    eligible_start_ranges: dict[int, tuple[tuple[int, int], ...]] | None = None
    physical_segments: dict[int, tuple[SemanticSegment, ...]] | None = None
    valid_frame_ranges: dict[int, tuple[tuple[int, int], ...]] | None = None
    episode_lengths: dict[int, int] | None = None
    future_action_frames: int = 0


def with_semantic_temporal_filter(catalog: SemanticViewCatalog, dataset) -> SemanticViewCatalog:
    """Index legal input/target windows without requiring unrelated stages or goals."""
    reader = dataset.reader
    random_history = reader._random_history_sampling
    random_keys = set() if random_history is None else set(random_history.keys)
    fixed_offsets = [
        offset
        for key, offsets in (reader.delta_indices or {}).items()
        if key not in random_keys
        for offset in offsets
    ]
    past = max(0, -min(fixed_offsets, default=0))
    future = max(
        0,
        max(
            (
                offset
                for key, offsets in (reader.delta_indices or {}).items()
                if key != ACTION and key not in random_keys
                for offset in offsets
            ),
            default=0,
        ),
    )
    action_future = max(0, max((reader.delta_indices or {}).get(ACTION, [0])))
    if random_history is not None:
        past = max(past, random_history.minimum_span_frames)
    valid_ranges = {}
    eligible = {}
    episode_lengths = {}
    for episode in dataset.episodes or range(dataset.meta.total_episodes):
        row = dataset.meta.episodes[episode]
        length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        episode_lengths[episode] = length
        components = []
        physical = catalog.segments if catalog.physical_segments is None else catalog.physical_segments
        for segment in sorted(physical.get(episode, ()), key=lambda s: s.frame_start):
            if not 0 <= segment.frame_start < segment.frame_stop <= length:
                raise ValueError(f"semantic segment exceeds physical episode {episode}")
            if components and segment.frame_start < components[-1][1]:
                raise ValueError(f"Overlapping physical stage ranges in episode {episode}")
            if components and components[-1][1] == segment.frame_start:
                components[-1] = (components[-1][0], segment.frame_stop)
            else:
                components.append((segment.frame_start, segment.frame_stop))
        valid_ranges[episode] = tuple(components)
        eligible[episode] = tuple(
            (left, right)
            for start, stop in components
            for left, right in [(start + (past if start else 0), stop - (future if stop < length else 0))]
            if left < right
        )
    result = replace(
        catalog,
        eligible_start_ranges=eligible,
        valid_frame_ranges=valid_ranges,
        episode_lengths=episode_lengths,
        future_action_frames=action_future,
    )
    reader.set_valid_frame_ranges(valid_ranges, catalog=result)
    return result


def with_incomplete_static_start_filter(catalog: SemanticViewCatalog, dataset) -> SemanticViewCatalog:
    """Index exact-zero B2 + inactive first actions, independently of future/gripper/reset."""
    names = list(dataset.meta.features[ACTION].get("names") or [])
    required = ("b2_vx", "b2_vy", "b2_omega_z", "arm_teleop_inactive", "task_complete")
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"Incomplete static-start filtering requires action fields: {sorted(missing)}")
    b2 = [names.index(name) for name in required[:3]]
    inactive = names.index("arm_teleop_inactive")
    complete = names.index("task_complete")
    static_frames = set()
    complete_frames = set()
    columns = dataset.hf_dataset.select_columns(["episode_index", "frame_index", ACTION]).with_format("numpy")
    for batch in columns.iter(batch_size=65_536):
        actions = np.asarray(batch[ACTION])
        episodes = np.asarray(batch["episode_index"])
        frames = np.asarray(batch["frame_index"])
        static = (actions[:, b2] == 0.0).all(axis=1) & (actions[:, inactive] >= 0.5)
        static_frames.update(zip(episodes[static].tolist(), frames[static].tolist(), strict=True))
        done = actions[:, complete] >= 0.5
        complete_frames.update(zip(episodes[done].tolist(), frames[done].tolist(), strict=True))
    return replace(
        catalog,
        static_first_action_frames=frozenset(static_frames),
        recorded_complete_frames=frozenset(complete_frames),
    )


def semantic_view_start_allowed(
    catalog: SemanticViewCatalog, view: SemanticView, episode: int, frame: int
) -> bool:
    """Decide eligibility from this instruction's final first-step supervision."""
    if catalog.eligible_start_ranges is not None and not any(
        start <= frame < stop for start, stop in catalog.eligible_start_ranges.get(episode, ())
    ):
        return False
    complete = (
        view.status == "complete"
        or (view.completion_frame is not None and frame >= view.completion_frame)
        or (
            view.completion_frame is None
            and view.completion_source == "recorded"
            and (episode, frame) in catalog.recorded_complete_frames
        )
    )
    if catalog.valid_frame_ranges is not None:
        component_stop = next(
            stop for start, stop in catalog.valid_frame_ranges[episode] if start <= frame < stop
        )
        last_teacher = (
            frame if complete or view.action_supervision == "hold" else frame + catalog.future_action_frames
        )
        if view.completion_frame is not None and not complete:
            last_teacher = min(last_teacher, view.completion_frame - 1)
        if component_stop < catalog.episode_lengths[episode] and last_teacher >= component_stop:
            return False
    if catalog.static_first_action_frames is None or complete:
        return True
    return view.action_supervision != "hold" and (episode, frame) not in catalog.static_first_action_frames


def semantic_sample_views(catalog, episode, frame, state_instruction_matrix=None):
    """Shared pre-sampling and materialization eligibility, with schema errors kept explicit."""
    views = tuple(
        view
        for view in catalog.episodes.get(episode, ())
        if view.frame_start is None or view.frame_start <= frame < view.frame_stop
    )
    if state_instruction_matrix:
        state = semantic_state_for_frame(catalog, episode, frame, state_instruction_matrix)
        views = tuple(v for v in views if semantic_view_matches_relation(v, state, state_instruction_matrix))
    return tuple(v for v in views if semantic_view_start_allowed(catalog, v, episode, frame))


def load_semantic_views(path: str | Path) -> SemanticViewCatalog:
    path = Path(path).expanduser().resolve()
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError(f"Unsupported semantic-view sidecar: {path}")
    intents = payload.get("intents")
    episodes = payload.get("episodes")
    if not isinstance(intents, dict) or not isinstance(episodes, dict):
        raise ValueError("semantic-view sidecar requires object-valued intents and episodes")
    parsed: dict[int, tuple[SemanticView, ...]] = {}
    parsed_segments: dict[int, tuple[SemanticSegment, ...]] = {}
    for episode_key, episode in episodes.items():
        views_raw = episode.get("views") if isinstance(episode, dict) else None
        if not isinstance(views_raw, list) or not views_raw:
            raise ValueError(f"episode {episode_key} has no semantic views")
        views: list[SemanticView] = []
        for raw in views_raw:
            intent_id = str(raw.get("intent_id", ""))
            definition = intents.get(intent_id)
            instructions = definition.get("instructions") if isinstance(definition, dict) else None
            status = str(raw.get("status", ""))
            supervision = str(raw.get("action_supervision", ""))
            completion_source = str(raw.get("completion_source", "zero"))
            weight = float(raw.get("weight", 1.0))
            if (
                not isinstance(instructions, list)
                or not instructions
                or not all(isinstance(item, str) and item.strip() for item in instructions)
            ):
                raise ValueError(f"intent {intent_id!r} has no usable instructions")
            if status not in VALID_STATUS or supervision not in VALID_ACTION_SUPERVISION:
                raise ValueError(f"invalid view status/supervision in episode {episode_key}: {raw}")
            if completion_source not in {"zero", "recorded"} or weight <= 0:
                raise ValueError(f"invalid completion source/weight in episode {episode_key}: {raw}")
            if status != "active" and supervision != "hold":
                raise ValueError("complete/blocked views must use deterministic hold supervision")
            if status != "active" and completion_source != "zero":
                raise ValueError("only active views may inherit recorded completion labels")
            frame_start = raw.get("frame_start")
            frame_stop = raw.get("frame_stop")
            completion_frame = raw.get("completion_frame")
            if (frame_start is None) != (frame_stop is None):
                raise ValueError("frame_start and frame_stop must be configured together")
            if frame_start is not None and not 0 <= int(frame_start) < int(frame_stop):
                raise ValueError(f"invalid view frame range in episode {episode_key}: {raw}")
            views.append(
                SemanticView(
                    intent_id=intent_id,
                    canonical_intent=str(definition.get("canonical_intent", intent_id)),
                    instructions=tuple(item.strip() for item in instructions),
                    status=status,
                    action_supervision=supervision,
                    weight=weight,
                    completion_source=completion_source,
                    frame_start=None if frame_start is None else int(frame_start),
                    frame_stop=None if frame_stop is None else int(frame_stop),
                    completion_frame=None if completion_frame is None else int(completion_frame),
                    view_kind=str(raw.get("view_kind", "primary")),
                )
            )
        parsed[int(episode_key)] = tuple(views)
        segments_raw = episode.get("segments", [])
        if not isinstance(segments_raw, list):
            raise ValueError(f"episode {episode_key} segments must be a list")
        segments: list[SemanticSegment] = []
        for raw in segments_raw:
            phase = str(raw.get("phase", "")).strip()
            frame_start = int(raw.get("frame_start", -1))
            frame_stop = int(raw.get("frame_stop", -1))
            source_group = str(raw.get("source_group", "default")).strip()
            if not phase or not source_group or not 0 <= frame_start < frame_stop:
                raise ValueError(f"invalid semantic segment in episode {episode_key}: {raw}")
            segments.append(SemanticSegment(phase, frame_start, frame_stop, source_group))
        ordered = sorted(segments, key=lambda item: (item.frame_start, item.frame_stop, item.phase))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.frame_start < previous.frame_stop:
                raise ValueError(f"semantic segments overlap in episode {episode_key}")
        parsed_segments[int(episode_key)] = tuple(ordered)
    stride = int(payload.get("action_stride_dataset_frames", 1))
    if stride < 1:
        raise ValueError("action_stride_dataset_frames must be positive")
    semantic_names = tuple(
        sorted({f"{view.canonical_intent}/{view.status}" for views in parsed.values() for view in views})
    )
    selected_raw = payload.get("selected_episodes")
    selected_episodes = None if selected_raw is None else tuple(int(item) for item in selected_raw)
    if selected_episodes is not None:
        if len(set(selected_episodes)) != len(selected_episodes):
            raise ValueError("selected_episodes contains duplicates")
        if set(selected_episodes) != set(parsed):
            raise ValueError("selected_episodes must exactly match semantic-view episode keys")
    physical_segments = None
    boundaries_path = path.parent / "stage_boundaries.json"
    if boundaries_path.is_file():
        boundaries = json.loads(boundaries_path.read_text())
        physical_segments = {}
        for episode_key, entry in boundaries["episodes"].items():
            physical_segments[int(episode_key)] = tuple(
                SemanticSegment(
                    phase,
                    int(definition["frame_start"]),
                    int(definition["frame_stop"]),
                    str(definition["source_group"]),
                )
                for phase, definition in entry["phases"].items()
                if definition["selected"]
            )
    return SemanticViewCatalog(
        path,
        parsed,
        hashlib.sha256(payload_bytes).hexdigest(),
        stride,
        semantic_names,
        selected_episodes,
        parsed_segments,
        recording_ids={
            int(key): episode["recording_id"] if "recording_id" in episode else episode["source_path"]
            for key, episode in episodes.items()
            if "recording_id" in episode or "source_path" in episode
        },
        physical_segments=physical_segments,
    )


def semantic_phase_sampling_groups(
    catalog: SemanticViewCatalog,
    episode_from_indices: list[int],
    selected_episodes: list[int],
    *,
    phases: list[str],
    phase_weights: dict[str, float] | None = None,
    source_weights: dict[str, float] | None = None,
    state_instruction_matrix: dict[str, dict[str, dict[str, str]]] | None = None,
    view_kind_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[float], list[str]]:
    """Resolve configured semantic segments into absolute frame pools.

    Sampling is hierarchical: phase weights choose a phase.  By default all
    physical sources in that phase form one frame-uniform pool.  Explicit source
    weights split the phase into source groups and control their conditional
    mixture.  This keeps phase composition independent of segment duration and
    avoids materializing phase-specific datasets.
    """
    if not phases:
        raise ValueError("semantic phase sampling requires at least one phase")
    if len(set(phases)) != len(phases):
        raise ValueError("semantic phases contain duplicates")
    phase_weight_map = phase_weights or dict.fromkeys(phases, 1.0)
    if set(phase_weight_map) != set(phases):
        raise ValueError("semantic phase weights must exactly match selected phases")
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in phase_weight_map.values()):
        raise ValueError("semantic phase weights must be finite and positive")
    source_weight_map = source_weights or {}
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in source_weight_map.values()):
        raise ValueError("semantic source weights must be finite and positive")

    pools: dict[tuple[str, str], list[np.ndarray]] = {}
    selected = {int(value) for value in selected_episodes}
    for episode in selected:
        segments = catalog.segments.get(episode, ())
        for segment in segments:
            if segment.phase not in phase_weight_map:
                continue
            offset = int(episode_from_indices[episode])
            frames = np.arange(segment.frame_start, segment.frame_stop, dtype=np.int64)
            frames = np.asarray(
                [
                    frame
                    for frame in frames
                    if any(
                        view.weight
                        * (
                            1.0
                            if state_instruction_matrix
                            else _view_kind_weight(view_kind_weights, view.view_kind)
                        )
                        > 0
                        for view in semantic_sample_views(
                            catalog, episode, int(frame), state_instruction_matrix
                        )
                    )
                ],
                dtype=np.int64,
            )
            if len(frames):
                pools.setdefault((segment.phase, segment.source_group), []).append(offset + frames)
    missing = [phase for phase in phases if not any(key[0] == phase for key in pools)]
    if missing:
        raise ValueError(f"selected semantic phases have no frames: {missing}")
    available_sources = {source for _, source in pools}
    if source_weight_map and set(source_weight_map) != available_sources:
        raise ValueError(
            "semantic source weights must exactly match selected source groups: "
            f"configured={sorted(source_weight_map)} available={sorted(available_sources)}"
        )

    phase_total = float(sum(phase_weight_map.values()))
    groups: list[np.ndarray] = []
    weights: list[float] = []
    names: list[str] = []
    for phase in phases:
        source_names = sorted(source for candidate_phase, source in pools if candidate_phase == phase)
        if not source_weight_map:
            groups.append(
                np.concatenate([part for source in source_names for part in pools[(phase, source)]])
            )
            weights.append(float(phase_weight_map[phase] / phase_total))
            names.append(f"{phase}/*")
            continue
        raw_source_weights = np.asarray(
            [source_weight_map[source] for source in source_names], dtype=np.float64
        )
        raw_source_weights /= raw_source_weights.sum()
        for source, conditional_weight in zip(source_names, raw_source_weights, strict=True):
            groups.append(np.concatenate(pools[(phase, source)]))
            weights.append(float(phase_weight_map[phase] / phase_total * conditional_weight))
            names.append(f"{phase}/{source}")
    eligible = np.unique(np.concatenate(groups))
    if len(eligible) != sum(len(group) for group in groups):
        raise ValueError("selected semantic phase/source groups overlap")
    return eligible, groups, weights, names


def _choice_index(weights: list[float], token: bytes) -> int:
    point = int.from_bytes(token, "big") / float(1 << (8 * len(token))) * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if point < cumulative:
            return index
    return len(weights) - 1


@dataclass(frozen=True)
class SemanticRelationSampling:
    eligible_frames: np.ndarray
    frame_groups: list[np.ndarray]
    group_weights: list[float]
    states: list[str]
    intents: list[str]
    names: list[str]
    frame_weights: list[np.ndarray]
    episode_from_indices: np.ndarray


def semantic_relation_sampling_groups(
    catalog,
    episode_from_indices,
    selected_episodes,
    *,
    state_instruction_matrix,
    phase_weights=None,
    source_weights=None,
    interior_static_frames=(),
    terminal_static_frames=(),
    interior_static_weight=1.0,
    terminal_static_weight=1.0,
) -> SemanticRelationSampling:
    """Choose a state/instruction cell first, independently of its eligible frame count."""
    phase_weights = phase_weights or dict.fromkeys(state_instruction_matrix, 1.0)
    source_weights = source_weights or {}
    if set(phase_weights) != set(state_instruction_matrix):
        raise ValueError("State weights must match the configured relation matrix")
    if any(not np.isfinite(w) or w <= 0 for w in (*phase_weights.values(), *source_weights.values())):
        raise ValueError("Relation/source weights must be finite and positive")
    if (
        not 0 <= interior_static_weight <= 1
        or not np.isfinite(terminal_static_weight)
        or terminal_static_weight <= 0
    ):
        raise ValueError("Invalid static-horizon weights")
    interior, terminal = set(interior_static_frames), set(terminal_static_frames)
    pools = {}
    for episode in selected_episodes:
        offset = int(episode_from_indices[episode])
        for segment in catalog.segments.get(episode, ()):
            if segment.phase not in state_instruction_matrix:
                continue
            for view in catalog.episodes.get(episode, ()):
                if view.weight <= 0 or not semantic_view_matches_relation(
                    view, segment, state_instruction_matrix
                ):
                    continue
                start = max(
                    segment.frame_start, segment.frame_start if view.frame_start is None else view.frame_start
                )
                stop = min(
                    segment.frame_stop, segment.frame_stop if view.frame_stop is None else view.frame_stop
                )
                frames, weights = [], []
                for frame in range(start, stop):
                    if not semantic_view_start_allowed(catalog, view, episode, frame):
                        continue
                    absolute = offset + frame
                    complete = view.status == "complete" or (
                        view.completion_frame is not None and frame >= view.completion_frame
                    )
                    weight = (
                        terminal_static_weight
                        if absolute in terminal
                        else (interior_static_weight if absolute in interior and not complete else 1.0)
                    )
                    if weight > 0:
                        frames.append(absolute)
                        weights.append(weight)
                if frames:
                    key = (segment.phase, view.canonical_intent, segment.source_group)
                    pools.setdefault(key, []).append(
                        (np.asarray(frames, dtype=np.int64), np.asarray(weights))
                    )
    groups, frame_weights, group_weights, states, intents, names = [], [], [], [], [], []
    if source_weights and set(source_weights) != {source for _, _, source in pools}:
        raise ValueError("Source weights must match eligible physical source groups")
    phase_total = sum(
        phase_weights[state] * len(instructions) for state, instructions in state_instruction_matrix.items()
    )
    for state, instructions in state_instruction_matrix.items():
        for intent in instructions:
            sources = sorted(source for phase, goal, source in pools if phase == state and goal == intent)
            if not sources:
                raise ValueError(
                    f"Configured state/instruction cell has no eligible samples: {state} x {intent}"
                )
            source_sets = [[source] for source in sources] if source_weights else [sources]
            source_masses = [source_weights[source] for source in sources] if source_weights else [1.0]
            for sources_in_group, mass in zip(source_sets, source_masses, strict=True):
                parts = [part for source in sources_in_group for part in pools[(state, intent, source)]]
                frames = np.concatenate([part[0] for part in parts])
                weights = np.concatenate([part[1] for part in parts])
                order = np.argsort(frames)
                frames, weights = frames[order], weights[order]
                if len(np.unique(frames)) != len(frames):
                    raise ValueError(f"Duplicate supervision in {state} x {intent}")
                groups.append(frames)
                frame_weights.append(weights)
                group_weights.append(phase_weights[state] / phase_total * mass / sum(source_masses))
                states.append(state)
                intents.append(intent)
                names.append(f"{state}/{intent}/" + ("+".join(sources_in_group) if source_weights else "*"))
    return SemanticRelationSampling(
        np.unique(np.concatenate(groups)),
        groups,
        group_weights,
        states,
        intents,
        names,
        frame_weights,
        np.asarray(episode_from_indices),
    )


def semantic_pair_probabilities(plan, state, intent, absolute_frames):
    result = np.zeros(len(absolute_frames), dtype=np.float64)
    for phase, goal, frames, weights, mass in zip(
        plan.states, plan.intents, plan.frame_groups, plan.frame_weights, plan.group_weights, strict=True
    ):
        if (phase, goal) != (state, intent):
            continue
        positions = np.searchsorted(frames, absolute_frames)
        safe = np.minimum(positions, len(frames) - 1)
        present = (positions < len(frames)) & (frames[safe] == absolute_frames)
        result[present] += mass * weights[safe[present]] / weights.sum()
    return result


def _relation_status_priors(catalog, plan, horizon):
    result = {"task_complete": 0.0, "task_blocked": 0.0}
    for intent, frames, weights, mass in zip(
        plan.intents, plan.frame_groups, plan.frame_weights, plan.group_weights, strict=True
    ):
        episodes = np.searchsorted(plan.episode_from_indices, frames, side="right") - 1
        for episode in np.unique(episodes):
            selected = episodes == episode
            local = frames[selected] - plan.episode_from_indices[episode]
            probabilities = mass * weights[selected] / weights.sum()
            covered = np.zeros(len(local), dtype=np.int64)
            for view in catalog.episodes[int(episode)]:
                if view.canonical_intent != intent:
                    continue
                mask = (
                    np.ones(len(local), dtype=bool)
                    if view.frame_start is None
                    else (local >= view.frame_start) & (local < view.frame_stop)
                )
                covered += mask
                if view.status == "complete":
                    fraction = np.ones(mask.sum())
                elif view.completion_frame is not None:
                    distance = np.maximum(0, view.completion_frame - local[mask])
                    first = (
                        distance + catalog.action_stride_dataset_frames - 1
                    ) // catalog.action_stride_dataset_frames
                    fraction = 1.0 - np.minimum(first, horizon) / horizon
                else:
                    fraction = np.zeros(mask.sum())
                result["task_complete"] += float(np.dot(probabilities[mask], fraction))
                if view.status == "blocked":
                    result["task_blocked"] += float(probabilities[mask].sum())
            if not np.all(covered == 1):
                raise ValueError(
                    "Relation sampling plan must identify exactly one view per frame/instruction"
                )
    return result


def _view_kind_weight(view_kind_weights: dict[str, float] | None, view_kind: str) -> float:
    """Resolve one view-kind weight without silently enabling omitted kinds.

    An empty mapping means that no filtering was requested and preserves the
    catalog weights.  Once a mapping is supplied, it is an allowlist: omitted
    kinds are disabled.  This makes a configuration such as {"primary": 1.0}
    mean exactly primary-only.
    """
    if not view_kind_weights:
        return 1.0
    return float(view_kind_weights.get(view_kind, 0.0))


def semantic_state_for_frame(
    catalog: SemanticViewCatalog,
    episode: int,
    frame: int,
    state_instruction_matrix: dict[str, dict[str, dict[str, str]]],
) -> SemanticSegment:
    """Return the unique configured physical state containing one frame."""
    matches = [
        segment
        for segment in catalog.segments.get(episode, ())
        if segment.phase in state_instruction_matrix and segment.frame_start <= frame < segment.frame_stop
    ]
    if len(matches) != 1:
        raise ValueError(
            "semantic state-instruction matrix requires exactly one configured state at "
            f"episode={episode}, frame={frame}; matches={[item.phase for item in matches]}"
        )
    return matches[0]


def semantic_view_matches_relation(
    view: SemanticView,
    state: SemanticSegment,
    state_instruction_matrix: dict[str, dict[str, dict[str, str]]],
) -> bool:
    """Match a concrete view against an explicit state-instruction cell."""
    relation = state_instruction_matrix[state.phase].get(view.canonical_intent)
    if relation is None:
        return False
    requested_boundary = relation["completion_boundary"]
    if requested_boundary == "none":
        completion_matches = view.completion_frame is None
    elif requested_boundary == "state_end":
        completion_matches = view.completion_frame == state.frame_stop
    elif requested_boundary == "goal_end":
        completion_matches = (
            view.completion_frame is not None
            and state.frame_start < view.completion_frame <= state.frame_stop
        )
    elif requested_boundary == "future_goal_end":
        completion_matches = view.completion_frame is not None and view.completion_frame > state.frame_stop
    else:
        raise ValueError(f"unsupported completion boundary: {requested_boundary}")
    return (
        view.status == relation["status"]
        and view.action_supervision == relation["action_supervision"]
        and completion_matches
    )


def _hold_action(action: torch.Tensor) -> torch.Tensor:
    if action.ndim != 2 or action.shape[-1] != 25:
        raise ValueError(f"semantic hold expects one 25D action chunk, got {tuple(action.shape)}")
    held = action.clone()
    held[:, :3] = 0.0
    held[:, 3] = 1.0
    held[:, 4] = 0.0
    # Keep the current executed target and gripper command for the whole chunk.
    held[:, 5:14] = action[0, 16:25]
    held[:, 14] = action[0, 14]
    held[:, 16:25] = action[0, 16:25]
    return held


def _hold_action_tail(action: torch.Tensor, start: int) -> torch.Tensor:
    """Replace the post-completion suffix by a hold at the final active target."""
    if not 0 < start < action.shape[0]:
        return action
    result = action.clone()
    anchor = action[start - 1]
    result[start:, :3] = 0.0
    result[start:, 3] = 1.0
    result[start:, 4] = 0.0
    result[start:, 5:14] = anchor[16:25]
    result[start:, 14] = anchor[14]
    result[start:, 16:25] = anchor[16:25]
    return result


def apply_semantic_views_to_batch(
    batch: dict[str, Any],
    catalog: SemanticViewCatalog,
    *,
    step: int,
    seed: int,
    randomize: bool,
    view_kind_weights: dict[str, float] | None = None,
    state_instruction_matrix: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict[str, int]:
    """Materialize one deterministic semantic view for each physical sample."""
    episode_values = torch.as_tensor(batch["episode_index"]).reshape(-1).tolist()
    frame_values = torch.as_tensor(batch["frame_index"]).reshape(-1).tolist()
    sample_values = torch.as_tensor(batch.get("index", range(len(episode_values)))).reshape(-1).tolist()
    actions = batch.get(ACTION)
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise ValueError("semantic views require a batched temporal action tensor")
    tasks = list(batch.get("task", [""] * len(episode_values)))
    sampled_intents = batch.pop("semantic_sampling_intent", None)
    blocked = torch.zeros(actions.shape[:2], dtype=actions.dtype, device=actions.device)
    counts = {"active": 0, "complete": 0, "blocked": 0, "hold": 0}
    semantic_names: list[str] = []
    for row, (episode, frame, sample) in enumerate(
        zip(episode_values, frame_values, sample_values, strict=True)
    ):
        episode_views = catalog.episodes.get(int(episode))
        views = tuple(
            view
            for view in episode_views or ()
            if view.frame_start is None or view.frame_start <= int(frame) < view.frame_stop
        )
        if state_instruction_matrix:
            state = semantic_state_for_frame(
                catalog,
                int(episode),
                int(frame),
                state_instruction_matrix,
            )
            views = tuple(
                view
                for view in views
                if semantic_view_matches_relation(view, state, state_instruction_matrix)
            )
            if not views:
                raise ValueError(
                    "semantic state-instruction matrix requires at least one view at "
                    f"episode={int(episode)}, frame={int(frame)}"
                )
        if not views:
            raise ValueError(
                "semantic-view sidecar has no enabled state-instruction relation for "
                f"episode={int(episode)}, frame={int(frame)}"
            )
        views = tuple(
            view
            for view in views
            if semantic_view_start_allowed(catalog, view, int(episode), int(frame))
            and (sampled_intents is None or view.canonical_intent == sampled_intents[row])
        )
        if not views:
            raise ValueError(
                f"Sampler selected an excluded incomplete static start: episode={episode}, frame={frame}"
            )
        digest = hashlib.blake2b(
            f"{seed}:{step}:{int(sample)}:{int(episode)}".encode(), digest_size=16
        ).digest()
        weights = (
            [view.weight for view in views]
            if state_instruction_matrix
            else [view.weight * _view_kind_weight(view_kind_weights, view.view_kind) for view in views]
        )
        if any(weight < 0 for weight in weights) or not any(weight > 0 for weight in weights):
            raise ValueError(f"semantic view weights disable every view for episode={episode} frame={frame}")
        view_index = (
            next(index for index, weight in enumerate(weights) if weight > 0)
            if not randomize
            else _choice_index(weights, digest[:8])
        )
        view = views[view_index]
        semantic_names.append(f"{view.canonical_intent}/{view.status}")
        language_index = 0 if not randomize else int.from_bytes(digest[8:], "big") % len(view.instructions)
        tasks[row] = view.instructions[language_index]
        if view.action_supervision == "hold":
            actions[row] = _hold_action(actions[row])
            counts["hold"] += 1
        if view.status == "complete":
            actions[row, :, 15] = 1.0
        elif view.completion_frame is not None:
            action_frames = (
                int(frame)
                + torch.arange(actions.shape[1], device=actions.device) * catalog.action_stride_dataset_frames
            )
            completion_mask = action_frames >= view.completion_frame
            actions[row, :, 15] = completion_mask.to(actions.dtype)
            first_complete = torch.nonzero(completion_mask, as_tuple=False)
            if len(first_complete):
                first_complete_index = int(first_complete[0])
                actions[row] = (
                    _hold_action(actions[row])
                    if first_complete_index == 0
                    else _hold_action_tail(actions[row], first_complete_index)
                )
        elif view.completion_source != "recorded":
            actions[row, :, 15] = 0.0
        if view.status == "blocked":
            blocked[row] = 1.0
        counts[view.status] += 1
    batch["task"] = tasks
    batch[TASK_BLOCKED_KEY] = blocked
    batch["semantic_view_name"] = semantic_names
    return counts


def semantic_status_priors(
    catalog: SemanticViewCatalog,
    episode_lengths: dict[int, int],
    *,
    action_horizon: int = 1,
    view_kind_weights: dict[str, float] | None = None,
    state_instruction_matrix: dict[str, dict[str, dict[str, str]]] | None = None,
    phases: list[str] | None = None,
    phase_weights: dict[str, float] | None = None,
    source_weights: dict[str, float] | None = None,
    relation_sampling: SemanticRelationSampling | None = None,
) -> dict[str, float]:
    """Return expected per-action-element status priors under view sampling."""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if relation_sampling is not None:
        return _relation_status_priors(catalog, relation_sampling, action_horizon)
    group_expected: dict[tuple[str, str], dict[str, float]] = {}
    group_frames: dict[tuple[str, str], int] = {}
    for episode, length in episode_lengths.items():
        if phases:
            segments = [segment for segment in catalog.segments.get(episode, ()) if segment.phase in phases]
            frame_groups = [
                (frame, (segment.phase, segment.source_group))
                for segment in segments
                for frame in range(segment.frame_start, segment.frame_stop)
            ]
        else:
            frame_groups = [(frame, ("__all__", "__all__")) for frame in range(length)]
        for frame, group in frame_groups:
            views = [
                view
                for view in catalog.episodes.get(episode, ())
                if view.frame_start is None or view.frame_start <= frame < view.frame_stop
            ]
            if state_instruction_matrix:
                state = semantic_state_for_frame(
                    catalog,
                    episode,
                    frame,
                    state_instruction_matrix,
                )
                views = [
                    view
                    for view in views
                    if semantic_view_matches_relation(view, state, state_instruction_matrix)
                ]
                weights = [view.weight for view in views]
            else:
                weights = [
                    view.weight * _view_kind_weight(view_kind_weights, view.view_kind) for view in views
                ]
            allowed = [
                (view, weight)
                for view, weight in zip(views, weights, strict=True)
                if semantic_view_start_allowed(catalog, view, episode, frame)
            ]
            if not allowed:
                continue
            views, weights = zip(*allowed, strict=True)
            total = sum(weights)
            if total <= 0:
                raise ValueError(f"no enabled semantic view for episode={episode} frame={frame}")
            complete_mass = 0.0
            for view, weight in zip(views, weights, strict=True):
                if view.status == "complete":
                    positive_fraction = 1.0
                elif view.completion_frame is not None:
                    delta = view.completion_frame - frame
                    first_positive = (
                        0
                        if delta <= 0
                        else (delta + catalog.action_stride_dataset_frames - 1)
                        // catalog.action_stride_dataset_frames
                    )
                    positive_fraction = (
                        action_horizon - min(first_positive, action_horizon)
                    ) / action_horizon
                else:
                    positive_fraction = 0.0
                complete_mass += weight * positive_fraction
            accumulator = group_expected.setdefault(group, {"task_complete": 0.0, "task_blocked": 0.0})
            accumulator["task_complete"] += complete_mass / total
            accumulator["task_blocked"] += (
                sum(weight for view, weight in zip(views, weights, strict=True) if view.status == "blocked")
                / total
            )
            group_frames[group] = group_frames.get(group, 0) + 1
    if not group_frames:
        raise ValueError("semantic status priors require at least one physical frame")
    if not phases:
        frames = group_frames[("__all__", "__all__")]
        return {name: value / frames for name, value in group_expected[("__all__", "__all__")].items()}

    configured_phase_weights = phase_weights or dict.fromkeys(phases, 1.0)
    phase_total = float(sum(configured_phase_weights.values()))
    available_sources = {group[1] for group in group_frames}
    if source_weights and set(source_weights) != available_sources:
        raise ValueError(
            "semantic source weights must exactly match selected source groups: "
            f"configured={sorted(source_weights)} available={sorted(available_sources)}"
        )
    result = {"task_complete": 0.0, "task_blocked": 0.0}
    for phase in phases:
        groups = [group for group in group_frames if group[0] == phase]
        if source_weights:
            raw_source_weights = np.asarray([source_weights[group[1]] for group in groups], dtype=np.float64)
        else:
            raw_source_weights = np.asarray([group_frames[group] for group in groups], dtype=np.float64)
        raw_source_weights /= raw_source_weights.sum()
        for group, conditional_weight in zip(groups, raw_source_weights, strict=True):
            group_weight = configured_phase_weights[phase] / phase_total * conditional_weight
            for name in result:
                result[name] += group_weight * group_expected[group][name] / group_frames[group]
    return result
