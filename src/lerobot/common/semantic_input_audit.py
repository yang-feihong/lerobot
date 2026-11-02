"""Bounded evidence from actual sampled, tokenized training inputs."""

import json

from lerobot.datasets.semantic_views import semantic_state_for_frame
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class SemanticInputAudit:
    def __init__(self, catalog, matrix, tokenizer, path, *, continuous_state_memory=False):
        self.catalog, self.matrix, self.tokenizer, self.path = catalog, matrix, tokenizer, path
        self.continuous_state_memory = continuous_state_memory
        self.examples = {}
        self.complete = False

    def record(self, batch, original_tasks, step):
        if self.complete:
            return
        changed = False
        for row, original in enumerate(original_tasks):
            episode, frame = int(batch["episode_index"][row]), int(batch["frame_index"][row])
            state = semantic_state_for_frame(self.catalog, episode, frame, self.matrix).phase
            intent = batch["semantic_view_name"][row].rsplit("/", 1)[0]
            if intent not in self.matrix[state]:
                raise ValueError(f"Sampled instruction outside preset: {state} x {intent}")
            key = f"{state}/{intent}"
            examples = self.examples.setdefault(key, [])
            if len(examples) >= 4 or any(item["natural_language"] == original for item in examples):
                continue
            if not original.strip() or original.strip() == intent:
                raise ValueError("Training input contains a semantic key instead of natural language")
            cleaned = original.strip().replace("_", " ").replace("\n", " ")
            if not batch["task"][row].startswith(f"Task: {cleaned}"):
                raise ValueError("Tokenizer prompt does not contain the selected natural-language annotation")
            if self.continuous_state_memory and batch["task"][row] != f"Task: {cleaned};\nAction: ":
                raise ValueError("Continuous MEM must not inherit a discrete-state tokenizer prompt")
            mask = batch[OBS_LANGUAGE_ATTENTION_MASK][row].bool()
            ids = batch[OBS_LANGUAGE_TOKENS][row][mask].detach().cpu().tolist()
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            examples.append(
                {
                    "step": step,
                    "episode_index": episode,
                    "frame_index": frame,
                    "natural_language": original,
                    "tokenizer_prompt": batch["task"][row],
                    "decoded_model_tokens": decoded,
                    "token_ids": ids,
                }
            )
            changed = True
        if changed:
            expected = [f"{state}/{intent}" for state, row in self.matrix.items() for intent in row]
            self.complete = all(len(self.examples.get(key, [])) >= 4 for key in expected)
            self.path.write_text(
                json.dumps(
                    {
                        "configured_cells": expected,
                        "observed_cells": sorted(self.examples),
                        "examples": self.examples,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
