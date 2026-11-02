"""Resolve a job to its prepared release, or emit its data preparation request."""
import argparse
import json
from pathlib import Path

from dataset_release import read, resolve_release, verify_release
from semantic_plan import load_plan, resolve_task, selected_matrix, state_weights, validate_sidecar


def request_for_job(payload, task_id):
    task=resolve_task(payload,task_id)
    if task["semantic_views_file"] != "semantic_views.json":
        raise ValueError("Published releases use only meta/semantic_views.json; remove the manual sidecar override")
    return {"recipe":task.get("data_recipe",task.get("dataset_family")),
            "state_weights":state_weights(payload,task_id),
            "state_instruction_matrix":selected_matrix(payload,task_id)}


def resolve(datasets_root,payload,task_id,checkpoint=None):
    request=request_for_job(payload,task_id)
    pinned=None
    if checkpoint is not None:
        pinned=read(Path(checkpoint)/"pretrained_model/train_config.json")["dataset"]["root"]
    task=resolve_task(payload,task_id)
    root=resolve_release(datasets_root,request,pinned_root=pinned,artifact_id=task.get("dataset_artifact_id"))
    validate_sidecar(payload,task_id,root/"meta/semantic_views.json")
    return root


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets",type=Path,default=here/"semantic_training_presets.json")
    parser.add_argument("--jobs",type=Path,default=here/"training_jobs.json")
    parser.add_argument("--task",required=True)
    parser.add_argument("--datasets-root",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--request-only",action="store_true")
    parser.add_argument("--format-lines",action="store_true")
    args=parser.parse_args()
    payload=load_plan(args.presets,args.jobs)
    if args.request_only:
        print(json.dumps(request_for_job(payload,args.task),sort_keys=True))
        return
    if args.datasets_root is None:
        parser.error("--datasets-root required for resolution")
    root=resolve(args.datasets_root,payload,args.task,args.checkpoint)
    receipt=verify_release(root)
    if args.format_lines:
        print(root)
        print(root/"meta/semantic_views.json")
        print(receipt["artifact_id"])
    else:
        print(json.dumps(dict(root=str(root),artifact_id=receipt["artifact_id"])))


if __name__=="__main__":
    main()
