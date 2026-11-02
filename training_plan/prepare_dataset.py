"""Fixed host preparation bridge; model training never constructs data itself."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from semantic_plan import load_plan
from resolve_dataset import request_for_job


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",required=True)
    parser.add_argument("--datasets-root",type=Path,required=True)
    parser.add_argument("--presets",type=Path,default=here/"semantic_training_presets.json")
    parser.add_argument("--jobs",type=Path,default=here/"training_jobs.json")
    parser.add_argument("--data-code",type=Path,default=here.parents[1]/"data_postprocessing")
    args=parser.parse_args()
    request=request_for_job(load_plan(args.presets,args.jobs),args.task)
    with tempfile.TemporaryDirectory(prefix="training-data-request-") as directory:
        path=Path(directory)/"request.json"
        path.write_text(json.dumps(request))
        subprocess.run(["bash",str(args.data_code/"tools/prepare_training_dataset.sh"),
                        str(args.datasets_root),str(path)],check=True)


if __name__=="__main__":
    main()
