from __future__ import annotations

# Load environment before importing anything else
from dotenv import load_dotenv
load_dotenv()

from swelancer import SWELancerEval 
import argparse
import nanoeval
import json
import os
import datetime
from nanoeval.evaluation import EvalSpec, RunnerArgs
from nanoeval.examples._gpqa import GPQAEval
from nanoeval.recorder import dummy_recorder
from nanoeval.json_recorder import json_recorder
from nanoeval.setup import nanoeval_entrypoint
from swelancer_agent import SimpleAgentSolver

def parse_args():
    parser = argparse.ArgumentParser(description='Run SWELancer evaluation')
    parser.add_argument('--issue_ids', nargs='*', type=str, help='List of ISSUE_IDs to evaluate. If not specified, all issues will be evaluated.')
    parser.add_argument('--model', type=str, required=True, help='The model to use for evaluation (e.g., "gpt-4o-mini", "o1")')
    return parser.parse_args()

async def main() -> None:
    args = parse_args()
    taskset = args.issue_ids if args.issue_ids else []
    model = args.model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    report = await nanoeval.run(
        EvalSpec(
            # taskset is a list of ISSUE_IDs you wish to evaluate (e.g., ["123", "456_789"])
            eval=SWELancerEval(
                solver=SimpleAgentSolver(model=model),
                taskset=taskset
            ),
            runner=RunnerArgs(
                concurrency=10,
                experimental_use_multiprocessing=False,
                enable_slackbot=False,
                recorder=json_recorder(),
                max_retries=5
            ),
        )
    )
    
    # Save report to disk with timestamp and model name
    report_filename = f"swelancer_report_{model}_{timestamp}.json"
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", report_filename), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to reports/{report_filename}")


if __name__ == "__main__":
    nanoeval_entrypoint(main())
