import argparse
import json
from pathlib import Path

from behavioral_coupling.eval.eval import atomic_write_json


def flatten(rows):
    for row in rows:
        yield from row.get("questions", [row])


def summarize_file(path):
    data = json.loads(Path(path).read_text())
    rows = list(flatten(data["responses"]))
    labels = [row.get("eval", {}).get("label") for row in rows]
    counts = {label: labels.count(label) for label in ("yes", "no", "refusal")}
    completed = sum(counts.values())
    answered = counts["yes"] + counts["no"]
    return {
        "score_file": str(path),
        "judge_model_name": data.get("summary", {}).get("judge_model_name"),
        "total": len(rows), "completed": completed, "pending": len(rows)-completed,
        **counts,
        "yes_rate_all": counts["yes"]/completed if completed else None,
        "yes_rate_answered": counts["yes"]/answered if answered else None,
        "refusal_rate": counts["refusal"]/completed if completed else None,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize_file(path) for path in args.scores]
    atomic_write_json({"results": rows}, args.output)
    print(json.dumps(rows, indent=2))
    if any(row["pending"] for row in rows):
        raise SystemExit("Incomplete scores; pending items are not counted as NO.")


if __name__ == "__main__":
    main()
