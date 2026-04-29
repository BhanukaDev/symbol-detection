"""Verify that a generated dataset's category mapping matches the deployment categories.json.

Run from python/ directory:
    python test_categories.py
    python test_categories.py --dataset dataset --deploy ../hf-deploy/model/categories.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from python/ without activating the venv
_here = Path(__file__).parent
for _p in ['src', 'floor-grid/src', 'effects/src']:
    _full = str(_here / _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

from symbol_detection.dataset.generator import SYMBOL_CATEGORIES

EXPECTED = {c["id"]: c["name"] for c in SYMBOL_CATEGORIES}


def check(dataset_json: Path, deploy_json: Path):
    errors = []

    # --- Check 1: generated dataset annotations.json ---
    with open(dataset_json) as f:
        dataset = json.load(f)
    dataset_cats = {c["id"]: c["name"] for c in dataset["categories"]}

    print(f"Checking {dataset_json}")
    for cid, name in EXPECTED.items():
        actual = dataset_cats.get(cid)
        status = "OK" if actual == name else "FAIL"
        print(f"  {status} id={cid}  expected='{name}'  got='{actual}'")
        if actual != name:
            errors.append(f"dataset id={cid}: expected '{name}', got '{actual}'")

    # --- Check 2: deployed categories.json ---
    print(f"\nChecking {deploy_json}")
    with open(deploy_json) as f:
        deploy = json.load(f)
    deploy_cats = {c["id"]: c["name"] for c in deploy["categories"]}

    for cid, name in EXPECTED.items():
        actual = deploy_cats.get(cid)
        status = "OK" if actual == name else "FAIL"
        print(f"  {status} id={cid}  expected='{name}'  got='{actual}'")
        if actual != name:
            errors.append(f"deploy id={cid}: expected '{name}', got '{actual}'")

    print()
    if errors:
        print(f"FAIL — {len(errors)} mismatch(es):")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print("PASS — all category IDs match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/annotations.json")
    parser.add_argument("--deploy", default="../hf-deploy/model/categories.json")
    args = parser.parse_args()
    check(Path(args.dataset), Path(args.deploy))
