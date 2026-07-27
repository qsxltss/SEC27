#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from ae_experiments import rq1, rq2, rq3
from ae_experiments.pipeline import DATASETS, METHODS, RETRIEVERS


def choices(value: str, allowed: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    selected = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid {label}: {', '.join(invalid)}")
    return selected


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Anonymous artifact experiment runner")
    subparsers = result.add_subparsers(dest="command", required=True)

    rq1_parser = subparsers.add_parser("rq1", help="run RQ1 retrieval and membership evaluation")
    rq1_parser.add_argument("--datasets", default=",".join(DATASETS))
    rq1_parser.add_argument("--retrievers", default=",".join(RETRIEVERS))
    rq1_parser.add_argument("--methods", default=",".join(METHODS))
    rq1_parser.add_argument("--output-dir", type=Path, default=None)

    subparsers.add_parser("rq2", help="run the cached Lakera evaluation")
    subparsers.add_parser("rq3", help="run query-exposure metrics")
    subparsers.add_parser("all", help="run RQ1, RQ2, and RQ3")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "rq1":
        rq1.run(
            output_dir=args.output_dir,
            datasets=choices(args.datasets, DATASETS, "dataset"),
            retrievers=choices(args.retrievers, RETRIEVERS, "retriever"),
            methods=choices(args.methods, METHODS, "method"),
        )
    elif args.command == "rq2":
        rq2.run()
    elif args.command == "rq3":
        rq3.run()
    elif args.command == "all":
        rq1.run()
        rq2.run()
        rq3.run()


if __name__ == "__main__":
    main()
