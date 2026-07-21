"""Target-disease efficacy scan workflow."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def run(**kwargs: Any) -> dict[str, Any]:
    """Deprecated v3 workflow. Use run_v4() instead."""
    raise NotImplementedError(
        "The v3 target_efficacy_scan workflow has been removed. Use run_v4() instead."
    )


def run_v4(
    drugs: list[str],
    disease: str,
    datadir: str | None = None,
    projects_root: str = "projects",
) -> dict:
    """Run the 0.6.0 target-disease efficacy scan workflow."""
    from dargus.iris.commander import Iris

    if projects_root:
        os.environ["DARGUS_HOME"] = projects_root

    iris = Iris()
    if datadir:
        iris.train(datadir)
    predictions = iris.predict(
        drug_ids=drugs,
        disease_id=disease,
        endpoints=[],
    )
    return {"predictions": predictions}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a target-disease efficacy scan")
    parser.add_argument("--target", required=True)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--endpoints", nargs="+")
    parser.add_argument("--drugs", nargs="+")
    args = parser.parse_args()
    result = run_v4(
        drugs=args.drugs,
        disease=args.disease,
    )
    print(f"Predictions: {result['predictions']}")
