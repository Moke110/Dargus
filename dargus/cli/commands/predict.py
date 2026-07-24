"""Predict subcommand handler — parses CLI args, builds task_spec, calls API."""

from __future__ import annotations

from argparse import Namespace


def handle_predict(args: Namespace) -> int:
    """Execute the ``dargus predict`` subcommand.

    Parses ``--drugs`` as comma-separated IDs and delegates to
    :func:`dargus.api.predict`.
    """
    from dargus.api import predict

    drug_ids = [d.strip() for d in args.drugs.split(",") if d.strip()]
    result = predict(
        drug_ids=drug_ids,
        disease_id=args.disease,
        endpoints=args.endpoints,
        max_rounds=args.max_rounds,
    )
    for drug, disease_eps in result.items():
        print(f"{drug}:")
        for disease, endpoints_dict in disease_eps.items():
            for endpoint, pred in endpoints_dict.items():
                print(
                    f"  {disease}/{endpoint}: "
                    f"[{pred['efficacy_low']:.3f}, {pred['efficacy_up']:.3f}]"
                )
    return 0
