"""IrisAnalog — drug analog similarity prediction, wrapped as Tool."""

from __future__ import annotations

from typing import Any

from dargus.dbase import DBase
from dargus.iris.analog import IrisAnalog


def analog_predict(
    drug_ids: list[str],
    disease_id: str,
    endpoints: list[str] | None = None,
) -> dict[str, Any]:
    dbase = DBase.global_instance()
    agent = IrisAnalog()
    return agent.predict(dbase, drug_ids, disease_id, endpoints or [])
