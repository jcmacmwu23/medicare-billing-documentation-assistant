import re
from collections.abc import Callable

from .fixtures import load_fixture

_CLAIM_ID = re.compile(r"SYN-[0-9]{4}")
_CLAIM_FIELDS = (
    "service_month",
    "hcpcs",
    "diagnosis_code",
    "diagnosis_pointer_present",
    "frequency_category",
    "previous_covered_screening_month",
    "carc",
    "rarc",
)


def valid_claim_id(claim_id: str) -> bool:
    return isinstance(claim_id, str) and _CLAIM_ID.fullmatch(claim_id) is not None


def get_synthetic_claim(
    claim_id: str,
    *,
    allowed: bool,
    load_claims: Callable[[], dict] = lambda: load_fixture("claims.json"),
) -> dict:
    """Look up a single fixture under an application-supplied permission."""
    if not allowed:
        return {"status": "denied"}
    if not valid_claim_id(claim_id):
        return {"status": "invalid_id"}

    claim = load_claims().get(claim_id)
    if claim is None:
        return {"status": "not_found"}
    return {
        "status": "ok",
        "claim": {"claim_id": claim_id, **{field: claim[field] for field in _CLAIM_FIELDS}},
    }
