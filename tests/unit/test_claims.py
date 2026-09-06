import pytest
from pydantic import ValidationError

from board_of_scientists.evidence.claims import Claim, normalize_claim_text


def test_normalize_claim_text_collapses_whitespace():
    assert normalize_claim_text("  A\nclaim\twith   spacing.  ") == "A claim with spacing."


def test_claim_model_normalizes_text_on_validation():
    claim = Claim(id="c1", text="  The   model\nworks. ", source="paper:p2")
    assert claim.text == "The model works."


def test_claim_evidence_level_is_bounded():
    with pytest.raises(ValidationError):
        Claim(id="c1", text="claim", source="paper", evidence_level=8)
    with pytest.raises(ValidationError):
        Claim(id="c1", text="claim", source="paper", evidence_level=-1)
