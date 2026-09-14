import pytest

from cvslab.service import LabService
from cvslab.store import Store


@pytest.fixture()
def lab(tmp_path):
    """A fresh lab service over a throwaway store."""
    return LabService(Store(tmp_path / "labstore"))


@pytest.fixture()
def seeded_lab(lab):
    """A lab with the full Phase 0 proof executed at tiny scale."""
    lab.result = lab.run_phase0_proof(n_games=25, epochs=3, seeds=(0,), control_width=1, intervention_width=8)
    return lab
