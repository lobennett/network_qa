"""Tests for network_qa.exclusions.behavioral — ported from
neuro_workflow/tests/exclusions/test_behavioral.py and
neuro_workflow/tests/events/test_behavioral_exclusion_generator.py
(two separate monolith test files covering the same generator),
import-repointed."""
from argparse import Namespace

import pytest
from pathlib import Path
from unittest.mock import patch

from network_qa.exclusions.behavioral import BehavioralGenerator


def test_generator_attributes():
    g = BehavioralGenerator()
    assert g.name == "behavioral"
    assert g.description


def test_generate_returns_empty(tmp_path):
    g = BehavioralGenerator()
    config = {"bids_dir": str(tmp_path)}
    args = Namespace()
    entries = g.generate("test", config, args)
    assert entries == []


class TestBehavioralGenerator:
    def test_registered_as_behavioral(self):
        from network_qa.exclusions.base import get_generator
        import network_qa.exclusions.behavioral  # noqa: F401 (trigger registration)
        gen = get_generator("behavioral")
        assert gen is not None
        assert gen.name == "behavioral"

    def test_generate_returns_list(self, tmp_path):
        from network_qa.exclusions.base import get_generator
        import network_qa.exclusions.behavioral  # noqa: F401
        gen = get_generator("behavioral")
        # Create minimal sourcedata structure
        beh_dir = tmp_path / "sourcedata" / "sub-s01" / "ses-01" / "beh"
        beh_dir.mkdir(parents=True)
        # Also need bids_dir with func dir
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        func_dir.mkdir(parents=True)

        from argparse import Namespace
        args = Namespace(behavioral_dir=str(tmp_path / "sourcedata"))
        config = {"bids_dir": str(tmp_path)}
        result = gen.generate("test", config, args)
        assert isinstance(result, list)

    def test_exclusion_entry_format(self):
        """Entries must have required fields for the exclusions system."""
        entry = {
            "subject": "sub-s01",
            "session": "ses-01",
            "task": "stopSignal",
            "run": "run-1",
            "action": "exclude",
            "source": "behavioral-qc",
            "reason": "test reason",
        }
        required = {"subject", "session", "task", "run", "action", "reason"}
        assert required.issubset(entry.keys())
