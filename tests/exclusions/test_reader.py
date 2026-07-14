"""Tests for the consumer-side is_excluded() reader — the per-scan query helper
network_glm/lev1 uses to enforce exclusions from a compiled lockfile.

Ported/adapted from neuro_workflow/tests/core/test_exclusions.py::test_is_excluded.

Matching semantics verified against neuro_workflow's
core/exclusions.py::is_excluded (line ~221): it does an EXACT tuple match on
(subject, session, task, run) against each entry (via _scan_key), considering
ONLY entries whose action is in {"exclude", "trim"}. There is NO entity
normalization inside is_excluded — the lockfile stores BIDS-prefixed entities
(sub-/ses-/task-/run-) and the caller (lev1) queries with prefixed entities, so
a bare query does NOT match a prefixed entry. Normalization stays the caller's
job; matching the monolith exactly keeps network_glm byte-compatible with the
monolith's lev1 consumer.
"""
from network_qa.compile import is_excluded


_ENTRIES = [
    {"subject": "sub-s01", "session": "ses-01", "task": "task-rest", "run": "run-1",
     "source": "motion", "action": "exclude", "reason": "High FD"},
]


def test_exact_match_hit():
    assert is_excluded("sub-s01", "ses-01", "task-rest", "run-1", _ENTRIES) is True


def test_non_match_miss_different_session():
    # Same scan but ses-02 -> not in the list -> False (monolith test case).
    assert is_excluded("sub-s01", "ses-02", "task-rest", "run-1", _ENTRIES) is False


def test_absent_scan_returns_false():
    assert is_excluded("sub-s99", "ses-01", "task-rest", "run-1", _ENTRIES) is False


def test_empty_exclusions_returns_false():
    assert is_excluded("sub-s01", "ses-01", "task-rest", "run-1", []) is False


def test_prefixed_vs_bare_no_normalization():
    """Faithful to the monolith: is_excluded does NOT normalize entities.
    The lockfile holds prefixed entities; a bare-entity query does not match a
    prefixed entry (normalization is the caller's responsibility)."""
    assert is_excluded("s01", "01", "rest", "1", _ENTRIES) is False
    # ...and the prefixed form of the same scan DOES match.
    assert is_excluded("sub-s01", "ses-01", "task-rest", "run-1", _ENTRIES) is True


def test_trim_action_counts_as_excluded():
    """Monolith counts action in {'exclude', 'trim'} as excluded."""
    trim_entries = [
        {"subject": "sub-s03", "session": "ses-11", "task": "task-stop", "run": "run-1",
         "source": "neg-events", "action": "trim", "reason": "Non-monotonic"},
    ]
    assert is_excluded("sub-s03", "ses-11", "task-stop", "run-1", trim_entries) is True


def test_non_exclude_action_is_ignored():
    """An entry with a non-exclude/non-trim action (e.g. force-include) does not
    count as excluded, matching the monolith's action filter."""
    fi_entries = [
        {"subject": "sub-s02", "session": "ses-05", "task": "task-rest", "run": "run-1",
         "source": "override", "action": "force-include", "reason": "Override"},
    ]
    assert is_excluded("sub-s02", "ses-05", "task-rest", "run-1", fi_entries) is False


def test_reader_over_a_compiled_lockfile_list():
    """End-to-end: compile a lockfile in-memory, then query its exclusions list."""
    from argparse import Namespace
    from network_qa import compile as nqc
    from network_qa.exclusions.base import register_generator

    class _Gen:
        name = "reader_fixture"; description = "test"
        def add_cli_args(self, p): pass
        def generate(self, ds, cfg, args):
            return [{"subject": "sub-s10", "session": "ses-02", "task": "task-cuedTS",
                     "run": "run-1", "source": "reader_fixture", "action": "exclude",
                     "reason": "y"}]

    register_generator(_Gen())
    lock = nqc.compile_exclusions("discovery", {}, Namespace(),
                                  generator_names=["reader_fixture"])
    assert is_excluded("sub-s10", "ses-02", "task-cuedTS", "run-1", lock["exclusions"]) is True
    assert is_excluded("sub-s10", "ses-02", "task-cuedTS", "run-2", lock["exclusions"]) is False
