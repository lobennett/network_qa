"""Tests for network_qa.exclusions.base — registry + provenance.

Ported from neuro_workflow/tests/exclusions/test_base.py (register/get/list
generators, portable as-is) and neuro_workflow/tests/exclusions/test_provenance.py
(make_meta / _jsonify / _git_sha unit tests, portable as-is).

NOT ported: test_provenance.py's disk-persistence tests (save_source_entries,
load_source_entries, compile_exclusions with EXCLUSIONS_DIR/LOCKFILE_DIR) and
all of neuro_workflow/tests/core/test_exclusions.py (validate_entry,
save/load_source_entries, save/load_overrides, is_excluded, get_trim_info).
Those exercise neuro_workflow's `core.exclusions` sources-dir + manual-override
persistence layer, which network_qa's simplified design (Task 6: generators run
in-memory, merged directly by `compile.py`) does not replicate — there is no
`network_qa.core.exclusions` module. See tests/exclusions/test_compile.py for
the in-memory analogue of the compile/lockfile behavior.
"""
from argparse import Namespace
from pathlib import Path

from network_qa.exclusions.base import (
    register_generator,
    get_generator,
    list_generators,
)


class FakeGenerator:
    name = "fake"
    description = "A fake generator for testing"

    def add_cli_args(self, parser):
        pass

    def generate(self, dataset_name, dataset_config, args):
        return []


def test_register_and_get():
    gen = FakeGenerator()
    register_generator(gen)
    assert get_generator("fake") is gen


def test_get_unknown_returns_none():
    assert get_generator("nonexistent-gen") is None


def test_list_generators():
    gen = FakeGenerator()
    register_generator(gen)
    generators = list_generators()
    assert "fake" in generators


def test_make_meta_shape():
    """make_meta returns a dict with all expected keys."""
    from network_qa.exclusions.base import make_meta

    meta = make_meta("foo", Namespace(x=1, y="hello"), n_entries=5)

    assert set(meta.keys()) == {"generator", "ran_at", "code_sha", "args", "n_entries"}
    assert meta["generator"] == "foo"
    assert meta["n_entries"] == 5
    assert meta["args"] == {"x": 1, "y": "hello"}
    # ran_at is an ISO-8601 timestamp ending in Z (UTC)
    assert isinstance(meta["ran_at"], str)
    assert meta["ran_at"].endswith("Z")
    # code_sha is either a string or None
    assert meta["code_sha"] is None or isinstance(meta["code_sha"], str)


def test_make_meta_serializes_path_args():
    """args containing Path instances stringify to make the dict JSON-safe."""
    from network_qa.exclusions.base import make_meta

    meta = make_meta("foo", Namespace(decisions_tsv=Path("/tmp/x.tsv")), n_entries=0)
    assert meta["args"] == {"decisions_tsv": "/tmp/x.tsv"}


def test_make_meta_accepts_dict_args():
    """args can be a plain dict in addition to Namespace."""
    from network_qa.exclusions.base import make_meta

    meta = make_meta("foo", {"x": 1}, n_entries=0)
    assert meta["args"] == {"x": 1}


def test_make_meta_args_none():
    """args=None records null in the meta block."""
    from network_qa.exclusions.base import make_meta

    meta = make_meta("foo", None, n_entries=0)
    assert meta["args"] is None


def test_make_meta_strips_callable_from_args():
    """argparse Namespaces carry a `func` callback (set via subparser
    set_defaults). The audit-trail args dict must drop it so json.dumps
    succeeds on the saved sources file."""
    import json
    from network_qa.exclusions.base import make_meta

    def _stub_callback(args, remaining):
        pass

    meta = make_meta(
        "foo",
        Namespace(dataset="discovery", source="motion", func=_stub_callback),
        n_entries=0,
    )
    # callable stripped out
    assert "func" not in meta["args"]
    # other args preserved
    assert meta["args"]["dataset"] == "discovery"
    assert meta["args"]["source"] == "motion"
    # full meta JSON-serializes without crashing
    json.dumps(meta)
