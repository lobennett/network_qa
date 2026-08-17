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


# --- code_sha resolution ------------------------------------------------------
#
# Production lockfiles were landing with `"code_sha": null` because the `select`
# stage runs inside network_fmri.sif, where network_qa is a pip-installed package
# with no `.git` directory -- so the git-only lookup had nothing to read. The
# installed distribution does record the exact pinned commit in its dist-info
# `direct_url.json` (a PEP 610 VCS install), so `code_sha()` falls back to that
# before giving up. Resolution order: env override -> git HEAD -> dist-info.


def test_code_sha_prefers_env_override(monkeypatch):
    """An explicit NETWORK_QA_CODE_SHA wins over every other source: the caller
    (orchestrator / container build) is asserting what code this is."""
    from network_qa.exclusions import base

    monkeypatch.setenv("NETWORK_QA_CODE_SHA", "deadbee")
    monkeypatch.setattr(base, "_git_sha", lambda: "1111111")
    monkeypatch.setattr(base, "_dist_sha", lambda: "2222222")

    assert base.code_sha() == "deadbee"


def test_code_sha_falls_back_to_git(monkeypatch):
    """With no env override, a real git checkout reports its HEAD."""
    from network_qa.exclusions import base

    monkeypatch.delenv("NETWORK_QA_CODE_SHA", raising=False)
    monkeypatch.setattr(base, "_git_sha", lambda: "abc1234")
    monkeypatch.setattr(base, "_dist_sha", lambda: "2222222")

    assert base.code_sha() == "abc1234"


def test_code_sha_falls_back_to_dist_when_not_a_git_repo(monkeypatch):
    """The container case: no .git, so the pinned commit comes from dist-info."""
    from network_qa.exclusions import base

    monkeypatch.delenv("NETWORK_QA_CODE_SHA", raising=False)
    monkeypatch.setattr(base, "_git_sha", lambda: None)
    monkeypatch.setattr(base, "_dist_sha", lambda: "494ab9d")

    assert base.code_sha() == "494ab9d"


def test_code_sha_none_when_nothing_resolves(monkeypatch):
    """No env, no git, no dist metadata -> null, as before (never raises)."""
    from network_qa.exclusions import base

    monkeypatch.delenv("NETWORK_QA_CODE_SHA", raising=False)
    monkeypatch.setattr(base, "_git_sha", lambda: None)
    monkeypatch.setattr(base, "_dist_sha", lambda: None)

    assert base.code_sha() is None


def test_dist_sha_reads_pep610_commit_id(monkeypatch):
    """_dist_sha shortens the dist-info direct_url.json commit to 7 chars."""
    from network_qa.exclusions import base

    class _FakeDist:
        version = "0.1.0"

        def read_text(self, name):
            assert name == "direct_url.json"
            return (
                '{"url":"https://github.com/lobennett/network_qa.git",'
                '"vcs_info":{"vcs":"git","commit_id":'
                '"494ab9d035810b049ea9dbb200fcf8c34ad04a6b"}}'
            )

    monkeypatch.setattr(base, "_distribution", lambda name: _FakeDist())
    assert base._dist_sha() == "494ab9d"


def test_dist_sha_falls_back_to_version_for_non_vcs_install(monkeypatch):
    """A plain (non-VCS) install has no commit, so report the version, tagged so
    it can't be mistaken for a git sha."""
    from network_qa.exclusions import base

    class _FakeDist:
        version = "0.1.0"

        def read_text(self, name):
            return None

    monkeypatch.setattr(base, "_distribution", lambda name: _FakeDist())
    assert base._dist_sha() == "dist:0.1.0"


def test_dist_sha_none_when_package_not_installed(monkeypatch):
    """Running from a source tree with no installed dist -> None, no exception."""
    from network_qa.exclusions import base

    def _boom(name):
        raise base._PackageNotFoundError(name)

    monkeypatch.setattr(base, "_distribution", _boom)
    assert base._dist_sha() is None


def test_compile_stamps_resolved_code_sha(monkeypatch):
    """compile_exclusions puts the resolved sha in _meta (regression: it used to
    call the git-only helper directly, yielding null inside the container).

    Uses an unregistered generator name so no generator actually runs (an empty
    list would mean "all generators" -- `names = generator_names or all`).
    """
    from argparse import Namespace

    from network_qa import compile as compile_mod

    monkeypatch.setattr(compile_mod, "code_sha", lambda: "494ab9d")
    lock = compile_mod.compile_exclusions(
        "discovery", {}, Namespace(), generator_names=["__no_such_generator__"]
    )
    assert lock["_meta"]["code_sha"] == "494ab9d"
    assert lock["exclusions"] == []
