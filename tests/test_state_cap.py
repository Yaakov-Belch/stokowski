"""Tests for `agent.max_concurrent_agents_by_state` (YAA-6).

The dispatch loop in `orchestrator.py` looks this map up with
`issue.state.strip().lower()` — the Linear column name, trimmed and
lowercased. `parse_workflow_file()` used to store the map exactly as YAML
produced it, so a lookup only ever succeeded if the operator happened to
type the key already trimmed and lowercased: `"in progress"` matched,
`"In Progress"` (the exact Linear column name — what `linear_states:`
already requires elsewhere in this same config) and any internal workflow
state name never did, silently. `_normalize_state_caps()` closes that gap
at parse time.

Lowercasing the keys and validating the values are the same change, not
two: lowercasing without dropping invalid values turns a previously-inert
typo (the bad value was never read, because the key never matched) into a
`TypeError` that escapes `Orchestrator._tick()` on every poll.
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from pathlib import Path

import pytest

from stokowski.config import _normalize_state_caps, parse_workflow_file
from stokowski.models import Issue, RunAttempt
from stokowski.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parent.parent


# ── Parse-time normalization (config.py) ─────────────────────────────────────


def test_keys_are_trimmed_and_lowercased():
    """The exact-case Linear column name is what an operator naturally
    writes (`linear_states:` already requires exact case) — it must work."""
    assert _normalize_state_caps({"In Progress": 2}) == {"in progress": 2}
    assert _normalize_state_caps({"  Human Review  ": 1}) == {"human review": 1}


def test_already_normalized_keys_are_unaffected():
    """Idempotent on the one spelling that happened to work before the fix —
    no migration required for operators who stumbled onto it."""
    assert _normalize_state_caps({"in progress": 3}) == {"in progress": 3}


def test_non_positive_values_are_dropped_not_treated_as_a_cap():
    """A cap of 0 or negative must be ignored, not read as 'block everyone' —
    `state_count >= 0` is unconditionally true and would starve the state."""
    assert _normalize_state_caps({"in progress": 0}) == {}
    assert _normalize_state_caps({"in progress": -1}) == {}


def test_non_numeric_values_are_dropped_without_raising():
    """The defect this guards against: lowercasing alone makes a previously
    unreachable bad value reachable at the `>=` comparison in the dispatch
    loop, raising `TypeError` and aborting every tick. Dropping it here keeps
    it inert, the same as a key that never matched before the fix."""
    assert _normalize_state_caps({"in progress": "two"}) == {}


def test_bool_values_are_rejected():
    """`bool` is an `int` subclass in Python — `True` must not silently
    become a cap of 1, or a YAML `true`/`false` typo becomes a real limit."""
    assert _normalize_state_caps({"in progress": True}) == {}
    assert _normalize_state_caps({"in progress": False}) == {}


def test_non_integral_floats_are_rejected():
    """The spec calls for a positive integer; truncating 1.9 to 1 would
    silently honour a value the operator did not write."""
    assert _normalize_state_caps({"in progress": 1.9}) == {}


def test_whole_number_floats_are_accepted():
    assert _normalize_state_caps({"in progress": 2.0}) == {"in progress": 2}


def test_none_and_empty_input_produce_an_empty_map():
    assert _normalize_state_caps(None) == {}
    assert _normalize_state_caps({}) == {}


def test_mixed_map_keeps_only_the_valid_entries():
    raw = {
        "In Progress": 2,
        "Human Review": "two",
        "Rework": 0,
        "  Todo  ": -1,
        "Done": True,
    }
    assert _normalize_state_caps(raw) == {"in progress": 2}


def test_dropped_entries_are_logged_so_a_typo_is_not_silent(caplog):
    """A cap key that never matches anything is otherwise indistinguishable
    from 'no limit configured' — the legitimate common case. A warning at
    parse time is the only signal an operator gets that their value, not
    just their key, was rejected."""
    with caplog.at_level(logging.WARNING, logger="stokowski.config"):
        _normalize_state_caps({"Human Review": "two"})
    assert any("Human Review" in r.message for r in caplog.records)


def test_the_shipped_example_workflow_parses_with_a_working_cap():
    """workflow.example.yaml keys its cap by the Linear column name after
    this fix — regression guard against reintroducing the internal-state-name
    example that could never match, which is what shipped before YAA-6."""
    cfg = parse_workflow_file(REPO / "workflow.example.yaml").config
    caps = cfg.agent.max_concurrent_agents_by_state
    assert caps, "the shipped example should demonstrate a working cap"
    assert set(caps) <= {"in progress", "todo"}, (
        f"cap keys {set(caps)} are not in the reachable Linear-column key "
        f"space {{'in progress', 'todo'}} for this workflow's agent states"
    )


# ── End-to-end dispatch (orchestrator.py) ────────────────────────────────────
# `test_config.py`-style unit tests cover the normalization function in
# isolation; the tests below drive the actual `Orchestrator._tick()` dispatch
# loop, because the reported defect was never in the dict lookup itself — it
# was in which value reached that lookup. A unit test of `dict.get()` alone
# would have passed throughout the whole bug's lifetime.


def run(coro):
    return asyncio.run(coro)


class FakeLinear:
    """No network. Candidates are whatever the test assigns."""

    def __init__(self):
        self.candidates: list[Issue] = []

    async def fetch_candidate_issues(self, slug, states):
        return self.candidates

    async def fetch_issue_states_by_ids(self, ids):
        return {}


def _issue(n, state="In Progress"):
    return Issue(id=f"id-{n}", identifier=f"T-{n}", title=f"issue {n}", state=state, priority=1)


async def _noop(*a, **k):
    return None


@pytest.fixture
def orchestrator(tmp_path):
    """A minimal single-state workflow — only the cap check under test."""
    wf = textwrap.dedent("""
        tracker:
          kind: linear
          endpoint: https://api.linear.app/graphql
          api_key: fake-key
          project_slug: deadbeef
        workspace:
          root: {root}
        agent:
          max_concurrent_agents: 10
          max_concurrent_agents_by_state: {{CAPMAP}}
        linear_states:
          todo: Todo
          active: In Progress
          review: Human Review
          terminal: [Done]
        prompts:
          global_prompt: g.md
        states:
          investigate:
            type: agent
            prompt: p.md
            linear_state: active
            transitions: {{complete: done}}
          done:
            type: terminal
            linear_state: terminal
    """).format(root=tmp_path / "ws")
    (tmp_path / "g.md").write_text("g")
    (tmp_path / "p.md").write_text("p")

    def build(capmap_yaml: str) -> Orchestrator:
        (tmp_path / "workflow.yaml").write_text(wf.replace("{CAPMAP}", capmap_yaml))
        orch = Orchestrator(workflow_path=tmp_path / "workflow.yaml")
        errors = orch._load_workflow()
        assert not errors, f"config failed to load: {errors}"

        fake = FakeLinear()
        orch._linear = fake
        orch._ensure_linear_client = lambda: fake
        dispatched: list[str] = []
        orch._dispatch = lambda issue: dispatched.append(issue.identifier)
        orch._reconcile = _noop
        orch._handle_gate_responses = _noop
        orch._evict_terminal_gates = _noop
        orch._resolve_current_state = _noop

        orch._fake_linear = fake
        orch._dispatched = dispatched
        return orch

    return build


def _preload_running(orch, n: int, state="In Progress"):
    """Simulate `n` agents already dispatched into `state`."""
    for i in range(n):
        iid = f"running-{i}"
        orch.running[iid] = RunAttempt(issue_id=iid, issue_identifier=f"R-{i}")
        orch._last_issues[iid] = _issue(f"r{i}", state=state)


# ── Acceptance criterion 1: the exact-case Linear column name caps ──────────


def test_exact_case_linear_column_name_caps_dispatch(orchestrator):
    """'In Progress: 2' — the ticket's own acceptance criterion — must limit
    dispatch when a candidate is looked up as 'in progress'."""
    orch = orchestrator('{"In Progress": 1}')
    _preload_running(orch, 1)
    orch._fake_linear.candidates = [_issue(99)]

    run(orch._tick())

    assert orch._dispatched == []
    assert any("per-state cap" in q["reason"] for q in orch._queued)


def test_the_cap_is_not_applied_when_under_the_limit(orchestrator):
    orch = orchestrator('{"In Progress": 2}')
    _preload_running(orch, 1)
    orch._fake_linear.candidates = [_issue(99)]

    run(orch._tick())

    assert orch._dispatched == ["T-99"]


def test_already_lowercased_keys_still_work(orchestrator):
    """No migration burden for an operator who stumbled onto the one
    spelling that happened to work before the fix."""
    orch = orchestrator('{"in progress": 1}')
    _preload_running(orch, 1)
    orch._fake_linear.candidates = [_issue(99)]

    run(orch._tick())

    assert orch._dispatched == []


# ── Acceptance criterion 2: non-positive values are ignored ─────────────────


@pytest.mark.parametrize("bad_value", [0, -1])
def test_non_positive_values_do_not_starve_the_state(orchestrator, bad_value):
    """A value of 0 or -1 must be dropped, not enforced as 'block everyone' —
    `state_count >= 0` is unconditionally true."""
    orch = orchestrator(f'{{"in progress": {bad_value}}}')
    orch._fake_linear.candidates = [_issue(99)]

    run(orch._tick())

    assert orch._dispatched == ["T-99"]


# ── Acceptance criterion 3: non-numeric values are ignored, not fatal ───────


def test_non_numeric_values_are_ignored_and_do_not_crash_the_tick(orchestrator):
    """Before the fix this key never matched, so the bad value was never
    read. Lowercasing without validating would turn it into a live
    `TypeError` that aborts dispatch on every tick."""
    orch = orchestrator('{"in progress": "two"}')
    _preload_running(orch, 1)
    orch._fake_linear.candidates = [_issue(99)]

    run(orch._tick())

    assert orch._dispatched == ["T-99"]


# ── Regression: the bug as originally filed ─────────────────────────────────


def test_uncapped_dispatch_on_current_main_would_fail_this(orchestrator):
    """The exact scenario from the ticket: an operator writes the key in
    Linear's own case-sensitive spelling and expects it to cap. Before the
    fix, dispatch happened anyway because the dict was never lowercased."""
    orch = orchestrator('{"In Progress": 1}')
    _preload_running(orch, 1)
    orch._fake_linear.candidates = [_issue(99)]

    run(orch._tick())

    assert orch._dispatched == [], (
        "cap 'In Progress': 1 did not limit dispatch — the YAA-6 regression"
    )
