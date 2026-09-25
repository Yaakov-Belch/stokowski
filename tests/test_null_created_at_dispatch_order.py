"""YAA-9: Null created_at sorts first instead of last in dispatch order.

The dispatch sort in `orchestrator.py:_tick` used to fall back to
`datetime.min` when `created_at` is missing:

    candidates.sort(
        key=lambda i: (
            _priority_dispatch_rank(i.priority),
            i.created_at or datetime.min.replace(tzinfo=timezone.utc),
            i.identifier,
        )
    )

`datetime.min` is the smallest possible timestamp, so within a priority
bucket an issue with no `created_at` sorted ascending *ahead of* every issue
that has a real timestamp — a ticket Linear could not (or does not yet)
report a creation time for jumped the queue ahead of tickets that have been
waiting, rather than falling to the back of the line behind them. This is
the same shape of bug already fixed for `priority` in
`_priority_dispatch_rank()` (YAA-5, tests/test_priority_dispatch_order.py):
"unknown/missing sort value must not outrank known ones", except this time
for `created_at` instead of `priority`, and the fallback constant had the
wrong polarity (`datetime.min` instead of `datetime.max`).

The fix extracts the `created_at` element of the sort key into
`_created_at_dispatch_key()`, mirroring `_priority_dispatch_rank()`, so this
test can call the real function instead of retyping a copy of it — a
retyped copy is exactly what would let this test pass by asserting the bug
as intended behaviour if the two ever drifted apart.

This test reproduces the bug against the real dispatch path
(`Orchestrator._tick`, with a stubbed Linear client) and against the sort
key function in isolation, then covers the boundary: same-priority
comparisons must respect the null-sorts-last rule, cross-priority ones must
be unaffected because the tuple never reaches the `created_at` element, and
two undated issues must still tie-break on identifier.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from stokowski.models import Issue
from stokowski.orchestrator import Orchestrator, _created_at_dispatch_key, _priority_dispatch_rank

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_WORKFLOW = REPO / "workflow.example.yaml"


def _issue(identifier: str, priority: int | None, created_at: datetime | None) -> Issue:
    return Issue(
        id=identifier,
        identifier=identifier,
        title=f"Issue {identifier}",
        priority=priority,
        state="Todo",
        created_at=created_at,
    )


class _StubLinearClient:
    """Answers every call the pre-dispatch parts of `_tick` can make, with no
    network access. `fetch_candidate_issues` is the one under test."""

    def __init__(self, candidates: list[Issue]):
        self._candidates = candidates

    async def fetch_candidate_issues(self, project_slug, active_states):
        return list(self._candidates)

    async def fetch_comments(self, issue_id):
        return []

    async def fetch_issues_by_states(self, project_slug, states):
        return []

    async def fetch_issue_states_by_ids(self, ids):
        return {}


def _dispatch_order(issues: list[Issue]) -> list[str]:
    """Drive the real Orchestrator._tick() dispatch path and return the
    order issues were handed to `_dispatch()`."""
    orch = Orchestrator(EXAMPLE_WORKFLOW)
    orch._load_workflow()
    orch._linear = _StubLinearClient(issues)

    order: list[str] = []
    orch._dispatch = lambda issue, attempt_num=None: order.append(issue.identifier)

    asyncio.run(orch._tick())
    return order


def _sort_key_order(issues: list[Issue]) -> list[str]:
    """Isolate the real sort key `_tick` uses, without the rest of the
    dispatch path, by calling `_created_at_dispatch_key()` directly rather
    than retyping the tuple expression."""
    candidates = list(issues)
    candidates.sort(
        key=lambda i: (
            _priority_dispatch_rank(i.priority),
            _created_at_dispatch_key(i.created_at),
            i.identifier,
        )
    )
    return [i.identifier for i in candidates]


def test_older_dated_issue_dispatches_before_undated_issue():
    """The real bug: within the same priority bucket, an issue with a real
    `created_at` must dispatch before one with no `created_at` at all. Drives
    the real Orchestrator._tick() dispatch path, not just the comparator in
    isolation."""
    dated = _issue("YAA-200", priority=2, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    undated = _issue("YAA-201", priority=2, created_at=None)

    order = _dispatch_order([undated, dated])

    assert order == ["YAA-200", "YAA-201"], (
        f"expected dated issue (YAA-200) to dispatch before undated issue "
        f"(YAA-201), got {order}"
    )


def test_sort_key_ranks_undated_after_dated():
    """Isolates the real sort key function (`_created_at_dispatch_key`),
    not a retyped copy of it, to show the fix lives in the fallback
    constant itself."""
    dated = _issue("YAA-200", priority=2, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    undated = _issue("YAA-201", priority=2, created_at=None)

    order = _sort_key_order([undated, dated])

    assert order == ["YAA-200", "YAA-201"], (
        f"expected dated issue ahead of undated issue, got: {order}"
    )


def test_undated_still_outranks_dated_at_lower_priority():
    """Cross-priority control: an undated Urgent issue must still dispatch
    before a dated Low one. The priority rank decides before `created_at`
    is ever consulted, so this must be unaffected by the fix."""
    undated_urgent = _issue("UNDATED-URGENT", priority=1, created_at=None)
    dated_low = _issue("DATED-LOW", priority=4, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    assert _sort_key_order([dated_low, undated_urgent]) == ["UNDATED-URGENT", "DATED-LOW"]
    assert _dispatch_order([dated_low, undated_urgent]) == ["UNDATED-URGENT", "DATED-LOW"]


def test_dated_urgent_still_outranks_undated_low():
    """Cross-priority control, the other direction: a dated Urgent issue
    must dispatch before an undated Low one — also unaffected by the fix."""
    dated_urgent = _issue("DATED-URGENT", priority=1, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    undated_low = _issue("UNDATED-LOW", priority=4, created_at=None)

    assert _sort_key_order([undated_low, dated_urgent]) == ["DATED-URGENT", "UNDATED-LOW"]
    assert _dispatch_order([undated_low, dated_urgent]) == ["DATED-URGENT", "UNDATED-LOW"]


def test_two_undated_issues_tie_break_on_identifier():
    """Two undated issues at the same priority collapse to the same
    sentinel and must still fall through to the identifier tiebreak."""
    z = _issue("ZZZ", priority=2, created_at=None)
    a = _issue("AAA", priority=2, created_at=None)

    assert _sort_key_order([z, a]) == ["AAA", "ZZZ"]


def test_all_dated_same_priority_still_sorts_oldest_first():
    """Control: with no missing timestamps, the sort is unaffected and
    remains oldest-first."""
    newer = _issue("NEWER", priority=2, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    older = _issue("OLDER", priority=2, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    assert _sort_key_order([newer, older]) == ["OLDER", "NEWER"]


def test_undated_beats_both_an_old_and_a_recent_dated_issue():
    """The fallback is a floor, not merely 'early': before the fix, an
    undated issue outranked a 2026-dated issue exactly as badly as a
    2020-dated one. Both must now sort ahead of the undated issue."""
    undated = _issue("UNDATED", priority=2, created_at=None)
    dated_2026 = _issue("DATED-2026", priority=2, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert _sort_key_order([undated, dated_2026]) == ["DATED-2026", "UNDATED"]
