"""YAA-5: Priority 0 (No priority) dispatches before Urgent (1).

Linear's `priority` field is a raw int passed straight through by
`linear.py:_issue_from_node` (stokowski/linear.py:219-224): 0 = No priority,
1 = Urgent, 2 = High, 3 = Medium, 4 = Low. That is Linear's own ordering, not
an internal Stokowski convention — confirmed live against Linear's GraphQL
API (`issuePriorityValues` and the `Issue.priority` field description both
state it verbatim) during the ground-check stage of this ticket.

The dispatch sort in `orchestrator.py:_tick` used to sort candidates ascending
by that raw integer:

    candidates.sort(
        key=lambda i: (
            i.priority if i.priority is not None else 999,
            i.created_at or datetime.min.replace(tzinfo=timezone.utc),
            i.identifier,
        )
    )

Ascending-by-raw-value treated "No priority" (0) as higher priority than
"Urgent" (1), so an unprioritised backlog ticket jumped the queue ahead of a
ticket someone explicitly marked Urgent. `Issue.priority` is non-nullable on
Linear's side, so an unset priority always arrives as `0`, never `None` — the
old code's `else 999` branch was dead for every real candidate, and the live
`0` value fell straight into the numeric comparison instead.

The fix, `_priority_dispatch_rank()` in orchestrator.py, maps priorities 1-4
to themselves (genuinely ascending by urgency) and everything else — 0, None,
and any out-of-range value — to a single rank after that bucket, per the
Symphony spec 8.2 wording quoted on this ticket: "priority ascending for
values 1..4; all other integers and null sort after that bucket".

This test reproduces the original bug against the real dispatch path
(Orchestrator._tick, with a stubbed Linear client) and against the sort key
in isolation, and then asserts the fixed full rank order.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from stokowski.models import Issue
from stokowski.orchestrator import Orchestrator, _priority_dispatch_rank

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_WORKFLOW = REPO / "workflow.example.yaml"


def _issue(identifier: str, priority: int | None, created_at: datetime) -> Issue:
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


def test_urgent_dispatches_before_no_priority():
    """The real bug: an Urgent (1) ticket must be queued ahead of a No
    priority (0) ticket. Drives the real Orchestrator._tick() dispatch path,
    not just the comparator in isolation."""
    same_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    no_priority = _issue("YAA-100", priority=0, created_at=same_time)
    urgent = _issue("YAA-101", priority=1, created_at=same_time)

    orch = Orchestrator(EXAMPLE_WORKFLOW)
    orch._load_workflow()
    orch._linear = _StubLinearClient([no_priority, urgent])

    dispatch_order: list[str] = []
    orch._dispatch = lambda issue, attempt_num=None: dispatch_order.append(issue.identifier)

    asyncio.run(orch._tick())

    assert dispatch_order == ["YAA-101", "YAA-100"], (
        f"expected Urgent (YAA-101) to dispatch before No priority (YAA-100), "
        f"got {dispatch_order}"
    )


def test_sort_key_ranks_urgent_above_no_priority():
    """Isolates `_priority_dispatch_rank()` (orchestrator.py) without the rest
    of _tick, to show the fix is in the key itself, not eligibility or
    concurrency logic around it."""
    same_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    no_priority = _issue("YAA-100", priority=0, created_at=same_time)
    urgent = _issue("YAA-101", priority=1, created_at=same_time)

    candidates = [no_priority, urgent]
    candidates.sort(
        key=lambda i: (
            _priority_dispatch_rank(i.priority),
            i.created_at or datetime.min.replace(tzinfo=timezone.utc),
            i.identifier,
        )
    )

    ordered = [i.identifier for i in candidates]
    assert ordered == ["YAA-101", "YAA-100"], (
        f"expected Urgent ahead of No priority, got: {ordered}"
    )


def test_full_priority_rank_order():
    """Urgent < High < Medium < Low < No priority — the full order the
    Symphony spec 8.2 wording quoted on this ticket describes, not just the
    single pair the ticket reported."""
    same_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    urgent = _issue("YAA-1", priority=1, created_at=same_time)
    high = _issue("YAA-2", priority=2, created_at=same_time)
    medium = _issue("YAA-3", priority=3, created_at=same_time)
    low = _issue("YAA-4", priority=4, created_at=same_time)
    no_priority = _issue("YAA-5", priority=0, created_at=same_time)

    candidates = [no_priority, low, medium, high, urgent]
    candidates.sort(
        key=lambda i: (
            _priority_dispatch_rank(i.priority),
            i.created_at or datetime.min.replace(tzinfo=timezone.utc),
            i.identifier,
        )
    )

    assert [i.identifier for i in candidates] == ["YAA-1", "YAA-2", "YAA-3", "YAA-4", "YAA-5"]


def test_null_and_out_of_range_priority_sort_with_no_priority():
    """A missing priority (defensive — `Issue.priority` is non-nullable on
    Linear's side, so `None` only reaches here via linear.py's own fallback
    when it cannot parse a raw value as an int) and an out-of-range value
    must land in the same bucket as No priority (0), not scattered by raw
    integer value. This is the 'all other integers and null sort after that
    bucket' half of the spec wording quoted on the ticket."""
    assert _priority_dispatch_rank(None) == _priority_dispatch_rank(0)
    assert _priority_dispatch_rank(7) == _priority_dispatch_rank(0)
    assert _priority_dispatch_rank(-1) == _priority_dispatch_rank(0)

    # And that shared bucket sorts after every real 1-4 rank.
    for real_priority in (1, 2, 3, 4):
        assert _priority_dispatch_rank(real_priority) < _priority_dispatch_rank(0)
