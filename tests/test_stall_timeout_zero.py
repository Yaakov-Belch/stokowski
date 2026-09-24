"""Reproduction and regression tests for YAA-8: stall_timeout_ms: 0 busy-loops
instead of disabling stall detection.

`stall_monitor()` in runner.py treated `stall_timeout_ms <= 0` as "never kill
the process" (the `stall_timeout_s > 0` guard before the kill), which is the
documented way to disable stall detection. But the sleep interval between
checks was computed as `min(stall_timeout_s / 4, 30)` with no floor, so at
`stall_timeout_ms=0` the monitor called `asyncio.sleep(0)` in a tight loop for
the entire duration of the run instead of idling.

The fix (per the ticket's scope) is to skip creating the stall monitor task
at all when `stall_timeout_ms <= 0`, not to floor the sleep interval — a
positive `stall_timeout_ms` was never observed to busy-loop (the monitor
exits via the kill branch's `return` well before the interval could matter),
so flooring it would be an unjustified, out-of-scope behaviour change.

These tests run the real `run_agent_turn()` against a fake "claude" script.
"""

import stat
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stokowski.config import ClaudeConfig, HooksConfig
from stokowski.models import Issue, RunAttempt
from stokowski.runner import run_agent_turn

SLEEP_SECONDS = 1.0


def _make_fake_claude(tmp_path: Path) -> Path:
    """A stand-in for the `claude` CLI: ignores all args, sleeps, exits 0.

    Emits no stdout, so `last_activity` never advances during the run --
    matching the "silent agent" case stall detection exists for.
    """
    script = tmp_path / "fake_claude.sh"
    script.write_text(f"#!/bin/sh\nsleep {SLEEP_SECONDS}\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _run_once(tmp_path: Path, stall_timeout_ms: int) -> tuple[float, str]:
    """Runs one turn and returns (CPU time consumed while the turn was in
    flight, final attempt.status). A correctly idling/disabled monitor burns
    near-zero CPU; a busy loop burns CPU proportional to SLEEP_SECONDS.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_claude = _make_fake_claude(tmp_path)
    claude_cfg = ClaudeConfig(
        command=str(fake_claude),
        stall_timeout_ms=stall_timeout_ms,
        turn_timeout_ms=60_000,
    )
    hooks_cfg = HooksConfig()
    issue = Issue(id="1", identifier="YAA-8", title="repro")
    attempt = RunAttempt(issue_id="1", issue_identifier="YAA-8", state_name="reproduce")

    import asyncio

    cpu_before = time.process_time()
    asyncio.run(
        run_agent_turn(
            claude_cfg,
            hooks_cfg,
            prompt="irrelevant",
            workspace_path=tmp_path,
            issue=issue,
            attempt=attempt,
        )
    )
    cpu_after = time.process_time()
    return cpu_after - cpu_before, attempt.status


@pytest.mark.parametrize("stall_timeout_ms", [0, -1])
def test_stall_timeout_disabled_does_not_busy_loop(tmp_path, stall_timeout_ms):
    """stall_timeout_ms <= 0 should idle like a monitor that never fires,
    per the spec ('If <= 0, stall detection is disabled') and the acceptance
    criteria ('no stall monitor task is created' for zero and for negative
    values). Instead it burned CPU in a tight asyncio.sleep(0) loop for the
    whole run.
    """
    cpu, status = _run_once(tmp_path / f"disabled-{stall_timeout_ms}", stall_timeout_ms)

    print(f"\nCPU time with stall_timeout_ms={stall_timeout_ms}: {cpu:.3f}s, status={status}")

    assert cpu < 0.2, (
        f"stall_timeout_ms={stall_timeout_ms} burned {cpu:.3f}s of CPU "
        f"busy-looping over a {SLEEP_SECONDS}s run (expected < 0.2s, like a "
        f"disabled monitor should)"
    )
    # A disabled monitor must never kill the process either -- this is the
    # "disabled" half of the contract, not just the CPU half.
    assert status == "succeeded", (
        f"stall_timeout_ms={stall_timeout_ms} should disable stall detection "
        f"entirely, but the turn ended with status={status!r}"
    )


def test_stall_timeout_positive_still_detects_stall(tmp_path):
    """Acceptance criterion: 'When stall_timeout_ms is positive, stall
    detection works as before.' A silent process that outlives a short
    positive stall_timeout_ms must still be killed as stalled.
    """
    cpu, status = _run_once(tmp_path / "positive", stall_timeout_ms=100)

    assert status == "stalled", (
        f"stall_timeout_ms=100 against a {SLEEP_SECONDS}s silent process "
        f"should be killed as stalled, got status={status!r}"
    )
    assert cpu < 0.2, f"stall detection at a positive timeout should not busy-loop either ({cpu:.3f}s CPU)"
