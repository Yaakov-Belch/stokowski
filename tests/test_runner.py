"""Tests for run_agent_turn's final-status decision (YAA-7).

CLAUDE.md documents that a `result` event with `is_error: true` (e.g.
`error_max_turns`) still makes the CLI process exit 0. `events.py` correctly
records that on `attempt.result_is_error`, but until this fix `run_agent_turn`
decided `attempt.status` from `proc.returncode` alone and never consulted it,
so an in-band failure with a clean exit was reported and acted on as success.

There was no test file for this decision at all before this one - the
producer (`events.py`) was covered, the consumer (`runner.py`) never was.

Each test spawns a real, minimal `claude` executable and drives it through
the real `run_agent_turn` (no mocking of the function under test), following
this repo's convention of driving async code via `asyncio.run` rather than
pytest-asyncio (see test_state_comments.py).
"""

from __future__ import annotations

import asyncio
import json
import stat
import sys
from pathlib import Path

from stokowski.config import ClaudeConfig, HooksConfig
from stokowski.models import Issue, RunAttempt
from stokowski.runner import run_agent_turn


def make_fake_claude(tmp_path: Path, events: list[dict], exit_code: int = 0) -> Path:
    """An executable standing in for `claude` that emits the given NDJSON
    events on stdout and exits with the given code."""
    script = tmp_path / "claude"
    # Each event is pre-serialized to a JSON string, then that string is
    # itself dumped so the generated script embeds it as a plain literal -
    # avoids any quoting ambiguity between the event JSON and the script.
    print_lines = "\n".join(f"print({json.dumps(json.dumps(e))})" for e in events)
    body = f"import sys\n{print_lines}\nsys.exit({exit_code})\n"
    script.write_text(f"#!{sys.executable}\n{body}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


async def _run(fake_claude: Path, tmp_path: Path) -> RunAttempt:
    claude_cfg = ClaudeConfig(command=str(fake_claude))
    hooks_cfg = HooksConfig()
    issue = Issue(id="issue-1", identifier="YAA-7", title="repro")
    attempt = RunAttempt(issue_id=issue.id, issue_identifier=issue.identifier)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    return await run_agent_turn(
        claude_cfg=claude_cfg,
        hooks_cfg=hooks_cfg,
        prompt="do the thing",
        workspace_path=workspace,
        issue=issue,
        attempt=attempt,
    )


def test_in_band_max_turns_error_with_clean_exit_is_marked_failed(tmp_path: Path):
    """The bug: an in-band failure (is_error=true) that exits 0 must not be
    reported as succeeded."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "fake-session-id",
         "model": "claude-sonnet-4-6", "tools": []},
        {"type": "result", "subtype": "error_max_turns", "is_error": True,
         "session_id": "fake-session-id", "usage": {}, "num_turns": 20,
         "stop_reason": "max_turns", "total_cost_usd": 0.01,
         "permission_denials": []},
    ]
    fake_claude = make_fake_claude(tmp_path, events, exit_code=0)
    result = asyncio.run(_run(fake_claude, tmp_path))

    assert result.result_is_error is True
    assert result.status == "failed"
    assert "error_max_turns" in (result.error or "")


def test_clean_success_is_still_marked_succeeded(tmp_path: Path):
    """Boundary: a real success (is_error absent/false, exit 0) must still be
    reported as succeeded - the fix must not over-correct."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "fake-session-id",
         "model": "claude-sonnet-4-6", "tools": []},
        {"type": "result", "subtype": "success", "is_error": False,
         "session_id": "fake-session-id", "usage": {}, "num_turns": 2,
         "stop_reason": "end_turn", "total_cost_usd": 0.01,
         "permission_denials": []},
    ]
    fake_claude = make_fake_claude(tmp_path, events, exit_code=0)
    result = asyncio.run(_run(fake_claude, tmp_path))

    assert result.result_is_error is False
    assert result.status == "succeeded"


def test_tool_error_with_clean_exit_does_not_trigger_in_band_failure(tmp_path: Path):
    """Boundary: an ordinary failed tool call, followed by a clean result,
    must not be mistaken for an in-band failure. Only a `result` event with
    is_error=true sets attempt.result_is_error - tool_result errors only
    increment tool_error_count (events.py's _handle_user)."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "fake-session-id",
         "model": "claude-sonnet-4-6", "tools": []},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "false"}}
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
             "content": "command failed"}
        ]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "session_id": "fake-session-id", "usage": {}, "num_turns": 3,
         "stop_reason": "end_turn", "total_cost_usd": 0.01,
         "permission_denials": []},
    ]
    fake_claude = make_fake_claude(tmp_path, events, exit_code=0)
    result = asyncio.run(_run(fake_claude, tmp_path))

    assert result.tool_error_count == 1
    assert result.result_is_error is False
    assert result.status == "succeeded"


def test_nonzero_exit_is_still_marked_failed(tmp_path: Path):
    """Existing behaviour preserved: a genuine process-level failure (no
    result event at all, non-zero exit) is still failed."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "fake-session-id",
         "model": "claude-sonnet-4-6", "tools": []},
    ]
    fake_claude = make_fake_claude(tmp_path, events, exit_code=1)
    result = asyncio.run(_run(fake_claude, tmp_path))

    assert result.result_is_error is False
    assert result.status == "failed"
    assert "Exit code 1" in (result.error or "")


def test_in_band_error_message_survives_a_later_nonzero_exit(tmp_path: Path):
    """Sibling to the main fix: if the CLI streams an in-band error result
    and then the process itself dies non-zero, the specific reason events.py
    already recorded must not be clobbered by the generic exit-code message."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "fake-session-id",
         "model": "claude-sonnet-4-6", "tools": []},
        {"type": "result", "subtype": "error_during_execution", "is_error": True,
         "session_id": "fake-session-id", "usage": {}, "num_turns": 5,
         "stop_reason": "error", "total_cost_usd": 0.01,
         "permission_denials": []},
    ]
    fake_claude = make_fake_claude(tmp_path, events, exit_code=1)
    result = asyncio.run(_run(fake_claude, tmp_path))

    assert result.result_is_error is True
    assert result.status == "failed"
    assert "error_during_execution" in (result.error or "")
    assert "Exit code" not in (result.error or "")
