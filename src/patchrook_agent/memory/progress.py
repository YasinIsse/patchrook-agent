"""A human-readable projection of structured working memory."""

from .state import WorkingMemory


def render_progress(memory: WorkingMemory) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None recorded."

    latest = memory.test_results[-1] if memory.test_results else {}
    lines = [
        "# Task Progress", "", "## Objective", "", memory.objective,
        "", "## Completed", "", bullets(memory.completed_steps[-12:]),
        "", "## In Progress", "", f"- {memory.active_task}",
        "", "## Next Steps", "", bullets(memory.remaining_steps),
        "", "## Relevant Files", "", bullets(memory.relevant_files[:30]),
        "", "## Modified Files", "", bullets(memory.modified_files),
        "", "## Verification", "", f"- Status: {memory.verification}",
    ]
    if latest:
        lines.append(f"- Latest test: {latest.get('path', '.')}; exit {latest.get('exit_code', 'unknown')}.")
    for error in memory.unresolved_errors:
        detail = error.get("error", {})
        message = detail.get("message", "") if isinstance(detail, dict) else str(detail)
        if not message:
            message = error.get("stderr") or error.get("stdout") or error.get("status", "Failed")
        # The JSON memory preserves complete errors; TASKS is a readable pointer.
        readable = " ".join(str(message).split())
        lines.append(f"- Outstanding issue: {readable[:300]}" + ("… (see state.json)" if len(readable) > 300 else ""))
    if memory.task_state.has_activity:
        task = memory.task_state
        lines.extend(["", "## Coordination", "", f"- Phase: {task.phase}"])
        if task.active_goal:
            lines.append(f"- Active goal: {task.active_goal}")
        if task.trigger:
            lines.append(f"- Review required: {task.trigger}")
        lines.extend(["", "## Recent Actions", ""])
        for item in task.history[-6:]:
            detail = " ".join(str(item.get(key, "")) for key in ("tool", "path", "outcome")).strip()
            if not detail:
                detail = str(item.get("event", "event"))
                if item.get("proposal_id"):
                    detail += " " + str(item["proposal_id"])
                if item.get("reason"):
                    detail += ": " + str(item["reason"])
            lines.append(f"- Step {item.get('step', 0)}: {detail}")
        if task.history_omitted:
            lines.append(f"- {task.history_omitted} older brief actions remain in the full event log.")
    lines.extend(["", "## Notes", "", bullets(memory.decisions + memory.blockers)])
    if memory.loaded_skills:
        lines.append("- Loaded skills: " + ", ".join(memory.loaded_skills))
    lines.extend([f"- Last completed agent step: {memory.last_completed_step}", ""])
    return "\n".join(lines)
