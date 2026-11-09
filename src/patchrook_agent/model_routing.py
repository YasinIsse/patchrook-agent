"""Deterministic escalation signals; classification never grants edit authority."""

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


@dataclass(frozen=True)
class CodeFailureContext:
    """Workflow-verified, ephemeral provenance; never constructed from model output."""
    baseline_evidence_id: str
    workspace: str
    modified_sources: tuple[str, ...]
    check_name: str
    trusted_argv: tuple[str, ...]
    corroborated_exception: tuple[str, str] | None = None
    traceback_evidence_id: str | None = None


def reported_code_exception(text):
    """Canonicalize only known exception prefixes and whitespace; conflicts fail closed."""
    pattern = (r"(?m)^[ \t]*(?:(?:E[ \t]+)|(?:FAIL [^:\r\n]{1,120}:[ \t]*))?"
               r"(ImportError|ModuleNotFoundError|NameError|TypeError):[ \t]*([^\r\n]*)$")
    values = {(match.group(1), " ".join(match.group(2).split())) for match in re.finditer(pattern, text)}
    return next(iter(values)) if len(values) == 1 and next(iter(values))[1] else None


def traceback_paths(text):
    pattern = (r'(?m)^[ \t]*(?:File "(?P<standard>[^"\r\n]+)", line [1-9]\d*(?:,|$)'
               r'|(?P<pytest>[^\r\n:]+\.py):[1-9]\d*:[ \t]+in[ \t]+\S)')
    return [match.group("standard") or match.group("pytest") for match in re.finditer(pattern, text)]


def _code_origin_failure(result, context, text):
    if (not isinstance(context, CodeFailureContext)
            or not isinstance(context.baseline_evidence_id, str) or not isinstance(context.workspace, str)
            or not isinstance(context.modified_sources, tuple)
            or any(not isinstance(path, str) for path in context.modified_sources)
            or result.get("success")
            or result.get("truncated") or result.get("verification_kind") != "configured_command"
            or result.get("name") != context.check_name
            or not isinstance(result.get("argv"), (list, tuple))
            or tuple(result["argv"]) != context.trusted_argv
            or result.get("exit_code") not in {1, 2, 4}
            or result.get("exit_code") == 4 and not any(
                str(part).rsplit("/", 1)[-1] in {"pytest", "py.test"} for part in context.trusted_argv)
            or not re.fullmatch(r"ev_[0-9a-f]{24}", context.baseline_evidence_id)
            or reported_code_exception(text) is None):
        return False
    root = PurePosixPath(context.workspace)
    if not root.is_absolute() or ".." in root.parts:
        return False
    paths = set(context.modified_sources)
    frames = traceback_paths(text)
    if not frames:
        # Summary text contributes identity only, never a path or inferred line.
        return bool(re.search(r"(?m)^[ \t]*FAIL [^:\r\n]{1,120}:[ \t]*(?:ImportError|ModuleNotFoundError|NameError|TypeError):", text)
            and context.corroborated_exception == reported_code_exception(text)
            and isinstance(context.traceback_evidence_id, str)
            and re.fullmatch(r"ev_[0-9a-f]{24}", context.traceback_evidence_id))
    # An ancestor importing an unrelated broken dependency is insufficient.
    # The innermost reported frame must identify the changed repository code.
    path = PurePosixPath(frames[-1])
    if ".." in path.parts:
        return False
    try:
        relative = path.relative_to(root) if path.is_absolute() else path
    except ValueError:
        return False
    return relative.suffix == ".py" and relative.as_posix() in paths


def infrastructure_failure(result, *, code_failure_context=None):
    """Default conservative classification, with caller-verified code provenance."""
    if result.get("timed_out") or result.get("error"):
        return True
    argv = result.get("argv", [])
    pytest_command = isinstance(argv, (list, tuple)) and any(
        str(part).rsplit("/", 1)[-1] in {"pytest", "py.test"} for part in argv)
    if (result.get("exit_code") in {126, 127, 137, -9}
            or pytest_command and result.get("exit_code") in {3, 5}):
        return True
    text = str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))
    lower = text.lower()
    if any(marker in lower for marker in (
        "command not found", "connectionrefusederror", "connectionerror", "permissionerror",
        "memoryerror", "out of memory", "no such file or directory",
    )):
        return True
    if pytest_command and result.get("exit_code") == 4 and (
            re.search(r"(?im)^[ \t]*(?:usage:|pytest: error:|ERROR: (?:usage|unrecognized arguments|unknown arguments))", text)
            or "unrecognized arguments" in lower or "unknown arguments" in lower):
        return True
    if _code_origin_failure(result, code_failure_context, text):
        return False
    return (bool(pytest_command and result.get("exit_code") in {2, 4})
            or any(marker in lower for marker in (
                "modulenotfounderror", "importerror", "no module named", "error collecting")))


def failure_fingerprint(name, result):
    text = str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))
    failed_tests = sorted(set(re.findall(r"^FAILED\s+(\S+)", text, re.MULTILINE)))
    if failed_tests:
        text = "\n".join(failed_tests)
    else:
        # Elapsed time, addresses and temporary paths are not new repair failures.
        text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:seconds?|s)\b", "<time>", text)
        text = re.sub(r"0x[0-9a-fA-F]+", "<address>", text)
        text = re.sub(r"/tmp/[^\s\"']+", "<temporary-path>", text)
    return hashlib.sha256((name + "\n" + text).encode()).hexdigest()


def _observation_keys(name, arguments, result, revision):
    # Count acquired facts rather than request spelling. Changing a page
    # size, offset, or query wording does not make already-seen source new.
    stable = {key: value for key, value in result.items() if key not in {
        "duration_seconds", "evidence_id", "source_evidence_id", "name",
    }}
    if name == "run_check":
        stable = {"revision": revision, "passed": bool(result.get("success")),
                  "failure": failure_fingerprint(str(arguments.get("name", "")), result)}
    identities = []
    path = result.get("path", arguments.get("path", "."))
    if result.get("success") and name == "read_file" and isinstance(result.get("content"), str):
        start = result.get("start_line", arguments.get("start_line", 1))
        for index, line in enumerate(result["content"].splitlines()):
            number = start + index
            text = re.sub(r"^\d+: ", "", line)
            identities.append(["source", path, result.get("source_sha256"), number, text])
    elif result.get("success") and name == "search_code":
        identities = [["match", item.get("path"), item.get("line_number"), item.get("text")]
                      for item in result.get("matches", []) if isinstance(item, dict)]
    elif result.get("success") and name == "list_files":
        identities = [["path", path, item] for item in result.get("entries", [])]
    if not identities:
        identities = [[name, {k: v for k, v in arguments.items()
                              if k not in {"question", "offset", "max_results", "max_entries"}}, stable]]
    keys = list(dict.fromkeys(hashlib.sha256(json.dumps(identity, sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest() for identity in identities))
    if len(keys) > 128:
        # A single oversized view must not evict its own early facts and
        # appear new forever. Summarize the complete set; smaller views
        # retain the existing per-fact overlap tracking.
        keys = [hashlib.sha256(json.dumps(["oversized_fact_set", sorted(keys)],
                separators=(",", ":")).encode()).hexdigest()]
    return keys


@dataclass
class RoutingProgress:
    invalid_responses: int = 0
    repeated_actions: int = 0
    observations: list[str] = field(default_factory=list)
    failed_repairs: list[list] = field(default_factory=list)

    def invalid(self, config):
        self.invalid_responses += 1
        return "invalid_responses" if self.invalid_responses >= config.routing_max_invalid_responses else None

    def observe(self, name, arguments, result, revision, config, *, code_failure_context=None):
        if (name == "apply_edits" and not result.get("success") and not result.get("timed_out")
                and result.get("error", {}).get("code") in {
                    "source_not_presented", "overlapping_edits", "invalid_arguments", "ambiguous_match",
                    "no_match", "duplicate_edit", "no_change", "python_syntax_error", "python_duplicate_function",
                    "python_syntax_validation_limit"}):
            # A valid transport can still carry a malformed or ungrounded edit.
            # Reuse the existing allowance and checkpoint field; new evidence or
            # an accepted mutation can repay the rejected-action debt.
            return self.invalid(config)
        if name == "run_check":
            if infrastructure_failure(result, code_failure_context=code_failure_context):
                self.repeated_actions = 0
                return None
            if result.get("success"):
                self.failed_repairs = [item for item in self.failed_repairs if item[0] != arguments.get("name")]
            elif revision > 0 and result.get("verification_kind") == "configured_command":
                fingerprint = failure_fingerprint(str(arguments.get("name", "")), result)
                record = [str(arguments.get("name", "")), revision, fingerprint]
                if record not in self.failed_repairs:
                    self.failed_repairs.append(record)
                    self.failed_repairs = self.failed_repairs[-128:]
                if sum(item[2] == fingerprint for item in self.failed_repairs) >= config.routing_max_failed_repairs:
                    return "failed_repairs"
        if name == "apply_edits" and result.get("success") and result.get("modified_files"):
            self.invalid_responses = 0
            self.repeated_actions = 0
            return None
        keys = _observation_keys(name, arguments, result, revision)
        new = [key for key in keys if key not in self.observations]
        if not new:
            self.repeated_actions += 1
        else:
            self.observations = [*self.observations, *new][-128:]
            self.repeated_actions = 0
            # A valid transport or a repeated read is not recovery from a
            # rejected proposal. Forgive the debt only for newly acquired
            # evidence or the accepted mutation handled above.
            if (result.get("success") and not result.get("stale")
                    and name in {"read_file", "search_code", "list_files", "read_evidence"}
                    or name == "run_check" and result.get("verification_kind") == "configured_command"):
                self.invalid_responses = 0
        return "repeated_actions" if self.repeated_actions >= config.routing_max_repeated_actions else None

    def seed(self, name, arguments, result, revision):
        """Remember actual startup evidence without counting a coding step.

        Automatic baseline checks and declaration reads precede the first
        model request. Their facts are already available to the coder, but
        they must not consume a repair allowance or trigger a stall signal.
        """
        if not (result.get("success") and name in {"read_file", "search_code", "list_files"}
                or name == "run_check" and result.get("verification_kind") == "configured_command"
                and not infrastructure_failure(result)):
            return
        keys = _observation_keys(name, arguments, result, revision)
        self.observations = [*self.observations,
                             *(key for key in keys if key not in self.observations)][-128:]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"invalid_responses", "repeated_actions", "observations", "failed_repairs"}:
            raise ValueError("Invalid routing progress checkpoint")
        for key in ("invalid_responses", "repeated_actions"):
            if type(value[key]) is not int or not 0 <= value[key] <= 100_000:
                raise ValueError("Invalid routing progress counter")
        observations, failures = value["observations"], value["failed_repairs"]
        if (not isinstance(observations, list) or len(observations) > 128
                or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in observations)):
            raise ValueError("Invalid routing observations")
        if (not isinstance(failures, list) or len(failures) > 128 or any(
                not isinstance(item, list) or len(item) != 3 or not isinstance(item[0], str)
                or type(item[1]) is not int or item[1] < 1 or not isinstance(item[2], str)
                or not re.fullmatch(r"[0-9a-f]{64}", item[2]) for item in failures)):
            raise ValueError("Invalid routing failure history")
        return cls(**value)


def restore_routing(value, config, active_model, steps, attempts):
    """Validate a serialized routing checkpoint without touching models or files.

    Routing state can select a model and schedule an unload on resume. Validate
    its complete shape and single repair episode before returning any state
    to the runtime. Decision scores are audit data, never edit authority.
    """
    keys = {"schema_version", "initial_model", "active_model", "escalation_model",
            "initial_decided", "initial_route", "pending_switch", "switches",
            "model_calls", "model_usage", "progress", "repair"}
    if not isinstance(value, dict):
        raise ValueError("Invalid routing checkpoint fields")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 2:
        raise ValueError("Unsupported routing checkpoint version; start a fresh routed run")
    if set(value) != keys:
        raise ValueError("Invalid routing checkpoint fields")
    if (type(steps) is not int or type(attempts) is not int
            or not 0 <= steps <= attempts):
        raise ValueError("Invalid routing execution counters")

    def local_model(name):
        return (isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", name)
                and ":cloud" not in name)

    for key in ("initial_model", "active_model", "escalation_model"):
        if not local_model(value[key]):
            raise ValueError("Invalid local model in routing checkpoint")
    base, escalation = value["initial_model"], value["escalation_model"]
    known_models = {base, escalation}
    if (base == escalation or escalation != config.escalation_model
            or value["active_model"] != active_model or active_model not in known_models):
        raise ValueError("Routing checkpoint model identities do not match execution")
    if type(value["initial_decided"]) is not bool:
        raise ValueError("Invalid initial routing state")

    def nonnegative(number):
        if type(number) not in (int, float) or number < 0:
            return False
        try:
            return math.isfinite(number)
        except OverflowError:
            return False

    route = value["initial_route"]
    labels = {"small", "large", "uncertain"}
    route_keys = {"label", "ranking", "scores", "fallback_reason", "elapsed_seconds",
                  "inference_seconds", "management_seconds", "input_tokens", "cache_hit",
                  "load_seconds", "management_events", "called", "error"}
    if not value["initial_decided"]:
        if route is not None:
            raise ValueError("Undecided routing checkpoint contains a decision")
    else:
        if (not isinstance(route, dict) or not {"label", "called"} <= set(route)
                or not set(route) <= route_keys or not isinstance(route["label"], str)
                or route["label"] not in labels):
            raise ValueError("Invalid initial routing decision")
        for key in ("called", "cache_hit"):
            if key in route and type(route[key]) is not bool:
                raise ValueError("Invalid routing decision flag")
        for key in ("elapsed_seconds", "inference_seconds", "management_seconds", "load_seconds"):
            if key in route and route[key] is not None and not nonnegative(route[key]):
                raise ValueError("Invalid routing decision timing")
        if "input_tokens" in route and route["input_tokens"] is not None and (
                type(route["input_tokens"]) is not int or route["input_tokens"] < 0):
            raise ValueError("Invalid routing decision tokens")
        for key in ("fallback_reason", "error"):
            if key in route and route[key] is not None and (
                    not isinstance(route[key], str) or len(route[key]) > 1000):
                raise ValueError("Invalid routing decision text")
        fallback = bool(route.get("fallback_reason"))
        if fallback and route["label"] != "small":
            raise ValueError("Routing fallback must retain the base coder")
        if "ranking" in route:
            ranking = route["ranking"]
            if (not isinstance(ranking, (list, tuple)) or len(ranking) > 12
                    or any(not isinstance(item, str) or not 1 <= len(item) <= 48 for item in ranking)
                    or (not fallback and (len(ranking) != 3 or set(ranking) != labels))):
                raise ValueError("Invalid routing decision ranking")
        if "scores" in route:
            scores = route["scores"]
            if (not isinstance(scores, dict) or len(scores) > 12
                    or any(not isinstance(key, str) or not 1 <= len(key) <= 48 for key in scores)
                    or any(not nonnegative(score) or score > 1 for score in scores.values())
                    or (not fallback and set(scores) != labels)):
                raise ValueError("Invalid routing decision scores")
        if "management_events" in route and (
                not isinstance(route["management_events"], (list, tuple))
                or len(route["management_events"]) > 32
                or any(not isinstance(item, str) or len(item) > 200 for item in route["management_events"])):
            raise ValueError("Invalid routing management events")

    repair = value["repair"]
    repair_keys = {"status", "reason", "calls_used", "call_inflight", "pending_checks",
                   "last_revision", "source_ranges"}
    statuses = {"idle", "requested", "active", "verified", "exhausted", "blocked"}
    if (not isinstance(repair, dict) or set(repair) != repair_keys
            or not isinstance(repair["status"], str) or repair["status"] not in statuses
            or not isinstance(repair["reason"], str) or len(repair["reason"]) > 1000
            or type(repair["calls_used"]) is not int
            or not 0 <= repair["calls_used"] <= min(config.routing_max_repair_calls, attempts)
            or type(repair["call_inflight"]) is not bool
            or type(repair["last_revision"]) is not int or not 0 <= repair["last_revision"] <= steps):
        raise ValueError("Invalid repair episode checkpoint")
    pending_checks = repair["pending_checks"]
    trusted_checks = {f"check_{index}" for index in range(1, len(config.evidence_test_commands) + 1)}
    if (not isinstance(pending_checks, list) or len(pending_checks) > 16
            or any(not isinstance(name, str) or name not in trusted_checks for name in pending_checks)
            or len(pending_checks) != len(set(pending_checks))):
        raise ValueError("Invalid pending repair checks")
    ranges = repair["source_ranges"]
    if not isinstance(ranges, list) or len(ranges) > 16:
        raise ValueError("Invalid repair source ranges")
    for source in ranges:
        if (not isinstance(source, dict) or set(source) != {"path", "start_line", "end_line"}
                or not isinstance(source["path"], str) or not source["path"]
                or len(source["path"]) > 4096 or "\x00" in source["path"] or "\\" in source["path"]
                or PurePosixPath(source["path"]).is_absolute()
                or ".." in PurePosixPath(source["path"]).parts
                or not PurePosixPath(source["path"]).parts
                or type(source["start_line"]) is not int or source["start_line"] < 1
                or type(source["end_line"]) is not int or source["end_line"] < source["start_line"]):
            raise ValueError("Invalid repair source range")
    status = repair["status"]
    if status in {"idle", "requested"} and (repair["calls_used"] or repair["call_inflight"] or pending_checks):
        raise ValueError("Unstarted repair episode contains calls or checks")
    if repair["call_inflight"] and (status != "active" or active_model != escalation or not repair["calls_used"]):
        raise ValueError("Invalid inflight repair call")
    if pending_checks and (status != "active" or active_model != escalation or not repair["calls_used"]
                           or repair["call_inflight"]):
        raise ValueError("Repair checks require a completed repair generation")
    if status == "verified" and (not repair["calls_used"] or repair["call_inflight"] or pending_checks):
        raise ValueError("Verified repair episode retains unresolved work")
    if status in {"exhausted", "blocked"} and (repair["call_inflight"] or pending_checks):
        raise ValueError("Terminal repair episode retains pending work")

    reasons = {"initial_large", "failed_repairs", "invalid_responses", "repeated_actions"}
    switches = value["switches"]
    if not isinstance(switches, list) or len(switches) > 2:
        raise ValueError("Routing checkpoint permits one repair episode and one return")
    previous_step = 0
    for index, record in enumerate(switches):
        if (not isinstance(record, dict)
                or set(record) != {"from_model", "to_model", "reason", "step", "management_seconds"}
                or record["from_model"] != (base if index == 0 else escalation)
                or record["to_model"] != (escalation if index == 0 else base)
                or not isinstance(record["reason"], str)
                or (record["reason"] not in reasons if index == 0 else record["reason"] != "repair_verified")
                or type(record["step"]) is not int or not previous_step <= record["step"] <= steps
                or not nonnegative(record["management_seconds"])):
            raise ValueError("Invalid routing switch checkpoint")
        previous_step = record["step"]
    if (active_model == escalation) != (len(switches) == 1):
        raise ValueError("Routing switch history does not match active model")
    if len(switches) == 2 and status != "verified":
        raise ValueError("A return to the base coder requires a verified repair")
    if switches and status in {"idle", "requested"}:
        raise ValueError("Started model switch conflicts with repair phase")
    if not switches and status == "verified":
        raise ValueError("Verified repair has no model transition")
    pending = value["pending_switch"]
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != {"target", "reason"}:
            raise ValueError("Invalid pending routing switch")
        forward = (not switches and active_model == base and pending["target"] == escalation
                   and isinstance(pending["reason"], str) and pending["reason"] in reasons
                   and status in {"requested", "active"})
        returning = (len(switches) == 1 and active_model == escalation and pending["target"] == base
                     and pending["reason"] == "repair_verified" and status == "verified")
        if not (forward or returning):
            raise ValueError("Invalid pending routing switch")
    if status == "active" and active_model == base and pending is None:
        raise ValueError("Active repair episode has no active or pending repair model")
    if not value["initial_decided"] and (switches or pending is not None):
        raise ValueError("Routing switch precedes initial routing decision")
    if not value["initial_decided"] and status != "idle":
        raise ValueError("Repair episode precedes initial routing decision")

    calls, usage = value["model_calls"], value["model_usage"]
    if (not isinstance(calls, dict) or not set(calls) <= known_models
            or any(type(count) is not int or count < 0 for count in calls.values())
            or sum(calls.values()) > attempts):
        raise ValueError("Invalid routing model call counts")
    if not value["initial_decided"] and sum(calls.values()):
        raise ValueError("Coder calls precede initial routing decision")
    if calls.get(escalation, 0) > repair["calls_used"]:
        raise ValueError("Completed repair calls exceed consumed repair budget")
    if not isinstance(usage, dict) or not set(usage) <= known_models:
        raise ValueError("Invalid routing model usage")
    for model_name, record in usage.items():
        if (not isinstance(record, dict)
                or set(record) != {"prompt_tokens", "completion_tokens", "inference_seconds", "load_seconds",
                                   "prompt_token_calls", "completion_token_calls", "load_calls"}
                or not nonnegative(record["inference_seconds"])):
            raise ValueError("Invalid routing model usage measurements")
        for amount, coverage in (("prompt_tokens", "prompt_token_calls"),
                                 ("completion_tokens", "completion_token_calls"),
                                 ("load_seconds", "load_calls")):
            if (type(record[coverage]) is not int or not 0 <= record[coverage] <= calls.get(model_name, 0)
                    or (record[amount] is None) != (record[coverage] == 0)):
                raise ValueError("Invalid routing model measurement coverage")
            if record[amount] is not None and (
                    not nonnegative(record[amount])
                    or (amount != "load_seconds" and type(record[amount]) is not int)):
                raise ValueError("Invalid routing model usage measurements")
    progress = RoutingProgress.from_dict(deepcopy(value["progress"]))
    # Native transport permits up to eight read/search actions in one turn.
    if (progress.invalid_responses > steps or progress.repeated_actions > steps * 8
            or any(record[1] > steps for record in progress.failed_repairs)):
        raise ValueError("Routing progress exceeds completed steps")
    result = deepcopy(value)
    result.pop("progress")
    return result, progress
