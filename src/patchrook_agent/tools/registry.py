"""The single validation and execution boundary for model-proposed actions."""

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .base import (
    TOOL_CAPTURE, ToolError, Workspace, bound_read_files_result, bound_read_result,
    observation_chars, record_tool_output, with_read_guidance,
)
from .files import edit_file, list_files, read_file, read_files, write_file
from .code import read_code
from .search import search_code
from .tests import run_tests


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PathArguments(Arguments):
    path: str = Field(min_length=1, max_length=4096)


class ListArguments(PathArguments):
    path: str = Field(default=".", min_length=1, max_length=4096)
    recursive: bool = False
    max_entries: int | None = Field(default=None, gt=0)
    offset: int = Field(default=0, ge=0)


class ReadArguments(PathArguments):
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class ReadFilesArguments(Arguments):
    files: list[ReadArguments] = Field(min_length=1, max_length=10)


class ReadCodeArguments(Arguments):
    paths: list[Annotated[str, Field(min_length=1, max_length=4096)]] = Field(min_length=1, max_length=10)


def _code_arguments(arguments: Any) -> tuple[Any, str | None]:
    """Accept the former read-only selectors without executing their hints."""
    if not isinstance(arguments, dict):
        return arguments, None
    field = "paths" if "paths" in arguments else "files" if "files" in arguments else None
    if field is None or arguments.keys() - {field, "symbol", "start_line", "end_line"}:
        return arguments, None
    if not isinstance(arguments[field], list):
        return {"paths": arguments[field]}, None
    paths = []
    changed = field == "files" or len(arguments) > 1
    for selection in arguments[field]:
        if isinstance(selection, str):
            paths.append(selection)
        elif (isinstance(selection, dict) and "path" in selection
              and not selection.keys() - {"path", "symbol", "start_line", "end_line"}):
            paths.append(selection["path"])
            changed = True
        else:
            raise ToolError("invalid_arguments", 'Use read_code with {"paths":["relative/file.py"]}.')
    return {"paths": paths}, "used paths; ignored selector hints" if changed else None


class SearchArguments(Arguments):
    pattern: str = Field(min_length=1, max_length=4096)
    path: str = Field(default=".", min_length=1, max_length=4096)
    max_results: int | None = Field(default=None, gt=0)
    offset: int = Field(default=0, ge=0)
    mode: Literal["literal", "regex"] = Field(
        default="literal", description="Plain-text literal search by default. Set regex only for an intentional regular expression.")


class WriteArguments(PathArguments):
    content: str = Field(max_length=1_048_576)
    overwrite: bool = False


class EditArguments(PathArguments):
    old_text: str = Field(min_length=1, max_length=1_048_576)
    new_text: str = Field(max_length=1_048_576)


class TestArguments(PathArguments):
    path: str = Field(default=".", min_length=1, max_length=4096)
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class SkillArguments(Arguments):
    name: str = Field(min_length=1, max_length=128)
    reference: str | None = Field(default=None, min_length=1, max_length=4096)


class ArtifactArguments(Arguments):
    artifact_id: str = Field(min_length=1, max_length=64)
    offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=1000, gt=0, le=2000)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[Arguments]
    execute: Callable[..., dict[str, Any]]


def _schema(model: type[Arguments]) -> dict[str, Any]:
    schema = model.model_json_schema()

    def compact(value):
        # Keep the model-facing shape small. The original strict Pydantic model
        # remains authoritative for bounds, defaults and argument validation.
        if not isinstance(value, dict):
            return value
        if value.get("$ref", "").startswith("#/$defs/"):
            # Inline the small read selection shape for Ollama tool adapters
            # that do not resolve local JSON-schema references.
            return compact(schema["$defs"][value["$ref"].removeprefix("#/$defs/")])
        if "anyOf" in value:
            choices = [choice for choice in value["anyOf"] if choice.get("type") != "null"]
            if len(choices) == 1:
                return compact(choices[0])
        result = {}
        for key in ("type", "required", "additionalProperties", "enum", "$ref", "description"):
            if key in value:
                result[key] = value[key]
        for key in ("properties",):
            if key in value:
                result[key] = {name: compact(item) for name, item in value[key].items()}
        if "items" in value:
            result["items"] = compact(value["items"])
        if "anyOf" in value:
            result["anyOf"] = [compact(item) for item in value["anyOf"]]
        return result

    return compact(schema)


def bounded_result(result: dict[str, Any], limit: int, *,
                   enabled_tools: set[str] | None = None) -> dict[str, Any]:
    """Bound observations without changing paths, outcomes, or continuation IDs.

    Only designated evidence fields are shortened. Numbered reads retain whole
    lines, and paginated collections advance past the entries actually shown.
    Raw results and process captures remain available to session artifacts.
    """
    result = deepcopy(result)
    # Fit evidence first, then derive optional guidance from the visible records.
    # Raw search output can contain more files than the final observation.
    result.pop("read_hint", None)
    result.pop("suggested_read", None)
    if observation_chars(result) <= limit:
        return with_read_guidance(result, limit, enabled_tools=enabled_tools)
    result["truncated"] = True
    if "files" in result and "requested_count" in result:
        result = bound_read_files_result(result, limit)
    if "content" in result and "start_line" in result and "end_line" in result:
        result = bound_read_result(result, limit)
    for key in ("entries", "matches"):
        if not isinstance(result.get(key), list):
            continue
        # A short continuation is enough after final boundary fitting; the tool
        # definition documents the argument, and match records retain locations.
        if observation_chars(result) > limit and "follow_up" in result:
            result["follow_up"] = "Use next_offset as offset"
        while result[key] and observation_chars(result) > limit:
            result[key].pop()
            result["returned_count"] = len(result[key])
            result["next_offset"] = result.get("offset", 0) + len(result[key])
    # Strings such as paths, tool identities, artifact IDs, and error codes must
    # never be shortened into plausible but incorrect values.
    while observation_chars(result) > limit:
        candidates = [(len(result[key]), result, key) for key in ("stdout", "stderr", "text", "content")
                      if isinstance(result.get(key), str) and result[key]
                      and not (key == "content" and "start_line" in result)]
        error = result.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
            candidates.append((len(error["message"]), error, "message"))
        if not candidates:
            # Oversized identities cannot be represented faithfully in a small
            # observation. Omit them explicitly rather than inventing a path.
            essential = {key: result[key] for key in ("success", "status", "exit_code", "artifact_id")
                         if key in result}
            essential.update(truncated=True, metadata_omitted=True)
            essential["error"] = {"code": error.get("code", "output_limit") if isinstance(error, dict) else "output_limit",
                                  "message": "Result metadata exceeds observation budget; inspect the artifact."}
            return essential
        _, container, key = max(candidates, key=lambda item: item[0])
        marker = "message_truncated" if key == "message" else f"{key}_truncated"
        container[marker] = True
        excess = observation_chars(result) - limit
        container[key] = container[key][:max(0, len(container[key]) - max(excess, 1))]
    return with_read_guidance(result, limit, enabled_tools=enabled_tools)


class ToolRegistry:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.capture_enabled = False
        self.observation_overhead_chars = 0
        self.guidance_tools: set[str] | None = None
        self.lean_mode = False
        self.code_context = True
        self._skill_names: tuple[str, ...] = ()
        self.last_execution: dict[str, Any] | None = None
        self.tools = {tool.name: tool for tool in (
            Tool("list_files", "List paths; search_code locates relevant lines, read_files batches known ranges. next_offset pages results.", ListArguments, list_files),
            Tool("read_file", "Read a narrow numbered UTF-8 range in one file; use read_files for multiple known files.", ReadArguments, read_file),
            Tool("read_files", "Read 1-10 files in one call; each selection has path and optional start_line/end_line. Use narrow ranges.", ReadFilesArguments, read_files),
            Tool("search_code", "Search file contents as literal text by default; mode=regex enables regular expressions. next_offset pages results.", SearchArguments, search_code),
            Tool("write_file", "Create a UTF-8 file and parent directories. Overwrites require overwrite=true.", WriteArguments, write_file),
            Tool("edit_file", "Replace exactly one literal occurrence in an existing UTF-8 file.", EditArguments, edit_file),
            Tool("run_tests", "run_tests() runs pytest at workspace root; optional path selects a file, directory, or node.", TestArguments, run_tests),
        )}

    def enable_lean_tools(self) -> None:
        self.lean_mode = True
        self.tools["read_code"] = Tool(
            "read_code", "Read useful code from 1-10 known paths. Use read_file for exact line ranges.",
            ReadCodeArguments, read_code,
        )

    def attach_skills(self, loader) -> None:
        self.tools.pop("read_skill", None)
        self._skill_names = ()
        if loader.enabled:
            index = getattr(loader, "index", None)
            self._skill_names = tuple(item["name"] for item in index() if isinstance(item, dict)
                                      and isinstance(item.get("name"), str)) if callable(index) else ()
            self.tools["read_skill"] = Tool(
                "read_skill", "Load instructions by name. Reference is optional for supporting documents; scripts are read as text.",
                SkillArguments, lambda workspace, **kwargs: loader.load(**kwargs),
            )

    def attach_artifacts(self, artifacts) -> None:
        self.tools.pop("read_artifact", None)
        self.observation_overhead_chars = 160 if artifacts is not None else 0
        if artifacts is not None:
            self.tools["read_artifact"] = Tool(
                "read_artifact", "Expand missing task evidence by ID/offset. Memory history is optional; finish when requested work and verification are complete.",
                ArtifactArguments, lambda workspace, **kwargs: artifacts.read(**kwargs),
            )

    def definitions(self, *, enabled_tools: set[str] | None = None,
                    lean: bool | None = None) -> list[dict[str, Any]]:
        if lean is not None:
            self.lean_mode = lean
        if self.lean_mode:
            self.enable_lean_tools()
        else:
            self.tools.pop("read_code", None)
        available = set(self.tools) if enabled_tools is None else set(self.tools) & enabled_tools
        lean_descriptions = {
            "list_files": "List repository paths; read known files with read_code.",
            "read_code": "Read useful code from paths: a list of 1-10 repository file paths.",
            "read_file": "Read an exact numbered line range from one file.",
            "search_code": "Search CONTENT as literal text; mode=regex enables regex. list_files discovers filenames.",
            "edit_file": "Replace one exact source occurrence.",
            "write_file": "Create a UTF-8 file; overwrites require overwrite=true.",
            "run_tests": "run_tests() runs pytest at workspace root; optional path selects a file, directory, or node.",
            "read_skill": "Load a trusted skill by name when requested; optional reference.",
            "read_artifact": "Read omitted session evidence by artifact_id and character offset.",
        }
        if self.lean_mode:
            available &= set(lean_descriptions)
        else:
            available.discard("read_code")
        definitions = []
        for tool in self.tools.values():
            if tool.name not in available:
                continue
            description = tool.description
            if self.lean_mode:
                description = lean_descriptions[tool.name]
                if "read_code" not in available and tool.name in {"search_code", "list_files"}:
                    description = ("Search file CONTENT as literal text by default; mode=regex enables regex." if tool.name == "search_code"
                                   else "List repository paths.")
            elif available != set(self.tools):
                reader = "read_files" if "read_files" in available else "read_file" if "read_file" in available else None
                if tool.name == "read_file" and "read_files" not in available:
                    description = "Read a narrow numbered UTF-8 range in one file."
                elif tool.name == "search_code":
                    description = "Search contents as literal text by default; mode=regex enables regex. next_offset pages results."
                    if reader:
                        description += f" Use {reader} for narrow known ranges."
                elif tool.name == "list_files":
                    description = "List bounded repository paths; next_offset pages results."
                    if "search_code" in available:
                        description += " search_code locates relevant lines."
                    if reader:
                        description += f" Use {reader} for narrow known ranges."
            parameters = _schema(tool.arguments)
            if tool.name == "read_artifact":
                # Expose the authoritative Pydantic bounds for this paginated
                # tool rather than asking the model to infer its output cap.
                original = tool.arguments.model_json_schema()["properties"]
                for field in ("offset", "max_chars"):
                    parameters["properties"][field].update({
                        key: value for key, value in original[field].items()
                        if key in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}})
            definitions.append({"type": "function", "function": {
                "name": tool.name, "description": description, "parameters": parameters,
            }})
        return definitions

    def error(self, code: str, message: str) -> dict[str, Any]:
        return bounded_result({"success": False, "error": {"code": code, "message": message}},
                              self.workspace.config.max_output_chars)

    def _code_error(self, code: str, message: str, arguments: Any) -> dict[str, Any]:
        hint = ' Use read_code with {"paths":["relative/file.py"]}.'
        if self.guidance_tools is None or "read_file" in self.guidance_tools:
            hint += " read_file accepts exact ranges."
        result = self.error(code, message + hint)
        raw = arguments.get("paths", arguments.get("files", [])) if isinstance(arguments, dict) else []
        paths = []
        for item in ([raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []):
            path = item.get("path") if isinstance(item, dict) else item
            try:
                resolved = self.workspace.resolve(path)
                canonical = self.workspace.relative(resolved)
                if resolved.is_file() and canonical not in paths:
                    paths.append(canonical)
            except (ToolError, OSError, TypeError):
                continue
            if len(paths) == 10:
                break
        while paths:
            candidate = {**result, "suggested_call": {"name": "read_code", "arguments": {"paths": paths}}}
            if observation_chars(candidate) <= self.workspace.config.max_output_chars:
                return candidate
            paths = paths[:-1]
        return result

    def _action_error(self, name: str, code: str, message: str, arguments: Any) -> dict[str, Any]:
        """Attach bounded next-action data without retrying or relaxing checks."""
        if name == "read_code":
            return self._code_error(code, message, arguments)
        if name == "search_code" and code == "invalid_regex" and isinstance(arguments, dict):
            result = self.error(code, message)
            retry = {key: value for key, value in arguments.items()
                     if key in {"pattern", "path", "max_results", "offset"}}
            retry["mode"] = "literal"
            candidate = {**result, "suggested_call": {"name": "search_code", "arguments": retry}}
            limit = max(256, self.workspace.config.max_output_chars - self.observation_overhead_chars)
            # Do not shorten the pattern to manufacture a different retry.
            if observation_chars(candidate) <= limit:
                return candidate
            return result
        if name not in {"run_tests", "read_skill"}:
            return self.error(code, message)
        result = self.error(code, message)
        available = set(self.tools) if self.guidance_tools is None else self.guidance_tools
        if name == "run_tests" and code == "not_found":
            if "list_files" in available:
                result["suggested_call"] = {"name": "list_files", "arguments": {
                    "path": ".", "recursive": True, "max_entries": 30}}
            elif "run_tests" in available and self.workspace.config.allow_tests:
                result["suggested_call"] = {"name": "run_tests", "arguments": {"path": "."}}
        elif (name == "read_skill" and code in {"invalid_arguments", "invalid_skill", "skill_not_found"}
              and "read_skill" in available and isinstance(arguments, dict)
              and arguments.get("name") not in self._skill_names):
            names = list(self._skill_names)
            record_tool_output(valid_skill_names=names)
            result["valid_names"] = names
            limit = max(256, self.workspace.config.max_output_chars - self.observation_overhead_chars)
            while len(result["valid_names"]) > 1 and observation_chars(result) > limit:
                result["valid_names"] = result["valid_names"][:-1]
                result["additional_valid_names"] = len(names) - len(result["valid_names"])
                result["truncated"] = True
            if not names:
                result["error"]["message"] += " No trusted skills are available in this session."
        limit = max(256, self.workspace.config.max_output_chars - self.observation_overhead_chars)
        return bounded_result(result, limit, enabled_tools=self.guidance_tools)

    def dispatch(self, name: str, arguments: Any, *,
                 remaining_seconds: float | None = None) -> dict[str, Any]:
        captured = {"max_bytes": self.workspace.config.max_file_bytes} if self.capture_enabled else None
        token = TOOL_CAPTURE.set(captured)
        try:
            result = self._dispatch(name, arguments, remaining_seconds=remaining_seconds)
            if self.guidance_tools is not None and self.guidance_tools != set(self.tools):
                # Restricted runs must not preserve generated advice for a
                # disabled tool in expandable raw-result artifacts either.
                result = with_read_guidance(result, self.workspace.config.max_output_chars,
                                            enabled_tools=self.guidance_tools)
            self.last_execution = {"name": name, "arguments": deepcopy(arguments),
                                   "result": deepcopy(result),
                                   "available_output": {key: value for key, value in (captured or {}).items()
                                                        if key != "max_bytes"}}
            limit = max(256, self.workspace.config.max_output_chars - self.observation_overhead_chars)
            return bounded_result(result, limit, enabled_tools=self.guidance_tools)
        finally:
            TOOL_CAPTURE.reset(token)

    def _dispatch(self, name: str, arguments: Any, *,
                  remaining_seconds: float | None = None) -> dict[str, Any]:
        if not isinstance(name, str) or name not in self.tools:
            return self.error("unknown_tool", "Unknown tool; use a registered tool name")
        tool = self.tools[name]
        try:
            if remaining_seconds is not None and remaining_seconds <= 0:
                raise ToolError("deadline_exceeded", "Task deadline exceeded before execution")
            if isinstance(arguments, str):
                if len(arguments) > 2_100_000:
                    raise ToolError("invalid_arguments", "Tool arguments exceed the input limit")
                arguments = json.loads(arguments)
            normalization = None
            if name == "read_code":
                arguments, normalization = _code_arguments(arguments)
            validated = tool.arguments.model_validate(arguments)
            kwargs = validated.model_dump(exclude_none=True)
            if name == "read_code":
                kwargs["files"] = [{"path": path} for path in kwargs.pop("paths")]
                kwargs["include_context"] = self.code_context
                overhead = (observation_chars({"normalization": normalization}) - 1) if normalization else 0
                kwargs["observation_limit"] = max(256, self.workspace.config.max_output_chars - self.observation_overhead_chars - overhead)
            if name in {"search_code", "run_tests"} and remaining_seconds is not None:
                kwargs["timeout_seconds"] = min(kwargs.get("timeout_seconds", remaining_seconds),
                                                remaining_seconds)
            result = tool.execute(self.workspace, **kwargs)
            if normalization:
                result = {**result, "normalization": normalization}
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
                               for e in exc.errors(include_input=False, include_url=False))
            return self._action_error(name, "invalid_arguments", detail, arguments)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return self._action_error(name, "invalid_arguments", str(exc), arguments)
        except ToolError as exc:
            return self._action_error(name, exc.code, exc.message, arguments)
        except OSError as exc:
            return self.error("io_error", str(exc))
        except Exception:
            return self.error("execution_error", "Tool execution failed unexpectedly")
        return result
