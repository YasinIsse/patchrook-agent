"""Small, read-only progressive skill loader. Referenced scripts never execute."""

from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Any

from patchrook_agent.tools.base import ToolError


_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")


class SkillLoader:
    def __init__(self, trusted_root: Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.root = Path(trusted_root if trusted_root is not None else self._default_root()).resolve()
        self._index: list[dict[str, str]] = []
        self._inventory: dict[str, str] = {}
        self._documents: dict[tuple[str, str | None], dict[str, str]] = {}
        if enabled:
            self._discover()

    @staticmethod
    def _default_root() -> Path:
        module = Path(__file__).resolve()
        bundled = module.with_name("builtin")
        if bundled.is_dir():
            return bundled
        checkout = module.parents[3]
        # Only an actual source checkout may use its canonical skill folder.
        # Never discover trusted instructions from a task's working directory.
        if ((checkout / "pyproject.toml").is_file()
                and (checkout / "src/patchrook_agent/skills/loader.py").resolve() == module):
            return checkout / "skills"
        return bundled

    def _resolve(self, name: str, reference: str | None = None) -> Path:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ToolError("invalid_skill", "Use a skill name from the trusted skill index")
        directory = (self.root / name).resolve()
        if not directory.is_relative_to(self.root):
            raise ToolError("outside_skills", "Skill directory escapes the trusted root")
        if reference is not None and (not isinstance(reference, str) or not reference or "\0" in reference):
            raise ToolError("invalid_reference", "Reference must be a nonempty relative path")
        path = (directory / (reference if reference is not None else "SKILL.md")).resolve()
        if not path.is_relative_to(directory) or not path.is_relative_to(self.root):
            raise ToolError("outside_skills", "Reference escapes its trusted skill directory")
        if reference is not None and Path(reference).is_absolute():
            raise ToolError("outside_skills", "Reference must be relative to the skill directory")
        if not path.is_file():
            raise ToolError("skill_not_found", "Requested skill or reference does not exist")
        return path

    @staticmethod
    def _read(path: Path) -> str:
        with path.open("rb") as stream:
            data = stream.read(16_385)
        if len(data) > 16_384:
            raise ToolError("skill_too_large", "Skill documents must not exceed 16 KiB")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("invalid_skill", "Skill documents must be UTF-8 text") from exc
        if "\0" in text:
            raise ToolError("invalid_skill", "Binary skill references are not supported")
        return text

    @staticmethod
    def _metadata(text: str, name: str) -> tuple[dict[str, str], str]:
        if not text.startswith("---\n") or "\n---\n" not in text[4:]:
            raise ToolError("invalid_skill", f"Skill {name} needs name, description and activation metadata")
        header, body = text[4:].split("\n---\n", 1)
        metadata = {}
        for line in header.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                metadata[key.strip()] = value.strip().strip("\"'")
        if metadata.get("name") != name or any(not metadata.get(key) for key in ("description", "activation")):
            raise ToolError("invalid_skill", f"Skill {name} has invalid metadata")
        if any(len(metadata[key]) > 240 for key in ("description", "activation")):
            raise ToolError("invalid_skill", "Skill index descriptions must be concise")
        return metadata, body.strip()

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _discover(self) -> None:
        if not self.root.is_dir():
            raise ToolError("skill_not_found", "Trusted skill root must be an existing directory")
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or not (entry / "SKILL.md").exists():
                continue
            if len(self._index) >= 16:
                raise ToolError("skill_limit", "At most 16 trusted skills can be indexed")
            path = self._resolve(entry.name)
            text = self._read(path)
            metadata, _ = self._metadata(text, entry.name)
            self._index.append({"name": entry.name, "description": metadata["description"],
                                "activation": metadata["activation"],
                                "reference": f"{entry.name}/SKILL.md"})
            # Freeze trust before model actions begin. Files created or changed by
            # the task cannot become instructions through the skill-loading tool.
            for candidate in sorted(entry.rglob("*")):
                if not candidate.is_file():
                    continue
                resolved = self._resolve(entry.name, str(candidate.relative_to(entry)))
                relative = resolved.relative_to(self.root).as_posix()
                self._inventory[relative] = self._digest(self._read(resolved))
                if len(self._inventory) > 64:
                    raise ToolError("skill_limit", "At most 64 trusted documents can be indexed")

    def index(self) -> list[dict[str, str]]:
        return deepcopy(self._index)

    def index_text(self) -> str:
        if not self.enabled or not self._index:
            return ""
        entries = ["Available trusted skills; load instructions with read_skill(name):"]
        entries.extend(f"- {item['name']}: {item['description']} When: {item['activation']} [{item['reference']}]"
                       for item in self._index)
        return "\n".join(entries)

    @property
    def loaded(self) -> list[str]:
        return list(dict.fromkeys(name for name, _ in self._documents))

    def load(self, name: str, reference: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise ToolError("skills_disabled", "Skill loading is disabled for this run")
        if name not in {item["name"] for item in self._index}:
            # Resolve validates suspicious names without accepting arbitrary paths.
            self._resolve(name, reference)
            raise ToolError("skill_not_found", "Use a skill from the trusted skill index")
        if isinstance(reference, str):
            # The index exposes paths relative to the trusted root. Accept a
            # copied index path as well as a path relative to the selected skill.
            parts = Path(reference).parts
            if parts and parts[0] == name:
                reference = str(Path(*parts[1:]))
        path = self._resolve(name, reference)
        canonical_reference = path.relative_to((self.root / name).resolve()).as_posix()
        if canonical_reference == "SKILL.md":
            canonical_reference = None
        if canonical_reference is not None and name not in self.loaded:
            raise ToolError("skill_not_loaded", "Load the skill instructions before requesting references")
        relative = path.relative_to(self.root).as_posix()
        text = self._read(path)
        digest = self._digest(text)
        if self._inventory.get(relative) != digest:
            raise ToolError("skill_changed", "Trusted skill content changed or was not indexed; start a new trusted session")
        key = (name, canonical_reference)
        already_loaded = key in self._documents
        if not already_loaded:
            body = self._metadata(text, name)[1] if canonical_reference is None else text
            self._documents[key] = {"name": name, "reference": canonical_reference, "path": relative,
                                    "content": body, "sha256": digest}
        return {"success": True, "name": name, "path": relative,
                "content": "" if already_loaded else self._documents[key]["content"],
                "is_reference": canonical_reference is not None, "already_loaded": already_loaded}

    def loaded_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": f"Trusted skill document: {item['path']}\n{item['content']}"}
                for item in self._documents.values()]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "trusted_root": str(self.root), "enabled": self.enabled,
                "inventory": deepcopy(self._inventory),
                "loaded": [{"name": name, "reference": reference, "sha256": item["sha256"]}
                           for (name, reference), item in self._documents.items()]}

    def restore(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("enabled") is not self.enabled:
            raise ToolError("invalid_skill_state", "Invalid or incompatible skill checkpoint")
        if data.get("trusted_root") != str(self.root) or not isinstance(data.get("loaded"), list):
            raise ToolError("invalid_skill_state", "Skill checkpoint does not match the trusted root")
        if data.get("inventory") != self._inventory:
            raise ToolError("skill_changed", "Trusted skills changed since the checkpoint")
        previous = deepcopy(self._documents)
        try:
            self._documents.clear()
            for item in data["loaded"]:
                if not isinstance(item, dict) or set(item) != {"name", "reference", "sha256"}:
                    raise ToolError("invalid_skill_state", "Invalid loaded skill checkpoint entry")
                self.load(item["name"], item["reference"])
                if self._documents[(item["name"], item["reference"])]["sha256"] != item["sha256"]:
                    raise ToolError("skill_changed", "Loaded skill changed since the checkpoint")
        except (ToolError, TypeError, KeyError) as exc:
            self._documents = previous
            if isinstance(exc, ToolError):
                raise
            raise ToolError("invalid_skill_state", "Invalid loaded skill checkpoint entry") from exc
