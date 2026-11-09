"""Bundle the canonical trusted skill documents without duplicating their source."""

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithTrustedSkills(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent / "skills"
        if (not source.is_dir() or source.is_symlink()
                or any(path.is_symlink() for path in source.rglob("*"))):
            raise ValueError("Trusted skills must be a real directory without symlinks")
        destination = Path(self.build_lib) / "patchrook_agent" / "skills" / "builtin"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


setup(cmdclass={"build_py": BuildWithTrustedSkills})
