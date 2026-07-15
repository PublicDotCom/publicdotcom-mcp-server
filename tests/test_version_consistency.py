"""The release workflow is gated on pyproject's version, but server.json carries
its own copies that feed the MCP registry. Nothing stops one from moving without
the others, which would publish a release the registry advertises under the wrong
version. These run in CI so the drift is caught at PR time rather than at publish.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Parsed by hand rather than with tomllib, which is 3.11+ while this package
# supports 3.10 (see requires-python). The [project] version is a plain literal,
# so a scoped regex is enough and avoids taking a tomli dependency just for this.
_PROJECT_SECTION = re.compile(r"^\[project\]$(.*?)^\[", re.MULTILINE | re.DOTALL)
_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
_NAME = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)


def _project_section() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    match = _PROJECT_SECTION.search(text)
    assert match, "could not locate the [project] section in pyproject.toml"
    return match.group(1)


def _pyproject_version() -> str:
    match = _VERSION.search(_project_section())
    assert match, "could not locate version in pyproject.toml's [project] section"
    return match.group(1)


def _pyproject_name() -> str:
    match = _NAME.search(_project_section())
    assert match, "could not locate name in pyproject.toml's [project] section"
    return match.group(1)


def _server_json() -> dict:
    return json.loads((REPO_ROOT / "server.json").read_text())


def test_pyproject_version_is_parseable():
    # Guards the regex itself: if the [project] layout changes, fail loudly here
    # rather than silently comparing None to None in the tests below.
    assert re.fullmatch(r"\d+\.\d+\.\d+", _pyproject_version())


def test_server_json_top_level_version_matches_pyproject():
    assert _server_json()["version"] == _pyproject_version()


def test_server_json_package_version_matches_pyproject():
    packages = _server_json()["packages"]
    assert len(packages) == 1, "expected a single package entry; update this test if that changes"
    assert packages[0]["version"] == _pyproject_version()


def test_server_json_package_identifier_matches_pyproject_name():
    assert _server_json()["packages"][0]["identifier"] == _pyproject_name()
