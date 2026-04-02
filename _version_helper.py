"""
Custom version scheme for pymilvus using setuptools_scm public API only.

Loaded via pyproject.toml:
    [tool.setuptools.dynamic]
    version = { attr = "_version_helper.version"}

Uses only public setuptools_scm API so it stays compatible with
setuptools_scm 10.x+ where private modules (_types, fallbacks) are removed.
"""
from __future__ import annotations

import re

from setuptools import build_meta as build_meta  # noqa
from setuptools_scm import get_version
from setuptools_scm.version import (
    SEMVER_MINOR,
    ScmVersion,
    get_no_local_node,
    guess_next_simple_semver,
    guess_next_version,
)

_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$")

fmt = "{guessed}rc{distance}"  # align with PEP440 public version that has no dot before rc


def _parse_branch_version(branch: str) -> str | None:
    """Extract a version string from a branch name like '2.6' or 'v2.6.1'.

    Returns the version without leading 'v', or None if the branch name
    doesn't look like a version number.
    """
    name = branch.split("/")[-1]
    m = _VERSION_RE.match(name)
    if m is None:
        return None
    return m.group(1)


def custom_version(version: ScmVersion) -> str:
    if version.exact:
        return version.format_with("{tag}")
    if version.branch is not None:
        branch_ver = _parse_branch_version(version.branch)
        if branch_ver is not None:
            tag_ver_up_to_minor = str(version.tag).split(".")[:SEMVER_MINOR]
            branch_ver_up_to_minor = branch_ver.split(".")[:SEMVER_MINOR]
            if branch_ver_up_to_minor == tag_ver_up_to_minor:
                # We're in a release/maintenance branch, next is a patch/rc/beta bump:
                return version.format_next_version(guess_next_version, fmt=fmt)
    # We're in a development branch, next is a minor bump:
    return version.format_next_version(guess_next_simple_semver, retain=SEMVER_MINOR, fmt=fmt)


def scm_version() -> str:
    return get_version(
        relative_to=__file__,
        version_scheme=custom_version,
        local_scheme=get_no_local_node,
    )


version: str


def __getattr__(name: str) -> str:
    if name == "version":
        global version
        version = scm_version()
        return version
    raise AttributeError(name)
