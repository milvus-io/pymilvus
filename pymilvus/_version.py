from __future__ import annotations

import re

from setuptools_scm import get_version as get_scm_version
from setuptools_scm.version import (
    SEMVER_MINOR,
    ScmVersion,
    get_no_local_node,
    guess_next_simple_semver,
    guess_next_version,
)

_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$")
_VERSION_FORMAT = "{guessed}rc{distance}"


def _parse_branch_version(branch: str) -> str | None:
    name = branch.rsplit("/", 1)[-1]
    match = _VERSION_RE.match(name)
    if match is None:
        return None
    return match.group(1)


def rc_version_scheme(version: ScmVersion) -> str:
    if version.exact:
        return version.format_with("{tag}")

    if version.branch is not None:
        branch_version = _parse_branch_version(version.branch)
        if branch_version is not None:
            tag_minor = str(version.tag).split(".")[:SEMVER_MINOR]
            branch_minor = branch_version.split(".")[:SEMVER_MINOR]
            if branch_minor == tag_minor:
                return version.format_next_version(guess_next_version, fmt=_VERSION_FORMAT)

    return version.format_next_version(
        guess_next_simple_semver,
        retain=SEMVER_MINOR,
        fmt=_VERSION_FORMAT,
    )


def get_version() -> str:
    return get_scm_version(
        root="..",
        relative_to=__file__,
        version_scheme=rc_version_scheme,
        local_scheme=get_no_local_node,
    )
