"""
this module is a hack only in place to allow for setuptools
to use the attribute for the versions

it works only if the backend-path of the build-system section
from pyproject.toml is respected
"""
from __future__ import annotations

import logging
from typing import Callable

from setuptools import build_meta as build_meta  # noqa

from setuptools_scm import _types as _t
from setuptools_scm import Configuration
from setuptools_scm import get_version
from setuptools_scm import git
from setuptools_scm import hg
from setuptools_scm.fallbacks import parse_pkginfo
from setuptools_scm.version import (
    get_no_local_node,
    _parse_version_tag,
    guess_next_simple_semver,
    SEMVER_MINOR,
    guess_next_version,
    ScmVersion,
)

log = logging.getLogger("setuptools_scm")
# todo: take fake entrypoints from pyproject.toml
try_parse: list[Callable[[_t.PathT, Configuration], ScmVersion | None]] = [
    parse_pkginfo,
    git.parse,
    hg.parse,
    git.parse_archival,
    hg.parse_archival,
]


def parse(root: str, config: Configuration) -> ScmVersion | None:
    for maybe_parse in try_parse:
        try:
            parsed = maybe_parse(root, config)
        except OSError as e:
            log.warning("parse with %s failed with: %s", maybe_parse, e)
        else:
            if parsed is not None:
                return parsed

fmt = "{guessed}rc{distance}" # align with PEP440 public version that has no dot before rc

def custom_version(version: ScmVersion) -> str:
    if version.exact:
        return version.format_with("{tag}")
    if version.branch is not None:
        # Does the branch name (stripped of namespace) parse as a version?
        branch_ver_data = _parse_version_tag(
            version.branch.split("/")[-1], version.config
        )
        if branch_ver_data is not None:
            branch_ver = branch_ver_data["version"]
            if branch_ver[0] == "v":
                # Allow branches that start with 'v', similar to Version.
                branch_ver = branch_ver[1:]
            # Does the branch version up to the minor part match the tag? If not it
            # might be like, an issue number or something and not a version number, so
            # we only want to use it if it matches.
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
        parse=parse,
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
