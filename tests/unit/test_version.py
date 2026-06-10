from pymilvus import _version


class FakeScmVersion:
    def __init__(self, exact=False, branch=None, tag="3.0.1"):
        self.exact = exact
        self.branch = branch
        self.tag = tag
        self.calls = []

    def format_with(self, fmt):
        self.calls.append(("format_with", fmt))
        return str(self.tag)

    def format_next_version(self, guess, **kwargs):
        self.calls.append(("format_next_version", guess, kwargs))
        return f"{kwargs['fmt'].format(guessed='3.1.0', distance=16)}"


def test_parse_branch_version():
    assert _version._parse_branch_version("origin/2.6") == "2.6"
    assert _version._parse_branch_version("feature/refactor") is None


def test_rc_version_scheme_exact_tag():
    version = FakeScmVersion(exact=True, tag="3.0.0")

    assert _version.rc_version_scheme(version) == "3.0.0"
    assert version.calls == [("format_with", "{tag}")]


def test_rc_version_scheme_release_branch():
    version = FakeScmVersion(branch="origin/3.0", tag="3.0.1")

    assert _version.rc_version_scheme(version) == "3.1.0rc16"
    _, guess, kwargs = version.calls[0]
    assert guess is _version.guess_next_version
    assert kwargs == {"fmt": "{guessed}rc{distance}"}


def test_rc_version_scheme_development_branch():
    version = FakeScmVersion(branch="feature/refactor", tag="3.0.1")

    assert _version.rc_version_scheme(version) == "3.1.0rc16"
    _, guess, kwargs = version.calls[0]
    assert guess is _version.guess_next_simple_semver
    assert kwargs == {"retain": _version.SEMVER_MINOR, "fmt": "{guessed}rc{distance}"}


def test_get_version_uses_rc_scheme(monkeypatch):
    def fake_get_scm_version(**kwargs):
        assert kwargs["root"] == ".."
        assert kwargs["relative_to"] == _version.__file__
        assert kwargs["version_scheme"] is _version.rc_version_scheme
        assert kwargs["local_scheme"] is _version.get_no_local_node
        return "3.1.0rc16"

    monkeypatch.setattr(_version, "get_scm_version", fake_get_scm_version)

    assert _version.get_version() == "3.1.0rc16"
