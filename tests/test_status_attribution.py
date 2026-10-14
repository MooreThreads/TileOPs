"""Tests for operator/case attribution in ``scripts/status/attribution.py``.

Ownership is declared, and inference is only a bootstrap, so these pin the
declaration precedence and the fallback rules that decide which operator a case
gates. They run without a GPU.
"""

import pytest

from scripts.status import attribution

pytestmark = pytest.mark.smoke

_NAMES = {"SoftmaxFwdOp", "LogSoftmaxFwdOp", "AddFwdOp", "MulFwdOp"}


class _Marker:
    def __init__(self, *args):
        self.args = args


class _Callspec:
    def __init__(self, params):
        self.params = params


class _Item:
    """A stand-in for a collected pytest item."""

    def __init__(
        self, *, name="test_x", params=None, markers=None, nodeid="tests/ops/test_x.py::test_x"
    ):
        self.name = name
        self.originalname = name.split("[", 1)[0]
        self.nodeid = nodeid
        self.callspec = _Callspec(params) if params is not None else None
        self._markers = markers or {}

    def get_closest_marker(self, name):
        return self._markers.get(name)


def _infer(item, *, relpath="tests/ops/test_x.py", body=None, scoped=None):
    return attribution.infer(
        item=item,
        relpath=relpath,
        manifest_names=_NAMES,
        body_ops_by_file=body or {},
        scoped_by_file=scoped or {},
    )


def test_mentions_name_requires_a_whole_identifier():
    assert attribution.mentions_name("SoftmaxFwdOp(dim=-1)", "SoftmaxFwdOp")
    assert not attribution.mentions_name("LogSoftmaxFwdOp(dim=-1)", "SoftmaxFwdOp")


def test_manifest_names_in_walks_nested_values():
    value = {"op": "AddFwdOp", "extra": ["MulFwdOp", {"k": "nope"}]}
    assert attribution.manifest_names_in(value, _NAMES) == {"AddFwdOp", "MulFwdOp"}


def test_declared_marker_wins_over_the_heuristic_signals():
    item = _Item(params={"name": "AddFwdOp"}, markers={"op": _Marker("SoftmaxFwdOp")})
    result = _infer(item)
    assert result.ops == {"SoftmaxFwdOp"}
    assert result.source == "declared"


def test_infra_marker_owns_nothing():
    item = _Item(params={"name": "AddFwdOp"}, markers={"infra": _Marker()})
    result = _infer(item)
    assert result.ops == frozenset()
    assert result.source == "infra"


def test_param_naming_one_operator_is_accepted_outside_its_source_file():
    """A file that declares no operator still attributes a param-named one."""
    item = _Item(params={"name": "AddFwdOp"})
    result = _infer(item, relpath="tests/ops/test_conformance.py")
    assert result.ops == {"AddFwdOp"}
    assert result.source == "param"


def test_param_naming_several_operators_is_narrowed_by_the_file():
    item = _Item(params={"a": "AddFwdOp", "b": "MulFwdOp"})
    scoped = {"tests/ops/test_x.py": {"AddFwdOp"}}
    result = _infer(item, scoped=scoped)
    assert result.ops == {"AddFwdOp"}
    assert result.source == "param"


def test_param_naming_several_operators_outside_the_file_keeps_all():
    item = _Item(params={"a": "AddFwdOp", "b": "MulFwdOp"})
    result = _infer(item, relpath="tests/ops/test_conformance.py")
    assert result.ops == {"AddFwdOp", "MulFwdOp"}


def test_body_naming_one_operator_is_accepted_outside_its_source_file():
    item = _Item()
    relpath = "tests/ops/test_conformance.py"
    body = {relpath: {"test_x": {"MulFwdOp"}}}
    result = _infer(item, relpath=relpath, body=body)
    assert result.ops == {"MulFwdOp"}
    assert result.source == "body"


def test_function_name_alias_resolves_when_nothing_else_does():
    item = _Item(name="test_softmax_1d")
    result = _infer(item)
    assert result.ops == {"SoftmaxFwdOp"}
    assert result.source == "alias"


def test_nothing_resolves_to_no_owner():
    item = _Item(name="test_unrelated_helper")
    result = _infer(item)
    assert result.ops == frozenset()
    assert result.source == "none"


def test_function_body_ops_ignores_parametrize_decorators(tmp_path):
    ops_dir = tmp_path / "tests" / "ops"
    ops_dir.mkdir(parents=True)
    (ops_dir / "test_x.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('op_cls', [SoftmaxFwdOp, LogSoftmaxFwdOp])\n"
        "def test_rounding(op_cls):\n"
        "    assert op_cls is not None\n",
        encoding="utf-8",
    )
    body = attribution.function_body_ops(tmp_path, _NAMES)
    assert body["tests/ops/test_x.py"]["test_rounding"] == set()


def test_function_body_ops_reads_the_body(tmp_path):
    ops_dir = tmp_path / "tests" / "ops"
    ops_dir.mkdir(parents=True)
    (ops_dir / "test_x.py").write_text(
        "def test_op():\n    SoftmaxFwdOp(dim=-1)\n",
        encoding="utf-8",
    )
    body = attribution.function_body_ops(tmp_path, _NAMES)
    assert body["tests/ops/test_x.py"]["test_op"] == {"SoftmaxFwdOp"}


def test_declaration_issues_flag_unknown_and_out_of_file_operators():
    item = _Item(
        markers={"op": _Marker("NopeFwdOp", "AddFwdOp")}, nodeid="tests/ops/test_x.py::test_x"
    )
    issues = attribution.declaration_issues(item, "tests/ops/test_x.py", _NAMES, {"MulFwdOp"})
    assert any("not in the manifest" in issue for issue in issues)
    assert any("source.test is not" in issue for issue in issues)


def test_coverage_counts_each_signal():
    cases = [
        {"ops": ["A"], "attribution": "declared"},
        {"ops": ["A"], "attribution": "param"},
        {"ops": [], "attribution": "infra"},
        {"ops": [], "attribution": "bogus"},
    ]
    counts = attribution.coverage(cases)
    assert counts["declared"] == 1
    assert counts["param"] == 1
    assert counts["infra"] == 1
    assert counts["none"] == 1
