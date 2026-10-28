"""Planted violations for the profile static scan: one per rule, one per shape."""

from __future__ import annotations

import textwrap

import pytest

from spectramr.cli.profile_static import RULES, scan_file, scan_source


def _rules(src: str, **kw) -> list[tuple[str, str]]:
    return [(f.rule, f.function) for f in scan_source(textwrap.dedent(src), **kw)]


@pytest.mark.parametrize(
    "body",
    [
        "for x in xs:\n    total += x.item()",
        "while go:\n    y = t.cpu()",
        "for x in xs:\n    y = x.numpy()",
        "for x in xs:\n    y = x.tolist()",
        "ys = [x.item() for x in xs]",
        "ys = {x.item() for x in xs}",
        "ys = (x.item() for x in xs)",
    ],
)
def test_host_sync_in_every_loop_shape(body):
    src = "def f(xs):\n" + textwrap.indent(body, "    ")
    assert ("host_sync_in_loop", "f") in _rules(src)


def test_host_sync_outside_a_loop_is_not_flagged():
    assert _rules("def f(t):\n    return t.item()") == []


def test_a_nested_def_inside_a_loop_does_not_inherit_the_loop():
    src = """
    def f(xs):
        for x in xs:
            def g(t):
                return t.item()
    """
    assert _rules(src) == []


def test_empty_cache_in_loop_only():
    src = """
    def f(n):
        torch.cuda.empty_cache()
        for _ in range(n):
            torch.cuda.empty_cache()
    """
    assert _rules(src) == [("empty_cache_in_loop", "f")]


@pytest.mark.parametrize(
    "call", ["np.fft.fft2(x)", "numpy.fft.ifft(x)", "scipy.fft.fftn(x)", "torch.fft.fft2(x)"]
)
def test_raw_fft_every_namespace(call):
    assert ("raw_fft", "f") in _rules(f"def f(x):\n    return {call}")


def test_fft_owner_module_is_exempt():
    assert _rules("def f(x):\n    return torch.fft.fft2(x)", fft_exempt=True) == []


@pytest.mark.parametrize(
    "iter_expr",
    ["range(x.shape[0])", "range(len(coils))", "range(x.size(0))", "range(b, x.shape[1])"],
)
def test_loop_over_extent_every_shape(iter_expr):
    src = f"def f(x, coils, b):\n    for i in {iter_expr}:\n        pass"
    assert ("loop_over_extent", "f") in _rules(src)


def test_constant_range_is_not_a_loop_over_extent():
    assert _rules("def f():\n    for i in range(3):\n        pass") == []


def test_qualnames_match_pythons_spelling():
    src = """
    class C:
        def m(self, xs):
            def inner(t):
                for _ in t:
                    t.item()
            for x in xs:
                x.item()
    """
    assert sorted(_rules(src)) == [
        ("host_sync_in_loop", "C.m"),
        ("host_sync_in_loop", "C.m.<locals>.inner"),
    ]


def test_every_emitted_rule_is_documented():
    src = """
    def f(x):
        for i in range(len(x)):
            x.item(); torch.cuda.empty_cache(); np.fft.fft(x)
    """
    assert {r for r, _ in _rules(src)} == set(RULES)


def test_scan_file_exempts_only_the_fft_owner(tmp_path):
    owner = tmp_path / "infrastructure" / "physics" / "fft_ops.py"
    other = tmp_path / "models" / "m.py"
    for p in (owner, other):
        p.parent.mkdir(parents=True)
        p.write_text("def f(x):\n    return torch.fft.fft2(x)\n")
    assert scan_file(owner) == []
    assert [f.rule for f in scan_file(other)] == ["raw_fft"]
