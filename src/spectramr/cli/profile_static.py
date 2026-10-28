"""AST scan for performance-rule violations, keyed by the function that holds them.

Scalene says where the time went; this says where the source breaks the
training-loop / backend rules (non-negotiables 2, 9, 25). :mod:`profile_hotspots`
joins the two, so a function that is both hot and flagged outranks one that is
only either. The scan is *syntactic*: a finding is a suspect, never a measurement.

Function names are Python ``__qualname__``s (``Class.method``,
``outer.<locals>.inner``) because that is the spelling Scalene reports, so the
join needs no translation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

#: Method names that force a device->host sync or copy when called on a tensor.
HOST_SYNC_ATTRS = frozenset({"item", "cpu", "numpy", "tolist"})

#: Dotted prefixes of FFT calls that bypass ``fft2c``/``ifft2c`` (rule 2).
RAW_FFT_PREFIXES = ("np.fft.", "numpy.fft.", "scipy.fft.", "torch.fft.")

#: The one module allowed to call ``torch.fft`` directly.
FFT_OWNER_SUFFIX = "infrastructure/physics/fft_ops.py"

RULES: dict[str, str] = {
    "host_sync_in_loop": "`.item()/.cpu()/.numpy()/.tolist()` inside a loop forces a host sync per iteration",
    "empty_cache_in_loop": "`empty_cache()` inside a loop fragments the allocator and stalls the stream",
    "raw_fft": "FFT call that bypasses fft2c/ifft2c (centering and ortho norm); spectral operators on real features are exempt",
    "loop_over_extent": "Python loop over a tensor extent (batch/coil/slice) that a batched op can replace",
}


@dataclass(frozen=True)
class StaticFinding:
    rule: str
    lineno: int
    function: str
    detail: str


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _ranges_over_extent(node: ast.For) -> bool:
    it = node.iter
    if not (isinstance(it, ast.Call) and _dotted(it.func) == "range"):
        return False
    for sub in (n for arg in it.args for n in ast.walk(arg)):
        if isinstance(sub, ast.Attribute) and sub.attr == "shape":
            return True
        if isinstance(sub, ast.Call) and (
            _dotted(sub.func) == "len"
            or (isinstance(sub.func, ast.Attribute) and sub.func.attr == "size")
        ):
            return True
    return False


class _Scan(ast.NodeVisitor):
    def __init__(self, *, fft_exempt: bool) -> None:
        self.fft_exempt = fft_exempt
        self.findings: list[StaticFinding] = []
        self._scope: list[tuple[str, bool]] = []  # (name, is_function)
        self._loop = 0

    def _qualname(self) -> str:
        out: list[str] = []
        for i, (name, is_func) in enumerate(self._scope):
            out.append(name)
            if is_func and i < len(self._scope) - 1:
                out.append("<locals>")
        return ".".join(out) or "<module>"

    def _add(self, rule: str, node: ast.AST, detail: str) -> None:
        self.findings.append(StaticFinding(rule, node.lineno, self._qualname(), detail))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append((node.name, False))
        self.generic_visit(node)
        self._scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope.append((node.name, True))
        outer, self._loop = self._loop, 0  # a def inside a loop does not run per iteration
        self.generic_visit(node)
        self._loop = outer
        self._scope.pop()

    visit_FunctionDef = _visit_function  # noqa: N815 - ast.NodeVisitor dispatch names
    visit_AsyncFunctionDef = _visit_function  # noqa: N815

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node, ast.For) and _ranges_over_extent(node):
                self._add("loop_over_extent", node, ast.unparse(node.iter))
            self.visit(node.iter)
        else:
            self.visit(node.test)
        self._loop += 1
        for stmt in [*node.body, *node.orelse]:
            self.visit(stmt)
        self._loop -= 1

    visit_For = visit_AsyncFor = visit_While = _visit_loop  # noqa: N815

    def _visit_comprehension(self, node: ast.AST) -> None:
        self._loop += 1
        self.generic_visit(node)
        self._loop -= 1

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension  # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if self._loop and attr in HOST_SYNC_ATTRS:
            self._add("host_sync_in_loop", node, f".{attr}()")
        if self._loop and attr == "empty_cache":
            self._add("empty_cache_in_loop", node, name)
        if not self.fft_exempt and name.startswith(RAW_FFT_PREFIXES):
            self._add("raw_fft", node, name)
        self.generic_visit(node)


def scan_source(source: str, *, fft_exempt: bool = False) -> list[StaticFinding]:
    """Scan one module's source. Raises ``SyntaxError`` on unparseable input."""
    scan = _Scan(fft_exempt=fft_exempt)
    scan.visit(ast.parse(source))
    return scan.findings


def scan_file(path: Path) -> list[StaticFinding]:
    """Scan one file; the FFT owner module is exempt from ``raw_fft``."""
    exempt = path.as_posix().endswith(FFT_OWNER_SUFFIX)
    return scan_source(path.read_text(encoding="utf-8"), fft_exempt=exempt)


__all__ = ["RULES", "StaticFinding", "scan_file", "scan_source"]
