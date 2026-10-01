from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class, Function

MAX_COMPLEXITY = 15
PACKAGE = Path(__file__).resolve().parents[1] / "trex"


def functions(blocks: list[Function | Class]) -> list[Function]:
    out: list[Function] = []
    for b in blocks:
        out += functions(b.methods) if isinstance(b, Class) else [b, *functions(b.closures)]
    return out


def test_every_python_function_stays_within_the_complexity_limit():
    over = sorted({f"{f.name}:{fn.lineno} {fn.fullname}: {fn.complexity}"
                   for f in PACKAGE.glob("*.py") for fn in functions(cc_visit(f.read_text())) if fn.complexity > MAX_COMPLEXITY})
    assert not over, "\n".join(over)
