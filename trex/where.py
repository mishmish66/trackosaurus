"""Run filters: a SQL WHERE clause over run fields, the same in the CLI and the browser (`static/where.js`).

    lr = 0.001 and seed in (0, 1)
    algo like 'PP%' or not name ~ 'debug'
    summary.eval/return >= 200 and tags = baseline
    info.notes is not null

Text with no comparison is a case-insensitive regex search of run names and paths. Numbers compare as numbers;
`=` on text is exact; `like` (`%`, `_`) and `~` (regex search) ignore case; a bare word is text; a list matches
when any element does; a missing value fails every comparison, and each negated form (`not`, `!=`, `!~`,
`not in`, `not like`) is the opposite of its positive.
"""

import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Final, NamedTuple, cast

type Getter = Callable[[str], object]
type Test = Callable[[Getter], bool]
type Literal = float | str | bool | None

_TOKEN: Final = re.compile(r"""\s*(?:('(?:[^']|'')*')|("(?:[^"]|"")*")|(>=|<=|!=|<>|==|!~|[=<>~(),])|([^\s'"=<>!~(),]+)|(\S))""")
_NUMBER: Final = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_MARKERS: Final = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_KEYWORDS: Final = frozenset({"and", "or", "not", "in", "like", "is", "null", "true", "false"})
_COMPARE: Final = frozenset({"=", "==", "!=", "<>", "<", "<=", ">", ">=", "~", "!~"})


class Where(NamedTuple):
    test: Test
    fields: list[str]  # fields the clause reads, in order


class Token(NamedTuple):
    kind: str  # "str", "id", "op", "word"
    text: str


def compile_where(text: str) -> Where:
    """The filter `text` means; ValueError for a clause that does not parse."""
    toks = _tokens(text)
    if toks is None or not any((t.kind == "op" and t.text in _COMPARE) or _is_kw(t, "in", "like", "is") for t in toks):
        return _search(text)
    p = _Parser(toks)
    test = p.expr()
    if p.i < len(toks):
        raise ValueError(f"unexpected {toks[p.i].text!r}")
    return Where(test, p.fields)


def _search(text: str) -> Where:
    try:
        rx = re.compile(text.strip(), re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(text.strip()), re.IGNORECASE)
    return Where(lambda g: any(rx.search(text_of(g(f))) for f in ("name", "path") if g(f) is not None), ["name", "path"])


def _tokens(text: str) -> list[Token] | None:
    out: list[Token] = []
    pos = 0
    while pos < len(text) and text[pos:].strip():
        m = _TOKEN.match(text, pos)
        if m is None or m.group(5) is not None:
            return None
        pos = m.end()
        if m.group(1) is not None:
            out.append(Token("str", m.group(1)[1:-1].replace("''", "'")))
        elif m.group(2) is not None:
            out.append(Token("id", m.group(2)[1:-1].replace('""', '"')))
        elif m.group(3) is not None:
            out.append(Token("op", m.group(3)))
        else:
            out.append(Token("word", m.group(4)))
    return out


def _is_kw(t: Token | None, *words: str) -> bool:
    return t is not None and t.kind == "word" and t.text.lower() in words


def as_number(v: object) -> float | None:
    """A number, or the text of one (including the markers "NaN", "Infinity", "-Infinity"); else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if v in _MARKERS:
            return _MARKERS[v]
        if _NUMBER.fullmatch(v.strip()):
            return float(v)
    return None


def text_of(v: object) -> str:
    """The text a value is matched as: strings as they are, true/false, integers without a point, JSON otherwise."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if math.isfinite(v) and float(v).is_integer() and abs(v) < 1e16:
            return str(int(v))
        return "NaN" if math.isnan(v) else "Infinity" if v == math.inf else "-Infinity" if v == -math.inf else repr(float(v))
    return json.dumps(v, separators=(",", ":"))


def _any(v: object, f: Callable[[object], bool]) -> bool:
    if v is None:
        return False
    if isinstance(v, list):
        return any(f(x) for x in cast(list[object], v) if x is not None)
    return f(v)


def _equal(v: object, lit: Literal) -> bool:
    if isinstance(v, bool) and isinstance(lit, bool):
        return v == lit
    a, b = as_number(v), as_number(lit)
    if a is not None and b is not None:
        return a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))
    return text_of(v) == text_of(lit)


def _order(v: object, lit: Literal, op: str) -> bool:
    a, b = as_number(v), as_number(lit)
    if a is not None and b is not None:
        return _ordered(a, b, op)
    if isinstance(v, str) and isinstance(lit, str):
        return _ordered(v, lit, op)
    return False


def _ordered[T: (float, str)](x: T, y: T, op: str) -> bool:
    return x < y if op == "<" else x <= y if op == "<=" else x > y if op == ">" else x >= y


def _like(pattern: str) -> re.Pattern[str]:
    parts = (".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern)
    return re.compile("".join(parts), re.IGNORECASE | re.DOTALL)


class _Parser:
    def __init__(self, toks: Sequence[Token]) -> None:
        self.toks, self.i, self.fields = toks, 0, list[str]()

    def peek(self) -> Token | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> Token:
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end")
        self.i += 1
        return t

    def expect(self, text: str) -> None:
        t = self.take()
        if t.text.lower() != text or t.kind not in ("op", "word"):
            raise ValueError(f"expected {text!r}, got {t.text!r}")

    def expr(self) -> Test:
        left = self.conj()
        while _is_kw(self.peek(), "or"):
            self.i += 1
            a, b = left, self.conj()
            left = lambda g, a=a, b=b: a(g) or b(g)  # noqa: E731
        return left

    def conj(self) -> Test:
        left = self.unary()
        while _is_kw(self.peek(), "and"):
            self.i += 1
            a, b = left, self.unary()
            left = lambda g, a=a, b=b: a(g) and b(g)  # noqa: E731
        return left

    def unary(self) -> Test:
        if _is_kw(self.peek(), "not"):
            self.i += 1
            e = self.unary()
            return lambda g: not e(g)
        t = self.peek()
        if t is not None and t.kind == "op" and t.text == "(":
            self.i += 1
            e = self.expr()
            self.expect(")")
            return e
        return self.comparison()

    def field(self) -> str:
        t = self.take()
        if t.kind == "id" or (t.kind == "word" and t.text.lower() not in _KEYWORDS):
            self.fields.append(t.text)
            return t.text
        raise ValueError(f"expected a field, got {t.text!r}")

    def value(self) -> Literal:
        t = self.take()
        if t.kind == "str":
            return t.text
        if t.kind != "word":
            raise ValueError(f"expected a value, got {t.text!r}")
        low = t.text.lower()
        if low in ("true", "false"):
            return low == "true"
        if low == "null":
            return None
        if low in _KEYWORDS:
            raise ValueError(f"expected a value, got {t.text!r}")
        return float(t.text) if _NUMBER.fullmatch(t.text) else t.text

    def comparison(self) -> Test:
        f = self.field()
        if _is_kw(self.peek(), "not"):
            self.i += 1
            if not _is_kw(self.peek(), "in", "like"):
                raise ValueError("expected 'in' or 'like' after 'not'")
            test = self.membership(f)
            return lambda g: not test(g)
        if _is_kw(self.peek(), "in", "like"):
            return self.membership(f)
        if _is_kw(self.peek(), "is"):
            self.i += 1
            want_null = not _is_kw(self.peek(), "not")
            if not want_null:
                self.i += 1
            self.expect("null")
            return lambda g: (g(f) is None) == want_null
        t = self.take()
        if t.kind != "op" or t.text not in _COMPARE:
            raise ValueError(f"expected an operator after {f!r}, got {t.text!r}")
        return _operator(f, t.text, self.value())

    def membership(self, f: str) -> Test:
        if _is_kw(self.take(), "like"):
            rx = _like(text_of(self.value()))
            return lambda g: _any(g(f), lambda v: rx.fullmatch(text_of(v)) is not None)
        self.expect("(")
        values = [self.value()]
        while (t := self.peek()) is not None and t.kind == "op" and t.text == ",":
            self.i += 1
            values.append(self.value())
        self.expect(")")
        return lambda g: _any(g(f), lambda v: any(_equal(v, x) for x in values))


def _operator(f: str, op: str, lit: Literal) -> Test:
    if lit is None and op in ("=", "==", "!=", "<>"):
        return lambda g: (g(f) is None) == (op in ("=", "=="))
    if op in ("=", "=="):
        return lambda g: _any(g(f), lambda v: _equal(v, lit))
    if op in ("!=", "<>"):
        return lambda g: not _any(g(f), lambda v: _equal(v, lit))
    if op in ("~", "!~"):
        rx = re.compile(text_of(lit), re.IGNORECASE)
        hit: Test = lambda g: _any(g(f), lambda v: rx.search(text_of(v)) is not None)  # noqa: E731
        return hit if op == "~" else lambda g: not hit(g)
    return lambda g: _any(g(f), lambda v: _order(v, lit, op))
