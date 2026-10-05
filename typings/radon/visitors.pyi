from typing import NamedTuple

class Function(NamedTuple):
    name: str
    lineno: int
    col_offset: int
    endline: int
    is_method: bool
    classname: str | None
    closures: list[Function]
    complexity: int
    @property
    def fullname(self) -> str: ...

class Class(NamedTuple):
    name: str
    lineno: int
    col_offset: int
    endline: int
    methods: list[Function]
    inner_classes: list[Class]
    real_complexity: int
    @property
    def fullname(self) -> str: ...
    @property
    def complexity(self) -> int: ...
