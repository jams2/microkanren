from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(slots=True)
class nil:
    pass


@dataclass(slots=True, repr=False)
class cons:
    head: Any
    tail: Any
    is_proper: bool = False

    def __init__(self, head, tail):
        self.head = head
        self.tail = tail
        match tail:
            case nil():
                self.is_proper = True
            case cons(_, _) as d if d.is_proper:
                self.is_proper = True
            case _:
                self.is_proper = False

    def __iter__(self):
        yield self.head
        yield self.tail

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head!r}, {self.tail!r})"

    @classmethod
    def from_python(cls, ls):
        accum = nil()
        for x in reversed(ls):
            accum = cons(x, accum)
        return accum

    def _to_str(self):
        return "".join(chr(char.i) for char in self._to_list())

    def _to_list(self):
        ls = [to_python(self.head)]
        cons = self.tail
        while cons != nil():
            ls.append(to_python(cons.head))
            cons = cons.tail
        return ls

    def to_python(self):
        if not self.is_proper:
            raise ValueError("Can't convert improper cons list to Python")
        if isinstance(self.head, Char):
            return self._to_str()
        return self._to_list()


Cons: TypeAlias = cons | nil


@dataclass(slots=True)
class Char:
    i: int

    def __repr__(self):
        return f"Char({self.i})"


def string(cs):
    return cons.from_python([Char(ord(c)) for c in cs])


def from_python(obj):
    if isinstance(obj, list):
        return cons.from_python(list(map(from_python, obj)))
    elif isinstance(obj, dict):
        return cons.from_python(
            [(from_python(k), from_python(v)) for k, v in obj.items()]
        )
    elif isinstance(obj, tuple):
        return tuple(from_python(x) for x in obj)
    elif isinstance(obj, str):
        return string(obj)
    return obj


def to_python(obj):
    if obj == nil():
        return []
    elif isinstance(obj, cons):
        if obj.is_proper:
            return obj.to_python()
        return cons(to_python(obj.head), to_python(obj.tail))
    elif isinstance(obj, tuple):
        return tuple(to_python(x) for x in obj)
    else:
        return obj
