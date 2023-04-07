from dataclasses import dataclass

from fastcons import cons


@dataclass(slots=True)
class Char:
    i: int

    def __repr__(self):
        return f"Char({self.i})"


def string(cs):
    return cons.from_xs(Char(ord(c)) for c in cs)
