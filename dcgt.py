from fastcons import cons

from microkanren import fresh
from microkanren.tabling import conj, disj, eq, tabled


def _(*xs):
    return cons.from_xs(xs)


def consᵒ(a, d, pair):
    return eq(cons(a, d), pair)


@tabled
def term(start, rest):
    return disj(
        fresh(
            lambda _1, _2: conj(
                term(start, _1),
                consᵒ("+", _2, _1),
                consᵒ("a", rest, _2),
            )
        ),
        consᵒ("a", rest, start),
    )
