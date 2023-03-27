from microkanren import (
    Goal,
    State,
    Stream,
    Var,
    conj,
    disj,
    eq,
    fail,
    fresh,
    ifte,
    neq,
)
from microkanren.cons import Cons, cons, nil


def appendo(xs: Cons | Var, ys: Cons | Var, zs: Cons | Var) -> Goal:
    return disj(
        nullo(xs) & eq(ys, zs),
        fresh(
            lambda a, d, res: conj(
                conso(a, d, xs),
                conso(a, res, zs),
                appendo(d, ys, res),
            )
        ),
    )


def membero(x, xs):
    return fresh(
        lambda a, d: conj(
            conso(a, d, xs),
            disj(eq(a, x), membero(x, d)),
        )
    )


def nullo(x):
    return eq(x, nil())


def notnullo(x):
    return neq(x, nil())


def conso(a, d, ls):
    return eq(cons(a, d), ls)


def caro(a, xs):
    return fresh(lambda d: conso(a, d, xs))


def cdro(d, xs):
    return fresh(lambda a: conso(a, d, xs))


def listo(xs):
    return nullo(xs) | fresh(lambda d: cdro(d, xs) & listo(d))


def inserto(x, ys, zs):
    def _inserto(state: State) -> Stream:
        return disj(
            appendo(cons(x, nil()), ys, zs),
            fresh(
                lambda a, d, res: conj(
                    eq((a, d), ys),
                    eq((a, res), zs),
                    inserto(x, d, res),
                ),
            ),
        )(state)

    return Goal(_inserto)


def assoco(x, xs, y):
    return ifte(
        eq(xs, nil()),
        fail(),
        fresh(
            lambda key, val, rest: disj(
                conj(
                    conso((key, val), rest, xs),
                    disj(
                        eq(key, x) & eq((key, val), y),
                        assoco(x, rest, y),
                    ),
                )
            ),
        ),
    )


def rembero(x, xs, out):
    def _rembero(state: State) -> Stream:
        return disj(
            nullo(xs) & nullo(out),
            fresh(
                lambda a, d, res: disj(
                    conso(x, d, xs) & eq(d, out),
                    conj(
                        neq(a, x),
                        conso(a, d, xs),
                        conso(a, res, out),
                        rembero(x, d, res),
                    ),
                )
            ),
        )(state)

    return Goal(_rembero)


def joino(prefix, suffix, sep, out):
    return disj(
        nullo(prefix) & eq(suffix, out),
        nullo(suffix) & eq(prefix, out),
        fresh(
            lambda tmp: notnullo(prefix)
            & notnullo(suffix)
            & appendo(prefix, sep, tmp)
            & appendo(tmp, suffix, out)
        ),
    )
