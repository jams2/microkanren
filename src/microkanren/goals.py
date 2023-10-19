from fastcons import cons, nil

from microkanren.core import (
    Goal,
    GoalProto,
    State,
    Stream,
    Var,
    bind,
    conj,
    disj,
    eq,
    fail,
    fresh,
    mzero,
    neq,
    succeed,
    unit,
)


def conda(*cases) -> GoalProto:
    _cases = []
    for case in cases:
        if isinstance(case, list | tuple):
            _cases.append((case[0], succeed) if len(case) < 2 else case)
        else:
            _cases.append((case, succeed()))

    def _conda(state: State) -> Stream:
        return starfoldr(ifte, _cases, fail())(state)

    return Goal(_conda)


def starfoldr(f, xs, initial):
    sentinel = object()
    accum = sentinel
    for args in reversed(xs):
        if accum is sentinel:
            _args = (*args, initial)
        else:
            _args = (*args, accum)
        accum = f(*_args)
    return accum


def ifte(g1, g2, g3=None) -> GoalProto:
    g3 = g3 or fail()

    def _ifte(state: State) -> Stream:
        # TODO: rewrite iteratively
        def ifte_loop(stream: Stream) -> Stream:
            match stream:
                case ():
                    return g3(state)
                case (_, _):
                    return bind(stream, g2)
                case _:
                    return lambda: ifte_loop(stream())

        return ifte_loop(g1(state))

    return Goal(_ifte)


def onceo(g: GoalProto) -> GoalProto:
    def _onceo(state: State):
        stream = g(state)
        while stream:
            match stream:
                case (s1, _):
                    return unit(s1)
                case _:
                    stream = stream()
        return mzero

    return _onceo


def appendo(xs: cons | Var, ys: cons | Var, zs: cons | Var) -> Goal:
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
