from fastcons import cons, nil

from microkanren import fresh, snooze
from microkanren.tabling import Cache, ReifiedVar, conj, disj, eq, run, tabled


def _(*xs):
    return cons.from_xs(xs)


def consᵒ(a, d, pair):
    return eq(cons(a, d), pair)


def test_simple_tabled_goal():
    @tabled
    def five(x):
        return eq(x, 5)

    assert run(3, lambda x: five(x)) == [5]


def test_disj():
    @tabled
    def goal(x):
        return disj(
            eq(x, 5),
            eq(x, 6),
        )

    assert not goal._table
    assert run(3, lambda x: goal(x)) == [5, 6]
    assert goal._table == {(ReifiedVar(0),): Cache(answers=[(5,), (6,)])}

    # Running the same goal a subsequent time will reuse the cache
    assert run(3, lambda x: goal(x)) == [5, 6]


def test_disj_operator():
    @tabled
    def goal(x):
        return eq(x, 5) | eq(x, 6)

    assert run(3, lambda x: goal(x)) == [5, 6]

    # Running the same goal a subsequent time will reuse the cache
    assert run(3, lambda x: goal(x)) == [5, 6]


def test_patho():
    def patho(x, y):
        return disj(
            arco(x, y),
            fresh(lambda z: conj(arco(x, z), patho(z, y))),
        )

    def arco(x, y):
        return disj(
            eq(x, "a") & eq(y, "b"),
            eq(x, "b") & eq(y, "a"),
            eq(x, "b") & eq(y, "d"),
        )

    # Without tabling, we get an endless series of results due to the cycle a → b → a
    assert run(6, lambda x: patho("a", x)) == ["b", "a", "d", "b", "a", "d"]

    @tabled
    def patho_t(x, y):
        return disj(
            arco(x, y),
            fresh(lambda z: conj(arco(x, z), patho_t(z, y))),
        )

    assert run(6, lambda x: patho_t("a", x)) == ["b", "a", "d"]


def test_mutually_recursive_goals_untabled():
    # Establish a baseline for the following tests
    def fᵒ(x):
        return eq(x, 0) | snooze(gᵒ, x)

    def gᵒ(x):
        return eq(x, 1) | snooze(fᵒ, x)

    assert run(5, lambda x: fᵒ(x)) == [0, 1, 0, 1, 0]


def test_mutually_recursive_goals_f_tabled():
    @tabled
    def fᵒ(x):
        return eq(x, 0) | snooze(gᵒ, x)

    def gᵒ(x):
        return eq(x, 1) | snooze(fᵒ, x)

    assert run(5, lambda x: fᵒ(x)) == [0, 1]
    assert run(5, lambda x: gᵒ(x)) == [1, 0, 1]


def test_mutually_recursive_goals_g_tabled():
    def fᵒ(x):
        return eq(x, 0) | snooze(gᵒ, x)

    @tabled
    def gᵒ(x):
        return eq(x, 1) | snooze(fᵒ, x)

    assert run(5, lambda x: fᵒ(x)) == [0, 1, 0]
    assert run(5, lambda x: gᵒ(x)) == [1, 0]


def test_mutually_recursive_goals_both_tabled():
    @tabled
    def fᵒ(x):
        return eq(x, 0) | snooze(gᵒ, x)

    @tabled
    def gᵒ(x):
        return eq(x, 1) | snooze(fᵒ, x)

    assert run(5, lambda x: fᵒ(x)) == [0, 1]
    assert run(5, lambda x: gᵒ(x)) == [1, 0]


def test_left_recursive_parser():
    # term → term '+' 'a' | 'a'

    @tabled
    def term(start, rest):
        return disj(
            consᵒ("a", rest, start),
            fresh(
                lambda _1, _2: conj(
                    term(start, _1),
                    consᵒ("+", _2, _1),
                    consᵒ("a", rest, _2),
                )
            ),
        )

    result = run(2, lambda x: term(_("a", "+", "a"), x))
    breakpoint()
    result = run(1, lambda x: term(x, nil()))
