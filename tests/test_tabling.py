from fastcons import cons

from microkanren.core import (
    State,
    Symbol,
    Var,
    call_fresh,
    conj,
    disj,
    empty_state,
    empty_sub,
    eq,
    reify,
    tabled,
    take,
)


def test_primary_call_cached():
    @tabled
    def goal(x):
        return eq(x, 5)

    call_fresh(goal)(empty_state())
    assert goal._table[(Symbol("_.0"),)] == [(5,)]


def test_compound_primary_call_cached():
    @tabled
    def goal(x):
        return eq(x, cons("a", "b"))

    x, y = Var(0), Var(1)
    args = (cons(x, y),)
    goal(*args)(empty_state())
    assert goal._table[args] == [(cons("a", "b"),)]


def test_primary_call_cached_with_ground_arg():
    @tabled
    def goal(x):
        return eq(x, 5)

    goal(5)(empty_state())
    assert goal._table[(5,)] == [(5,)]


def test_primary_call_arg_walked():
    @tabled
    def goal(x):
        return eq(x, 5)

    x = Var(0)
    goal(x)(State(empty_sub().set(x, 5)))
    assert goal._table[(5,)] == [(5,)]


def test_compound_primary_call_arg_walked():
    """
    Arguments to a tabled goal are walked before being used as keys in the table.
    """

    @tabled
    def goal(x):
        return eq(x, cons("a", "b"))

    x, y = Var("x"), Var("y")
    initial_state = State(empty_sub().set(x, "a").set(y, "b"))
    goal(cons(x, y))(initial_state)

    args = (cons("a", "b"),)
    assert goal._table[args] == [(cons("a", "b"),)]


def test_caching_idempotent():
    @tabled
    def goal(x):
        return eq(x, 5)

    call_fresh(goal)(empty_state())
    call_fresh(goal)(empty_state())
    assert goal._table[(Symbol("_.0"),)] == [(5,)]
    assert len(goal._table) == 1


def test_multiple_solutions_cached():
    @tabled
    def goal(x):
        return disj(eq(x, "a"), eq(x, "b"))

    call_fresh(goal)(empty_state())
    assert goal._table[(Symbol("_.0"),)] == [("a",), ("b",)]


def test_recursive_tabled_goal():
    """
    Tabling prevents infinite recursion in trivially recursive goals.
    """

    @tabled
    def fives(x):
        return disj(eq(x, 5), lambda s: fives(x)(s))

    result = call_fresh(fives)(empty_state())
    # Should only return one result (5) instead of infinitely recurring
    assert len(take(5, result)) == 1


def test_mutual_recursion_tabled():
    """
    Tabling works with mutually recursive goals.
    """

    @tabled
    def fives_and_sixes(x):
        return disj(
            eq(x, 5),
            sixes_and_fives(x),
        )

    @tabled
    def sixes_and_fives(x):
        return disj(
            eq(x, 6),
            fives_and_sixes(x),
        )

    result = call_fresh(fives_and_sixes)(empty_state())
    assert len(take(5, result)) == 2


def test_reuse_cached_results():
    """
    Test that cached results are properly reused in subsequent calls.
    """

    @tabled
    def goal(x):
        return disj(eq(x, 1), eq(x, 2))

    # Make first call to populate cache
    x = Var(0)
    s1 = goal(x)(empty_state())
    assert len(goal._table) == 1

    # Make second call - should reuse cached results
    s2 = goal(x)(empty_state())

    r1 = [reify(x, state.sub) for state in take(2, s1)]
    r2 = [reify(x, state.sub) for state in take(2, s2)]

    # Results should be identical
    assert r1 == r2


def test_tabled_with_multiple_vars():
    """
    Test tabling behavior with goals that involve multiple variables.
    """

    @tabled
    def goal(x, y):
        return conj(eq(x, 1), eq(y, 2))

    x, y = Var(0), Var(1)
    goal(x, y)(empty_state())

    assert len(goal._table) == 1
    assert goal._table[(Symbol("_.0"), Symbol("_.1"))] == [(1, 2)]


def test_tabled_alpha_equivalence():
    """
    Test that tabling correctly handles alpha-equivalent terms.
    """

    @tabled
    def goal(x, y):
        return eq(x, y)

    a, b, x, y = Var("a"), Var("b"), Var("x"), Var("y")

    # These calls should be considered equivalent
    result = take(
        5,
        conj(goal(a, b), goal(x, y))(empty_state()),
    )
    r1 = [reify((a, b), state.sub) for state in result]
    r2 = [reify((x, y), state.sub) for state in result]
    assert r1 == r2

    # Should use same cache entry
    assert len(goal._table) == 1
