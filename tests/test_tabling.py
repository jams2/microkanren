from fastcons import cons
from microkanren.core import (
    disj,
    eq,
    tabled,
    call_fresh,
    empty_state,
    Var,
    Symbol,
    State,
    empty_sub,
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

    call_fresh(lambda x: call_fresh(lambda y: goal(cons(x, y))))(empty_state())
    args = (cons(Var(0), Var(1)),)
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

    call_fresh(goal)(State(0, empty_sub().set(Var(0), 5)))
    assert goal._table[(5,)] == [(5,)]


def test_compound_primary_call_arg_walked():
    @tabled
    def goal(x):
        return eq(x, cons("a", "b"))

    call_fresh(lambda x: call_fresh(lambda y: goal(cons(x, y))))(
        State(0, empty_sub().set(Var(0), "a").set(Var(1), "b"))
    )
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

    thing = call_fresh(goal)(empty_state())
    assert goal._table[(Symbol("_.0"),)] == [("a",), ("b",)]
