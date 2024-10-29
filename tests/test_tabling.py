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


def test_recursive_tabled_goal():
    """
    Test that tabling prevents infinite recursion in recursive goals.
    """
    @tabled
    def fives(x):
        return disj(eq(x, 5), lambda s: fives(x)(s))
    
    result = call_fresh(fives)(empty_state())
    # Should only return one result (5) instead of infinitely recurring
    assert len(take(5, result)) == 1


def test_mutual_recursion_tabled():
    """
    Test that tabling works with mutually recursive goals.
    """
    @tabled 
    def odds(x):
        return disj(
            eq(x, 1),
            call_fresh(lambda y: conj(
                eq(x, cons(2, y)),
                evens(y)
            ))
        )
    
    @tabled
    def evens(x):
        return disj(
            eq(x, 2),
            call_fresh(lambda y: conj(
                eq(x, cons(1, y)),
                odds(y)
            ))
        )

    result = call_fresh(odds)(empty_state())
    # Should generate a finite number of odd/even alternating sequences
    assert len(take(5, result)) > 0


def test_reuse_cached_results():
    """
    Test that cached results are properly reused in subsequent calls.
    """
    @tabled
    def goal(x):
        return disj(eq(x, 1), eq(x, 2))
    
    # Make first call to populate cache
    s1 = call_fresh(goal)(empty_state())
    assert len(goal._table) == 1
    
    # Make second call - should reuse cached results
    s2 = call_fresh(goal)(empty_state())
    # Results should be identical
    assert take(2, s1) == take(2, s2)


def test_tabled_with_multiple_vars():
    """
    Test tabling behavior with goals that involve multiple variables.
    """
    @tabled
    def pair_sum(x, y, sum):
        return disj(
            conj(eq(x, 1), conj(eq(y, 2), eq(sum, 3))),
            conj(eq(x, 2), conj(eq(y, 2), eq(sum, 4)))
        )
    
    result = call_fresh(lambda a: 
                call_fresh(lambda b:
                    call_fresh(lambda c: 
                        pair_sum(a, b, c))))(empty_state())
    # Should cache and return both solutions
    assert len(take(2, result)) == 2


def test_tabled_alpha_equivalence():
    """
    Test that tabling correctly handles alpha-equivalent terms.
    """
    @tabled
    def goal(x, y):
        return eq(x, y)
    
    # These calls should be considered equivalent
    s1 = call_fresh(lambda a: call_fresh(lambda b: goal(a, b)))(empty_state())
    s2 = call_fresh(lambda c: call_fresh(lambda d: goal(c, d)))(empty_state())
    
    # Should use same cache entry
    assert len(goal._table) == 1
