import pytest
from fastcons import cons, nil

from microkanren import (
    OccursError,
    SENTINEL,
    State,
    Symbol,
    Var,
    empty_state,
    empty_sub,
    extend_substitution,
    mplus,
    mzero,
    pull,
    take,
    unit,
    unify,
    walk,
)


def test_extend_substitution():
    val = object()
    s = extend_substitution(Var(0), val, empty_sub())
    assert walk(Var(0), s) is val


def test_walk_unbound_var():
    assert walk(Var(0), empty_sub()) == Var(0)


def test_walk_unbound_value():
    assert walk("foo", empty_sub()) == "foo"


def test_recursive_walk():
    val = object()
    s = extend_substitution(
        Var(0), Var(1), extend_substitution(Var(1), val, empty_sub())
    )
    assert walk(Var(0), s) is val


def test_symbol_equality():
    s1 = Symbol("test")
    s2 = Symbol("test")
    s3 = Symbol("other")
    assert s1 is s2  # Same symbols are identical
    assert s1 == s2  # Same symbols are equal
    assert s1 != s3  # Different symbols are not equal
    assert str(s1) == "test"
    assert repr(s1) == "test"


def test_empty_state():
    state = empty_state()
    assert state.counter == 0
    assert state.sub == empty_sub()


def test_unify_basic():
    s = empty_sub()
    # Equal atoms unify
    assert unify(1, 1, s) == s
    assert unify("a", "a", s) == s
    # Different atoms don't unify
    assert unify(1, 2, s) is SENTINEL
    assert unify("a", "b", s) is SENTINEL


def test_unify_var():
    s = empty_sub()
    x = Var(0)
    y = Var(1)
    # Var unifies with anything
    assert unify(x, 1, s) == extend_substitution(x, 1, s)
    assert unify(1, x, s) == extend_substitution(x, 1, s)
    # Two vars unify
    assert unify(x, y, s) == extend_substitution(x, y, s)


def test_unify_sequences():
    s = empty_sub()
    # Equal sequences unify
    assert unify((1, 2), (1, 2), s) == s
    assert unify([1, 2], [1, 2], s) == s
    # Different sequences don't unify
    assert unify((1, 2), (1, 3), s) is SENTINEL
    assert unify([1, 2], [1], s) is SENTINEL


def test_stream_operations():
    state = empty_state()
    # Empty stream
    assert mzero() == ()
    # Unit stream
    assert unit(state) == (state, mzero())
    # Pull thunk
    thunk = lambda: unit(state)
    assert pull(thunk) == unit(state)
    # Take from stream
    assert take(1, unit(state)) == [state]
    assert take(2, unit(state)) == [state]
    assert take(1, mzero()) == []


def test_mplus():
    state = empty_state()
    s1 = unit(state)
    s2 = unit(State(1, empty_sub()))
    # Combine two streams
    combined = mplus(s1, s2)
    assert take(2, combined) == [state, State(1, empty_sub())]


@pytest.mark.parametrize(
    "val",
    [
        Var(0),
        cons(Var(0), nil()),
        (Var(0), Var(0)),
        [Var(0)],
        [[Var(0)]],
        ([Var(0)],),
    ],
)
def test_occurs_check_raises(val):
    with pytest.raises(OccursError):
        extend_substitution(Var(0), val, empty_sub())
