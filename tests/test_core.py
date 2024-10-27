import pytest
from fastcons import cons, nil

from microkanren import (
    OccursError,
    Var,
    empty_sub,
    extend_substitution,
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
