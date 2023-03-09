from pyrsistent import pmap

from microkanren import (
    Var,
    empty_sub,
    eq,
    extend_substitution,
    get_sub_prefix,
    run_all,
    walk,
)


class TestSubstitution:
    def test_extend_substitution(self):
        val = object()
        s = extend_substitution(Var(0), val, empty_sub())
        assert walk(Var(0), s) is val

    def test_walk_self(self):
        assert walk(Var(0), empty_sub()) == Var(0)

    def test_walk_constant(self):
        assert walk("foo", empty_sub()) == "foo"

    def test_recursive_walk(self):
        val = object()
        s = extend_substitution(
            Var(0), Var(1), extend_substitution(Var(1), val, empty_sub())
        )
        assert walk(Var(0), s) is val

    def test_get_sub_prefix(self):
        x = object()
        a = pmap({Var(0): 1})
        b = a.set(Var(1), Var(2))
        c = b.set(Var(2), x)
        assert get_sub_prefix(c, a) == pmap({Var(1): Var(2), Var(2): x})


class TestEq:
    def test_simple_eq(self):
        result = run_all(lambda x: eq(x, 1))
        assert result == [1]