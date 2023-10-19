from pyrsistent import pmap

from microkanren import (
    Var,
    disj,
    empty_sub,
    eq,
    extend_substitution,
    fresh,
    get_sub_prefix,
    run,
    run_all,
    snooze,
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
        initial = empty_sub()
        a = extend_substitution(Var(0), 1, initial)
        b = extend_substitution(
            Var(2),
            x,
            extend_substitution(Var(1), Var(2), a),
        )
        assert set(get_sub_prefix(b, a).items()) == set(
            pmap({Var(1): Var(2), Var(2): x}).items()
        )


class TestEq:
    def test_simple_eq(self):
        result = run_all(lambda x: eq(x, 1))
        assert result == [1]


def fives(x):
    return eq(x, 5) | snooze(fives, x)


def sixes(x):
    return eq(x, 6) | snooze(sixes, x)


def test_snooze():
    result = run(3, lambda x: fives(x))
    assert result == [5, 5, 5]

    result = run(8, lambda x: fives(x) | sixes(x))
    assert result == [5, 6, 5, 6, 5, 6, 5, 6]


def test_recursion():
    # Check we don't blow python's stack
    assert len(run(10000, lambda x: fives(x))) == 10000


def test_disj_interleaving():
    # Test that the order of results matches examples from the miniKanren paper

    def function_disj_relation(x):
        return disj(eq(x, 1), eq(x, 2), eq(x, 3), snooze(function_disj_relation, x))

    assert run(6, lambda x: function_disj_relation(x)) == [1, 2, 3, 1, 2, 3]

    def operator_disj_relation(x):
        return eq(x, 1) | eq(x, 2) | eq(x, 3) | snooze(operator_disj_relation, x)

    assert run(6, lambda x: function_disj_relation(x)) == [1, 2, 3, 1, 2, 3]

    def patho(x, y):
        return arco(x, y) | fresh(lambda z: arco(x, z) & patho(z, y))

    def arco(x, y):
        return disj(
            eq("a", x) & eq("b", y),
            eq("b", x) & eq("a", y),
            eq("b", x) & eq("d", y),
        )

    assert "".join(run(9, lambda x: patho("a", x))) == "badbadbad"
