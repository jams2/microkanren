from itertools import permutations

import pytest

from microkanren import (
    alldifffd,
    conj,
    domfd,
    eq,
    freshn,
    infd,
    ltefd,
    make_domain,
    mkrange,
    neq,
    neqfd,
    plusfd,
    run,
    run_all,
)


class TestFdConstraints:
    @pytest.mark.parametrize("domain", [make_domain(1, 2, 3), make_domain(5, 7, 9)])
    def test_domfd(self, domain):
        result = run_all(lambda x: domfd(x, domain))
        assert set(result) == {x for x in domain}

    @pytest.mark.parametrize(
        ("a", "b", "intersection"),
        [
            (make_domain(1, 2), make_domain(1, 2), {1, 2}),
            (make_domain(1, 2, 3), make_domain(2, 3, 4), {2, 3}),
            (make_domain(1, 2), make_domain(3, 4), make_domain()),
        ],
    )
    def test_domain_intersection(self, a, b, intersection):
        result = run_all(lambda x: domfd(x, a) & domfd(x, b))
        assert set(result) == intersection

    def test_violated_infd_fails(self):
        result = run_all(lambda x: domfd(x, make_domain(1, 2, 3)) & eq(x, 4))
        assert result == []

    @pytest.mark.parametrize(
        ("a", "b", "expected_x"),
        [
            (make_domain(1, 2, 3, 4), make_domain(2, 3), make_domain(1, 2, 3)),
            (make_domain(4, 5), make_domain(1, 2), make_domain()),
            (make_domain(3, 4), make_domain(2, 3, 4, 5), make_domain(3, 4)),
            (make_domain(1, 2, 3, 4), make_domain(1, 2, 3, 4), make_domain(1, 2, 3, 4)),
            (make_domain(1, 2, 3), make_domain(1, 2), make_domain(1, 2)),
        ],
    )
    def test_ltefd(self, a, b, expected_x):
        result = run_all(lambda x, y: domfd(x, a) & domfd(y, b) & ltefd(x, y))
        assert set(x[0] for x in result) == expected_x
        for x, y in result:
            assert x <= y

    def test_neq_with_domfd(self):
        """
        If neq(x, n), then n cannot be in the domain of x.
        """
        result = run_all(lambda x: domfd(x, make_domain(1, 2, 3)) & neq((x, 2)))
        assert set(result) == {1, 3}

    def test_neq_with_ltefd(self):
        """
        If neq(x, n), then n cannot be in the domain of x.
        """
        result = run_all(
            lambda x, y: domfd(x, make_domain(1, 2, 3))
            & domfd(y, make_domain(1, 2))
            & ltefd(x, y)
            & neq((x, 1))
        )
        assert result == [(2, 2)]

    def test_plusfd(self):
        result = run_all(
            lambda x, y, z: domfd(x, mkrange(1, 3))
            & domfd(y, mkrange(1, 3))
            & domfd(z, mkrange(1, 3))
            & plusfd(x, y, z)
        )
        assert set(result) == {(1, 1, 2), (1, 2, 3), (2, 1, 3)}

    def test_plusfd_with_ltefd(self):
        result = run_all(
            lambda x, y, z: domfd(x, mkrange(1, 3))
            & domfd(y, mkrange(1, 2))
            & domfd(z, mkrange(1, 4))
            & plusfd(x, y, z)
            & ltefd(x, y)
        )
        assert set(result) == {(1, 1, 2), (1, 2, 3), (2, 2, 4)}

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (make_domain(1), make_domain(1), set()),
            (make_domain(1, 2, 3), make_domain(1), {2, 3}),
            (make_domain(1, 2, 3), make_domain(3), {1, 2}),
            (make_domain(4, 5, 6), make_domain(3), {4, 5, 6}),
            (make_domain(4, 5, 6), make_domain(4, 5), {4, 5, 6}),
        ],
    )
    def test_neqfd(self, a, b, expected):
        result = run_all(lambda x, y: domfd(x, a) & domfd(y, b) & neqfd(x, y))
        assert set(x[0] for x in result) == expected
        for x, y in result:
            assert x != y

    def test_neqfd_with_ltefd(self):
        result = run_all(
            lambda x, y: domfd(x, mkrange(2, 3))
            & domfd(y, mkrange(1, 3))
            & ltefd(x, y)
            & neqfd(x, y)
        )
        assert set(result) == {(2, 3)}

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (1, 1, []),
            (2, 3, [(2, 3)]),
        ],
    )
    def test_alldifffd_constants(self, a, b, expected):
        result = run_all(lambda x, y: eq(x, a) & eq(y, b) & alldifffd(x, y))
        assert result == expected

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (make_domain(1, 2), make_domain(1, 2), {(1, 2), (2, 1)}),
            (make_domain(2), make_domain(1, 2, 3), {(2, 1), (2, 3)}),
            (
                make_domain(7, 9, 11),
                make_domain(1, 11),
                {(7, 1), (9, 1), (11, 1), (7, 11), (9, 11)},
            ),
        ],
    )
    def test_alldifffd_domains(self, a, b, expected):
        result = run_all(lambda x, y: domfd(x, a) & domfd(y, b) & alldifffd(x, y))
        assert set(result) == expected

    def test_alldifffd_many(self):
        result = run_all(
            lambda w, x, y, z: domfd(w, mkrange(1, 4))
            & domfd(x, mkrange(1, 4))
            & domfd(y, mkrange(1, 4))
            & domfd(z, mkrange(1, 4))
            & alldifffd(w, x, y, z)
        )
        assert set(result) == set(permutations((1, 2, 3, 4), 4))

    def test_infd(self):
        result = run_all(
            lambda a, b, c: infd((a, b, c), mkrange(1, 3)) & alldifffd(a, b, c)
        )
        assert set(result) == set(permutations((1, 2, 3), 3))


class TestLargeGoals:
    @pytest.mark.skip()
    def test_sudoku(self):
        def grido(grid, *vs):
            assert len(vs) == 81
            return eq(
                grid,
                [tuple(vs[j] for j in range(i, i + 9)) for i in range(0, 81, 9)],
            )

        def sudokuo(a, b):
            return freshn(
                162,
                lambda *vs: conj(
                    grido(a, *vs[:81]),
                    grido(b, *vs[81:]),
                    infd(vs, mkrange(1, 9)),
                    eq(a, b),
                ),
            )

        result = run(1, lambda a, b: sudokuo(a, b))
        breakpoint()
