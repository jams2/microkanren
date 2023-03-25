from itertools import permutations
from math import sqrt

import pytest

from microkanren import (
    alldifffd,
    compose_constraints,
    conj,
    default_enforce_constraints,
    default_process_prefix,
    domfd,
    enforce_constraints_fd,
    enforce_constraints_neq,
    eq,
    freshn,
    infd,
    ltefd,
    make_domain,
    mkrange,
    neq,
    neqfd,
    plusfd,
    process_prefix_fd,
    process_prefix_neq,
    run,
    run_all,
    set_enforce_constraints,
    set_process_prefix,
)


@pytest.fixture(autouse=True, scope="module")
def setup_process_prefix():
    def process_prefix(prefix, constraints):
        return compose_constraints(
            process_prefix_neq(prefix, constraints),
            process_prefix_fd(prefix, constraints),
        )

    yield set_process_prefix(process_prefix)
    set_process_prefix(default_process_prefix)


@pytest.fixture(autouse=True, scope="module")
def setup_enforce_constraints():
    def enforce_constraints(var):
        return conj(
            enforce_constraints_neq(var),
            enforce_constraints_fd(var),
        )

    yield set_enforce_constraints(enforce_constraints)
    set_enforce_constraints(default_enforce_constraints)


class TestFdConstraints:
    @pytest.mark.parametrize("domain", [make_domain(1, 2, 3), make_domain(5, 7, 9)])
    def test_domfd(self, domain):
        result = run_all(lambda x: domfd(x, domain))
        assert set(result) == set(domain)

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
        assert {x[0] for x in result} == expected_x
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
        assert {x[0] for x in result} == expected
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
    def test_sudoku(self):
        def sudokuo(a, b, size):
            def grido(grid, vs):
                return eq(
                    grid,
                    [
                        tuple(vs[j] for j in range(i, i + size))
                        for i in range(0, size * size, size)
                    ],
                )

            def blocko(vs, block_size):
                blocks = [
                    tuple(
                        vs[segment + block + row + i]
                        for row in range(0, size * block_size, size)
                        for i in range(block_size)
                    )
                    for segment in range(0, size * size, size * block_size)
                    for block in range(0, size, block_size)
                ]
                return conj(
                    *(alldifffd(*block) for block in blocks),
                )

            def rowo(vs):
                rows = [
                    tuple(vs[row + i] for i in range(size))
                    for row in range(0, size * size, size)
                ]
                return conj(*(alldifffd(*row) for row in rows))

            block_size = int(sqrt(size))

            return freshn(
                size * size,
                lambda *vs: conj(
                    grido(a, vs),
                    infd(vs, mkrange(1, size)),
                    rowo(vs),
                    blocko(vs, block_size),
                    eq(a, b),
                ),
            )

        result = run(1, lambda a, b: sudokuo(a, b, 4))
        assert result != []
