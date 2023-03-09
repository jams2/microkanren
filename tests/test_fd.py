import pytest

from microkanren import domfd, eq, ltefd, make_domain, neq, plusfd, rangefd, run_all


class TestFdConstraints:
    @pytest.mark.parametrize("domain", [make_domain(1, 2, 3), make_domain(5, 7, 9)])
    def test_domfd(self, domain):
        result = run_all(lambda x: domfd(x, domain))
        assert set(result) == domain

    @pytest.mark.parametrize(
        "a,b,intersection",
        [
            (make_domain(1, 2), make_domain(1, 2), make_domain(1, 2)),
            (make_domain(1, 2, 3), make_domain(2, 3, 4), make_domain(2, 3)),
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
        "a,b,expected",
        [
            (make_domain(1, 2, 3, 4), make_domain(2, 3), make_domain(1, 2, 3)),
            (make_domain(4, 5), make_domain(1, 2), make_domain()),
            (make_domain(3, 4), make_domain(2, 3, 4, 5), make_domain(3, 4)),
            (make_domain(1, 2, 3, 4), make_domain(1, 2, 3, 4), make_domain(1, 2, 3, 4)),
            (make_domain(1, 2, 3), make_domain(1, 2), make_domain(1, 2)),
        ],
    )
    def test_ltefd(self, a, b, expected):
        result = run_all(lambda x, y: domfd(x, a) & domfd(y, b) & ltefd(x, y))
        assert set(result) == expected

    @pytest.mark.xfail(reason="not implemented")
    def test_neq_with_domfd(self):
        """
        If neq(x, n), then n cannot be in the domain of x.
        """
        result = run_all(lambda x: domfd(x, make_domain(1, 2, 3)) & neq((x, 2)))
        assert set(result) == make_domain(1, 3)

    @pytest.mark.xfail(reason="not implemented")
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
        assert result == [2]

    def test_plusfd(self):
        result = run_all(
            lambda q, x, y, z: domfd(x, rangefd(1, 3))
            & domfd(y, rangefd(1, 3))
            & domfd(z, rangefd(1, 3))
            & plusfd(x, y, z)
            & eq(q, (x, y, z))
        )
        assert set(result) == {(1, 1, 2), (1, 2, 3), (2, 1, 3)}

    def test_plusfd_with_ltefd(self):
        result = run_all(
            lambda q, x, y, z: domfd(x, rangefd(1, 3))
            & domfd(y, rangefd(1, 2))
            & domfd(z, rangefd(1, 4))
            & plusfd(x, y, z)
            & ltefd(x, y)
            & eq(q, (x, y, z))
        )
        assert set(result) == {(1, 1, 2), (1, 2, 3), (2, 2, 4)}
