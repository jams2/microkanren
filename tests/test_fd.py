import pytest

from microkanren import domfd, eq, run_all


class TestFdConstraints:
    @pytest.mark.parametrize("domain", [{1, 2, 3}, {5, 7, 9}])
    def test_infd(self, domain):
        result = run_all(lambda x: domfd(x, domain))
        assert set(result) == domain

    @pytest.mark.parametrize(
        "a,b,intersection",
        [
            ({1, 2}, {1, 2}, {1, 2}),
            ({1, 2, 3}, {2, 3, 4}, {2, 3}),
            ({1, 2}, {3, 4}, set()),
        ],
    )
    def test_domain_intersection(self, a, b, intersection):
        result = run_all(lambda x: domfd(x, a) & domfd(x, b))
        assert set(result) == intersection

    def test_violated_infd_fails(self):
        result = run_all(lambda x: domfd(x, {1, 2, 3}) & eq(x, 4))
        assert result == []
