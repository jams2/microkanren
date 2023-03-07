from microkanren import eq, run_all


class TestEq:
    def test_simple_eq(self):
        result = run_all(lambda x: eq(x, 1))
        assert result == [1]
