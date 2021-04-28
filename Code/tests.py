import script
from script import nfa

N = 20000
M = 1000
n = 20


def test_script():
    for m in range(1, 10):
        assert nfa(n, N, m, M) is not None
