from itertools import filterfalse, tee

from fastcons import cons


def identity(x):
    return x


def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filter(pred, t1), filterfalse(pred, t2)


def foldr(f, b, xs):
    if not xs:
        return b
    x, *xs = xs
    return f(x, foldr(f, b, xs))


def _(*xs):
    return cons.from_xs(xs)
