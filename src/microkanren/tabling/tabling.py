"""
Implementation of µkanren with tabling, and without constraints.
"""
from collections import namedtuple
from functools import partial, wraps

import immutables
from fastcons import cons
from pyrsistent import PClass, field

from microkanren.core import (
    GoalProto,
    ReifiedVar,
    Stream,
    Substitution,
    Value,
    Var,
    empty_sub,
    unit,
    walk,
    walk_all,
)
from microkanren.goals import fail, succeed
from microkanren.utils import foldr


def occurs(x, v, s):
    match walk(v, s):
        case Var(_) as v:
            return v == x
        case (a, *d) | cons(a, d):
            return occurs(x, a, s) or occurs(x, d, s)
        case _:
            return False


def extend_substitution(x: "Var", v: Value, s: Substitution) -> Substitution:
    return None if occurs(x, v, s) else s.set(x, v)


# === Redefinitions === #


def ext_s_no_check(x: Var, v: Value, s: Substitution) -> Substitution:
    return s.set(x, v)


def unify(u: Value, v: Value, s: Substitution) -> Substitution | None:
    match walk(u, s), walk(v, s):
        case (Var(_) as vi, Var(_) as vj) if vi == vj:
            return s
        case (Var(_) as var, val):
            return extend_substitution(var, val, s)
        case (val, Var(_) as var):
            return extend_substitution(var, val, s)
        case cons(x, xs), cons(y, ys):
            s1 = unify(x, y, s)
            return unify(xs, ys, s1) if s1 is not None else None
        case (_, *_) as xs, (_, *_) as ys if (
            len(xs) == len(ys) and type(xs) == type(ys)
        ):
            for x, y in zip(xs, ys):  # NOQA: B905
                s = unify(x, y, s)
                if s is None:
                    return None
            return s
        case x, y if x == y:
            return s
        case _:
            return None


def mzero():
    return ()


def eq(u, v):
    def _eq(state):
        match unify(u, v, state.sub):
            case None:
                return mzero()
            case sub:
                return unit(state.set(sub=sub))

    return Goal(_eq)


def mplus(s1, s2):
    match s1:
        case ():
            return lambda: s2
        case f if callable(f):
            return lambda: mplus(s2, f())
        case tuple((head, tail)):
            return lambda: (head, mplus(s2, tail))
        case WaitingStream(_) as w:
            return w_check(
                w,
                lambda f: mplus(s2, f()),
                lambda: WaitingStream([*w_, *w])
                if isinstance((w_ := s2), WaitingStream)
                else mplus(w_, lambda: w),
            )
        case _:
            raise ValueError("Invalid stream")


def bind(stream, g):
    match stream:
        case ():
            return mzero()
        case f if callable(f):
            return lambda: bind(f(), g)
        case tuple((s1, s2)):
            return lambda: mplus(g(s1), bind(s2, g))
        case WaitingStream(_) as w:
            return w_check(
                w,
                lambda f: bind(f(), g),
                lambda: WaitingStream(
                    (
                        lambda ss: SuspendedStream(
                            ss.cache,
                            ss.answers,
                            lambda: bind(ss.f(), g),
                        )
                    )(ss)
                    for ss in w
                ),
            )
        case _:
            raise ValueError("Invalid stream")


def disj(*goals: GoalProto) -> GoalProto:
    return foldr(_disj, fail(), goals)


def _disj(g1: GoalProto, g2: GoalProto) -> GoalProto:
    def __disj(state: State) -> Stream:
        return mplus(g1(state), g2(state))

    return Goal(__disj)


def conj(*goals: GoalProto) -> GoalProto:
    return foldr(_conj, succeed(), goals)


def _conj(g1: GoalProto, g2: GoalProto) -> GoalProto:
    def __conj(state: State) -> Stream:
        return bind(g1(state), g2)

    return Goal(__conj)


class Goal:
    def __init__(self, goal):
        self.goal = goal

    def __call__(self, state):
        # Inverse η-delay happens here
        return lambda: self.goal(state)

    def __or__(self, other):
        return disj(self, other)

    def __and__(self, other):
        return conj(self, other)


# === Reification === #


def make_reify(representation):
    def _reify(v, state: "State"):
        v = walk_all(v, state.sub)
        next_state, sub = reify_sub(state, representation, v, empty_sub())
        return next_state, walk_all(v, sub)

    return _reify


def reify_sub(
    state: "State", representation, v: Value, s: Substitution
) -> ("State", Substitution):
    # Allows us to control how fresh Vars are reified
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            next_state, val = representation(state, s)
            return next_state, ext_s_no_check(v, val, s)
        case (a, *d) | cons(a, d):
            next_state, next_sub = reify_sub(state, representation, a, s)
            return reify_sub(next_state, representation, d, next_sub)
        case _:
            return state, s


reify = make_reify(ReifiedVar.with_state)
reify_var = make_reify(Var.with_state)


# === Datatypes === #


class State(PClass):
    sub = field(mandatory=True)
    var_count = field(mandatory=True)
    tables = field(mandatory=True)


def empty_table_store():
    return immutables.Map()


def empty_state():
    return State(sub=empty_sub(), var_count=0, tables=empty_table_store())


def empty_table():
    return immutables.Map()


Cache = namedtuple("Cache", "answers")


def empty_cache():
    return Cache([])


SuspendedStream = namedtuple("SuspendedStream", ("cache", "answers", "f"))


class WaitingStream(list[SuspendedStream]):
    pass


def ss_ready(ss: SuspendedStream):
    return ss.cache.answers != ss.answers


def w_check(w: WaitingStream, sk, fk):
    def _loop(w, a):
        match w:
            case WaitingStream([]):
                return fk()
            case first, *rest if ss_ready(first):
                return sk(
                    lambda: first.f()
                    if not (_w := WaitingStream([*a, *rest]))
                    else mplus(first.f(), lambda: _w)
                )
            case first, *rest:
                return _loop(WaitingStream(rest), WaitingStream([first, *a]))
            case _:
                raise ValueError("Invalid waiting stream")

    return _loop(w, WaitingStream([]))


# === Goal constructors === #


def init_table(g):
    g._table = {}


def tabled(g):
    init_table(g)

    @wraps(g)
    def wrapper(*args):
        def _g(state):
            next_state, key = reify(args, state)
            if cached := g._table.get(key):
                return reuse(args, cached, next_state)
            cache = empty_cache()
            g._table[key] = cache
            return conj(g(*args), master(args, cache))(next_state)

        return Goal(_g)

    return wrapper


def master(args, cache: Cache):
    def _master(state: State):
        equivalent_call = partial(alpha_equiv, args, state=state)
        if any(equivalent_call(answer_set) for answer_set in cache.answers):
            # The goal that triggered the master call has already been
            # called with `args', so fail.
            return mzero()
        next_state, answers = reify_var(args, state)
        cache.answers.append(answers)
        return unit(next_state)

    return Goal(_master)


def alpha_equiv(x, y, state: State):
    _, x = reify(x, state)
    _, y = reify(y, state)
    return x == y


def reuse(args, cache: Cache, state: State):
    def fix(start, end):
        def loop(answers):
            if answers == end:
                return WaitingStream(
                    [SuspendedStream(cache, start, lambda: fix(cache.answers, start))]
                )
            # Problematic call starts here
            next_state, reified_answers = reify_var(answers[0], state)
            next_sub = subunify(args, reified_answers, next_state.sub)
            return (
                next_state.set(sub=next_sub),
                lambda: loop(answers[1:]),
            )

        return loop(start)

    return fix(cache.answers, [])


def subunify(arg, ans, sub: Substitution):
    match walk(arg, sub):
        case _arg if _arg == ans:
            return sub
        case Var(_) as v:
            return ext_s_no_check(v, ans, sub)
        case cons(a, d):
            return subunify(d, ans.tail, subunify(a, ans.head, sub))
        case first, *rest:
            # args from a constructor call are a tuple
            return subunify(rest, ans[1:], subunify(first, ans[0], sub))
        case _:
            return sub


# === Interfaces === #


def run(n, f_fresh_vars):
    return _run(f_fresh_vars, partial(take, n), reify)


def pull(s):
    while callable(s):
        s = s()
    return s


def take(n, stream):
    if n == 0 or stream == ():
        return []
    match stream:
        case f if callable(f):
            return take(n, f())
        case WaitingStream(_) as w:
            return w_check(w, lambda f: take(n, f()), lambda: [])
        case first, rest:
            return [first, *take(n - 1, rest)]
        case _:
            raise ValueError


def _run(f_fresh_vars, take, reify):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = empty_state().set(var_count=n_vars)

    # We set up `q' and associate it with `fresh_vars' as the first goal
    # so all of the top-level vars requested by the user are reified wrt
    # the same substitution, giving accurate variable "numbers". Without
    # this, we may get incorrect multiple appearances of `_.0', for
    # instance.
    q = Var(-1)
    goal = conj(
        eq(q, fresh_vars if len(fresh_vars) > 1 else fresh_vars[0]),
        f_fresh_vars(*fresh_vars),
    )
    result = take(goal(state))
    return [reify(q, state)[1] for state in result]


### Testing

from microkanren import fresh


def _(*xs):
    return cons.from_xs(xs)


def consᵒ(a, d, pair):
    return eq(cons(a, d), pair)


@tabled
def term(start, rest):
    return disj(
        consᵒ("a", rest, start),
        fresh(
            lambda _1, _2: conj(
                term(start, _1),
                consᵒ("+", _2, _1),
                consᵒ("a", rest, _2),
            )
        ),
    )


### PROBLEM! cached answers have variables that may be out of sequence with the state in a run that's consuming the cache!


class _1:
    def __and__(self, other):
        return fresh(lambda x: conj(x, other))
