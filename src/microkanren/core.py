import itertools
from collections.abc import Callable
from functools import reduce
from typing import (
    Any,
    ClassVar,
    NamedTuple,
    Self,
    SupportsIndex,
    cast,
    overload,
)

import immutables
from fastcons import cons

# We could rely on Var object identity, but this might make tests
# simpler.
_COUNTER = itertools.count()


class OccursError(Exception): ...


class Sentinel: ...


SENTINEL = Sentinel()


class Var:
    name: str
    _id: int
    __match_args__ = ("name",)

    def __init__(self, name: str | int, _id: int | None = None):
        self.name = str(name)
        self._id = next(_COUNTER) if _id is None else _id

    def __repr__(self) -> str:
        return f"Var({self.name})"

    def __str__(self) -> str:
        return f"?{self.name}"

    def __hash__(self):
        return hash((self.__class__, self._id))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._id == other._id


class Symbol:
    name: str
    _cache: ClassVar[dict[str, Self] | None] = None

    def __new__(cls, name: str) -> Self:
        if (cache := cls._cache) is None:
            cache = {}
            cls._cache = cache

        if name in cache:
            return cache[name]

        instance = super().__new__(cls)
        cache[name] = instance
        return instance

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.__class__, self.name))

    def __eq__(self, other):
        return other is self


type Substitution = immutables.Map[Var, Any]


def empty_sub() -> Substitution:
    return immutables.Map()


class State(NamedTuple):
    sub: Substitution


type TableCache = list


class SuspendedStream(NamedTuple):
    cache: TableCache

    # A suffix of the tabled goal's cached answers. Indicates which of the
    # primary call's answer terms this stream has already processed.
    answer_terms: list

    # Produces the remainder of the stream.
    thunk: Callable[[], "Stream"]


def ready_stream(ss: SuspendedStream) -> bool:
    """
    Does the cache contain new answer terms not yet consumed by the stream?
    """
    return ss.cache != ss.answer_terms


class WaitingStream(list[SuspendedStream]):
    @overload
    def __getitem__(self, key: SupportsIndex) -> SuspendedStream: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(super().__getitem__(key))
        else:
            return super().__getitem__(key)

    def __add__(self, value) -> "WaitingStream":
        if not isinstance(value, self.__class__):
            raise TypeError(
                "WaitingStream can only be concatenated with another WaitingStream instance"
            )
        return WaitingStream([*self, *value])

    def __repr__(self) -> str:
        return f"WaitingStream({super().__repr__()})"

    def __str__(self) -> str:
        return f"WaitingStream({super().__str__()})"


type EmptyStream = tuple[()]
type ReadyStream = tuple[State, "Stream"]
type ThunkStream = Callable[[], "Stream"]
type Stream = EmptyStream | ReadyStream | ThunkStream | WaitingStream
type Goal = Callable[[State], Stream]
type GoalConstructor = Callable[..., Goal]


def empty(stream: Stream):
    return stream == ()


def empty_state():
    return State(empty_sub())


def walk(candidate: Any, sub: Substitution) -> Any:
    while isinstance(candidate, Var):
        result = sub.get(candidate, SENTINEL)
        if result is SENTINEL:
            return candidate
        candidate = result
    return candidate


def deep_walk(candidate: Any, sub: Substitution) -> Any:
    """
    Like `walk', but reify elements of lists/tuples.
    """
    candidate = walk(candidate, sub)
    if isinstance(candidate, list | tuple):
        container = type(candidate)
        return container(deep_walk(x, sub) for x in candidate)
    elif isinstance(candidate, cons):
        return cons(deep_walk(candidate.head, sub), deep_walk(candidate.tail, sub))
    else:
        return candidate


def extend_substitution(
    x: Var, v: Any, sub: Substitution, occurs_check: bool = True
) -> Substitution:
    if occurs_check and occurs(x, v, sub):
        raise OccursError(f"occurs_check failed ({x=}, {v=}, {sub=})")
    return sub.set(x, v)


def occurs(x: Var, v: Any, sub: Substitution) -> bool:
    """
    Does `x' occur in `v' with regards to `sub'?
    """
    v = walk(v, sub)
    if isinstance(v, Var):
        return v == x
    elif isinstance(v, cons):
        return occurs(x, v.head, sub) or occurs(x, v.tail, sub)
    elif isinstance(v, list | tuple):
        return any(occurs(x, term, sub) for term in v)
    else:
        return False


def unit(state: State) -> ReadyStream:
    return (state, mzero())


def succeed(state: State) -> ReadyStream:
    return unit(state)


def mzero() -> EmptyStream:
    return ()


def eq(u: Any, v: Any) -> Goal:
    def _eq(state: State) -> EmptyStream | ReadyStream:
        maybe_sub: Substitution | Sentinel = unify(u, v, state.sub)
        if isinstance(maybe_sub, Sentinel):
            return mzero()
        return unit(State(maybe_sub))

    return _eq


def unify(u: Any, v: Any, s: Substitution) -> Substitution | Sentinel:
    """
    Unify u and v in the Substitution s.
    """
    match walk(u, s), walk(v, s):
        case Var(_) as x, Var(_) as y if x is y:
            return s
        case Var(_) as x, y:
            return extend_substitution(x, y, s)
        case x, Var(_) as y:
            return extend_substitution(y, x, s)
        case cons(x, xs), cons(y, ys):
            s1 = unify(x, y, s)
            return SENTINEL if isinstance(s1, Sentinel) else unify(xs, ys, s1)
        case (x, *xs) as m, (y, *ys) as n if len(m) == len(n) and type(m) is type(n):
            result = unify(x, y, s)
            if isinstance(result, Sentinel):
                return SENTINEL
            return unify(xs, ys, result)
        case x, y if x == y:
            return s
        case _:
            return SENTINEL


def call_fresh(f: Callable[[Var], Goal]) -> Goal:
    """
    Return a goal that calls `f' (a goal constructor) with a fresh logic variable.
    """

    def _goal(state: State) -> Stream:
        # If the goal constructor was tabled, get the actual goal
        # constructor function so the var name is relevant.
        actual_gc = getattr(f, "__wrapped_goal_constructor__", f)
        return f(Var(actual_gc.__code__.co_varnames[0]))(state)

    return _goal


def bind(stream: Stream, g: Goal) -> Stream:
    if empty(stream):
        return mzero()
    elif callable(stream):
        return lambda: bind(stream(), g)
    elif isinstance(stream, WaitingStream):
        return w_check(
            stream,
            lambda x: lambda: bind(x(), g),
            lambda: WaitingStream(
                SuspendedStream(
                    x.cache,
                    x.answer_terms,
                    lambda x=x: bind(x.thunk(), g),
                )
                for x in stream
            ),
        )
    else:
        head, tail = cast(tuple[State, Stream], stream)
        return mplus(g(head), bind(tail, g))


def mplus(left: Stream, right: Stream) -> Stream:
    if empty(left):
        return right
    elif callable(left):
        return lambda: mplus(right, left())
    elif isinstance(left, WaitingStream):
        return w_check(
            left,
            lambda x: lambda: mplus(right, x),
            lambda: right + left
            if isinstance(right, WaitingStream)
            else mplus(right, lambda: left),
        )
    else:
        head, tail = cast(tuple[State, Stream], left)
        return (head, mplus(tail, right))


def disj(g1: Goal, g2: Goal) -> Goal:
    def _disj(state: State):
        return mplus(g1(state), g2(state))

    return _disj


def conj(g1: Goal, g2: Goal) -> Goal:
    def _conj(state: State):
        return bind(g1(state), g2)

    return _conj


### Reification


def pull(x):
    while callable(x):
        x = x()
    return x


def raise_(exc):
    raise exc


def take(n: int, stream: Stream):
    if n == 0:
        return []
    s = pull(stream)
    if empty(s):
        return []
    elif isinstance(s, WaitingStream):
        return take(
            n,
            w_check(
                s,
                lambda x: x,
                lambda: mzero(),
            ),
        )
    else:
        a, d = cast(ReadyStream, s)
        return [a, *take(n - 1, d)]


def reify_symbol(i: int) -> Symbol:
    return Symbol(f"_.{i}")


def make_reify(representation):
    def reify(v: Any, s: Substitution):
        v = deep_walk(v, s)
        return deep_walk(v, reify_sub(representation, v, empty_sub()))

    return reify


def reify_sub(representation: Callable, v: Any, sub: Substitution) -> Substitution:
    v = walk(v, sub)
    if isinstance(v, Var):
        return extend_substitution(v, representation(len(sub)), sub, occurs_check=False)
    elif isinstance(v, list | tuple):
        return reduce(lambda s, x: reify_sub(representation, x, s), v, sub)
    else:
        return sub


# Reify unbound logic variables as Symbols.
reify = make_reify(reify_symbol)

# Reify unbound logic variables as fresh logic variables.
reify_var = make_reify(Var)

# Apparently like Prolog's copy_term/2.
reify_tabled_var = make_reify(lambda i: Var(str(i)))


### Tabling


class Table(dict[tuple, TableCache]): ...


def tabled(gc: GoalConstructor) -> GoalConstructor:
    table = Table()

    def tabled_gc(*args):
        def tabled_goal(state: State) -> Stream:
            # Reify `args' in the current substitution, use this as
            # the cache key for a master call.
            key = reify(args, state.sub)
            if key not in table:
                table[key] = []
                return conj(
                    gc(*args),
                    primary_tabled_call(args, table[key]),
                )(state)

            # reuse_tabled_results gets a pointer to the cache, not a
            # copy, so the suspended streams have access to newly
            # cached results.
            return reuse_tabled_results(args, table[key], state)

        return tabled_goal

    tabled_gc._table = table
    tabled_gc.__wrapped_goal_constructor__ = gc
    return tabled_gc


def primary_tabled_call(args: tuple, cache: TableCache) -> Goal:
    def _goal(state: State) -> Stream:
        (sub,) = state
        reified_args = reify(args, sub)

        # Check if the result is alpha-equivalent to any previous cached result.
        if any(reified_args == reify(result, sub) for result in cache):
            # If so, contribute no state.
            return mzero()

        # Otherwise, we have a new state to cache and contribute to the result.
        cache.append(reify_tabled_var(args, sub))
        return unit(state)

    return _goal


def alpha_equivalent(x: Any, y: Any, s: Substitution) -> bool:
    return reify(x, s) == reify(y, s)


def reuse_tabled_results(args: tuple, cache: TableCache, state: State) -> Stream:
    # Fix is called at the beginning of the reuse call, with the whole cache.
    # Fix is called as a SS's thunk in w_check, in the success continuation.
    def fix(start, end) -> Stream:
        def loop(cached_results):
            if cached_results == end:
                # This will run on the first iteration of `loop'.
                return WaitingStream(
                    # TODO: Should `cache' be a copy of the tabled
                    # cache, or a pointer to it?
                    [SuspendedStream(cache, start, lambda: fix(cache, start))]
                )
            else:
                head, *tail = cached_results
                (sub,) = state

                # Produce a new state that is the result of unifying
                # the secondary call's args with the first cached
                # result.
                next_state = State(
                    subunify(args, reify_tabled_var(head, sub), sub),
                )

                # Concat the new state with the result of unifying the
                # secondary call's args with the rest of the cached
                # results.
                return mplus(
                    unit(next_state),
                    lambda: loop(tail),
                )

        return loop(start)

    return fix(cache, [])


def subunify(args, cached_result, sub: Substitution) -> Substitution:
    args = walk(args, sub)
    if args == cached_result:
        return sub
    elif isinstance(args, Var):
        return extend_substitution(args, cached_result, sub, occurs_check=False)
    elif isinstance(args, list | tuple):
        a, *d = args
        b, *e = cached_result
        return subunify(d, e, subunify(a, b, sub))
    else:
        return sub


def w_check(
    w: WaitingStream, sk: Callable[[Any], Stream], fk: Callable[[], Stream]
) -> Stream:
    # Find the first suspended stream in `w' whose cache contains new
    # answer terms.

    def loop(w: WaitingStream, a: WaitingStream) -> Stream:
        if not w:
            return fk()
        elif ready_stream(w[0]):
            # The first suspended is can contribute results. Invoke
            # its thunk, followed by any remaining suspended streams.
            head, *tail = w
            _, _, thunk = head
            rest_suspended_streams = WaitingStream(a[::-1] + tail)
            return sk(
                lambda: thunk()
                if not w
                else mplus(thunk(), lambda: rest_suspended_streams)
            )
        else:
            return loop(w[1:], WaitingStream([w[0], *a]))

    return loop(w, WaitingStream())
