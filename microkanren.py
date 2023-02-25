"""
TODO: maybe add a const_string or symbol type to avoid overhead for strings that won't
be combined/mutated
"""

from dataclasses import dataclass
from functools import reduce, update_wrapper, wraps
from typing import Union, Tuple, TypeAlias, Any, Callable, Literal


@dataclass(slots=True)
class Nil:
    pass


@dataclass(slots=True)
class Char:
    i: int

    def __repr__(self):
        return f"Char({self.i})"


@dataclass(slots=True, repr=False)
class Cons:
    head: Any
    tail: Any = Nil()
    is_proper: bool = False

    def __init__(self, head, tail=Nil()):
        self.head = head
        self.tail = tail
        match tail:
            case Nil():
                self.is_proper = True
            case Cons(_, _) as d if d.is_proper:
                self.is_proper = True
            case _:
                self.is_proper = False

    def __iter__(self):
        yield self.head
        yield self.tail

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head}, {self.tail})"

    @classmethod
    def from_python(cls, ls):
        cons = Nil()
        for x in reversed(ls):
            cons = Cons(x, cons)
        return cons

    def _to_str(self):
        return "".join(chr(char.i) for char in self._to_list())

    def _to_list(self):
        ls = [to_python(self.head)]
        cons = self.tail
        while cons != Nil():
            ls.append(to_python(cons.head))
            cons = cons.tail
        return ls

    def to_python(self):
        if not self.is_proper:
            raise ValueError("Can't convert improper cons list to Python")
        if isinstance(self.head, Char):
            return self._to_str()
        return self._to_list()


def string(cs):
    return Cons.from_python([Char(ord(c)) for c in cs])


def from_python(obj):
    if isinstance(obj, list):
        return Cons.from_python(list(map(from_python, obj)))
    elif isinstance(obj, dict):
        return Cons.from_python(
            [(from_python(k), from_python(v)) for k, v in obj.items()]
        )
    elif isinstance(obj, tuple):
        return tuple(from_python(x) for x in obj)
    elif isinstance(obj, str):
        return string(obj)
    return obj


def to_python(obj):
    if obj == Nil():
        return []
    elif isinstance(obj, Cons):
        if obj.is_proper:
            return obj.to_python()
        return Cons(to_python(obj.head), to_python(obj.tail))
    elif isinstance(obj, tuple):
        return tuple(to_python(x) for x in obj)
    else:
        return obj


@dataclass(frozen=True, slots=True)
class Var:
    i: int


@dataclass(frozen=True, slots=True)
class ReifiedVar:
    i: int

    def __repr__(self):
        return f"_.{self.i}"


Value: TypeAlias = Union[Var, int, str, bool, Tuple["Value", ...], Cons, Nil]
Substitution: TypeAlias = list[tuple[Var, Value]]
ConstraintStore: TypeAlias = list[list[tuple[Var, Value]]]


@dataclass(slots=True)
class State:
    sub: Substitution
    constraints: ConstraintStore
    var_count: int

    @classmethod
    def empty(cls):
        return cls([], [], 0)


Stream: TypeAlias = Tuple[()] | Callable[[], "Stream"] | Tuple[State, "Stream"]
Goal: TypeAlias = Callable[[State], Stream]

mzero = ()


def unit(state: State) -> Stream:
    return (state, mzero)


def mplus(s1: Stream, s2: Stream) -> Stream:
    match s1:
        case ():
            return s2
        case f if callable(s1):
            return lambda: mplus(f(), s2)
        case (t, u):
            return (t, mplus(s2, u))


def bind(stream: Stream, g: Goal) -> Stream:
    match stream:
        case ():
            return mzero
        case f if callable(f):
            return lambda: bind(f(), g)
        case (s1, s2):
            return mplus(g(s1), bind(s2, g))


def walk(u: Value, s: Substitution) -> Value:
    if isinstance(u, Var):
        match find(u, s):
            case (_, value):
                return walk(value, s)
            case _:
                return u
    return u


def find(x: Var, s: Substitution) -> tuple[Var, Value] | None:
    for (var, value) in s:
        if var == x:
            return (var, value)
    return None


def extend_substitution(x: Var, v: Value, s: Substitution) -> Substitution:
    return [(x, v), *s]


def occurs(x, v, s):
    v = walk(v, s)
    match v:
        case Var(_) as var:
            return var == x
        case (a, d):
            return occurs(x, a, s) or occurs(x, d, s)
        case _:
            return False


def unify(u: Value, v: Value, s: Substitution) -> Substitution | Literal[None]:
    match walk(u, s), walk(v, s):
        case (Var(_) as vi, Var(_) as vj) if vi == vj:
            return s
        case (Var(_) as var, val):
            return extend_substitution(var, val, s)
        case (val, Var(_) as var):
            return extend_substitution(var, val, s)
        case Cons(x, xs), Cons(y, ys):
            s1 = unify(x, y, s)
            return unify(xs, ys, s1) if s1 is not None else None
        case tuple(xs), tuple(ys) if len(xs) == len(ys):
            for x, y in zip(xs, ys):
                s = unify(x, y, s)
                if s is None:
                    return None
            return s
        case x, y if x == y:
            return s
        case _:
            return None


class goal:
    def __init__(self, goal_func: Goal):
        update_wrapper(self, goal_func)
        self.f = goal_func

    def __call__(self, state):
        return self.f(state)

    def __or__(self, other):
        return _disj(self.f, other.f)

    def __and__(self, other):
        return _conj(self.f, other.f)


def succeed() -> Goal:
    def _succeed(state):
        return eq(True, True)(state)

    return goal(_succeed)


def fail() -> Goal:
    def _fail(state):
        return eq(False, True)(state)

    return goal(_fail)


def delay(g: Goal) -> Goal:
    return goal(lambda state: lambda: g(state))


def disj(g: Goal, *goals: Goal) -> Goal:
    if goals == ():
        return delay(g)
    g2, *rest = goals
    return _disj(delay(g), disj(g2, *rest))


def _disj(g1: Goal, g2: Goal) -> Goal:
    def __disj(state: State) -> Stream:
        return mplus(g1(state), g2(state))

    return goal(__disj)


def conj(g: Goal, *goals: Goal) -> Goal:
    if goals == ():
        return delay(g)
    g2, *rest = goals
    return _conj(delay(g), conj(g2, *rest))


def _conj(g1: Goal, g2: Goal) -> Goal:
    def __conj(state: State) -> Stream:
        return bind(g1(state), g2)

    return goal(__conj)


def eq(u: Value, v: Value) -> Goal:
    def _eq(state: State) -> Stream:
        return verify_eq(unify(u, v, state.sub), state)

    return goal(_eq)


def verify_eq(new_sub: Substitution, state: State) -> Stream:
    if new_sub is None:
        # Unification failed
        return mzero
    elif new_sub == state.sub:
        # Unification succeeded without new associations
        return unit(state)

    # There are new associations, so we need to run the constraints
    remaining_constraints = verify_constraints(state.constraints, [], new_sub)
    if remaining_constraints is None:
        # A constraint was violated by the new association
        return mzero

    return unit(State(new_sub, remaining_constraints, state.var_count))


def verify_constraints(
    existing_constraints: ConstraintStore,
    verified_constraints: ConstraintStore,
    sub: Substitution,
) -> ConstraintStore | Literal[None]:
    if existing_constraints == []:
        return verified_constraints
    match unify_all(existing_constraints[0], sub):
        case None:
            # Unification failed, the constraint holds, discard it and continue
            return verify_constraints(
                existing_constraints[1:], verified_constraints, sub
            )
        case new_sub if new_sub == sub:
            # No new associations were made as a result of unification
            # therefore the the values were equal, violating constraints
            return None
        case new_sub:
            # New associations were added to the sub - constraints were simplified,
            # made more concrete, etc. Keep the simplified constraints and continue
            constraints = get_sub_prefix(new_sub, sub)
            return verify_constraints(
                constraints[1:], [constraints, *verified_constraints], sub
            )


def unify_all(constraint, sub: Substitution):
    match constraint:
        case []:
            return sub
        case [(a, d), *rest]:
            s = unify(a, d, sub)
            if s is not None:
                return unify_all(rest, s)
            return None
        case _:
            return None


def maybe_unify(
    pair: Tuple[Value, Value], sub: Substitution | Literal[None]
) -> Substitution | Literal[None]:
    if sub is None:
        return None
    u, v = pair
    return unify(u, v, sub)


def flip(f):
    @wraps(f)
    def _flipped(x, y):
        return f(y, x)

    return _flipped


def neq(pair, *pairs) -> Goal:
    def _neq(state: State) -> Stream:
        u, v = pair
        sub = reduce(flip(maybe_unify), pairs, unify(u, v, state.sub))
        return verify_neq(sub, state)

    return goal(_neq)


def verify_neq(new_sub: Substitution, state: State) -> Stream:
    if new_sub is None:
        # The unification failed: u and v cannot be unified, the disequality
        # constraint can never be violated
        return unit(state)
    elif new_sub == state.sub:
        # The sub has not been extended: u == v, violating the constraint
        return mzero
    else:
        # A new mapping was added to the constraint: the constraint has not yet
        # been violated, but may be in future
        constraint = get_sub_prefix(new_sub, state.sub)
        return unit(State(state.sub, [constraint, *state.constraints], state.var_count))


def get_sub_prefix(new_sub: Substitution, old_sub: Substitution) -> Substitution:
    if new_sub == old_sub:
        return []
    return [new_sub[0], *get_sub_prefix(new_sub[1:], old_sub)]


def conda(*cases) -> Goal:
    _cases = []
    for case in cases:
        if isinstance(case, (list, tuple)):
            _cases.append((case[0], succeed) if len(case) < 2 else case)
        else:
            _cases.append((case, succeed()))

    def _conda(state: State) -> Stream:
        return starfoldr(ifte, _cases, fail())(state)

    return goal(_conda)


def starfoldr(f, xs, initial):
    sentinel = object()
    accum = sentinel
    for args in reversed(xs):
        if accum is sentinel:
            _args = (*args, initial)
        else:
            _args = (*args, accum)
        accum = f(*_args)
    return accum


def ifte(g1, g2, g3=fail()) -> Goal:
    def _ifte(state: State) -> Stream:
        def ifte_loop(stream: Stream) -> Stream:
            match stream:
                case ():
                    return g3(state)
                case (_, _):
                    return bind(stream, g2)
                case _:
                    return lambda: ifte_loop(stream())

        return ifte_loop(g1(state))

    return goal(_ifte)


def fresh(fp: Callable) -> Goal:
    n = fp.__code__.co_argcount

    def _fresh(state: State) -> Stream:
        i = state.var_count
        vs = (Var(j) for j in range(i, i + n))
        return fp(*vs)(State(state.sub, state.constraints, i + n))

    return goal(_fresh)


def pull(s: Stream):
    if callable(s):
        return pull(s())
    return s


def take_all(s: Stream) -> list[State]:
    s1 = pull(s)
    if s1 == ():
        return []
    first, rest = s1
    return [first, *take_all(rest)]


def take(n, s: Stream) -> list[State]:
    if n == 0:
        return []
    s1 = pull(s)
    if s1 == ():
        return []
    first, rest = s1
    return [first, *take(n - 1, rest)]


def reify(states: list[State]):
    return [reify_state(s, 0) for s in states]


def reify_state(state: State, i: int = 0):
    v = walk_all(Var(i), state.sub)
    v = walk_all(v, reify_sub(v, []))
    return to_python(v)


def walk_all(v: Value, s: Substitution) -> Value:
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            return v
        case Cons(a, d):
            return Cons(walk_all(a, s), walk_all(d, s))
        case tuple(xs):
            return tuple(walk_all(x, s) for x in xs)
        case _:
            return v


def reify_sub(v: Value, s: Substitution):
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            return [Cons(v, ReifiedVar(len(s))), *s]
        case (a, d):
            return reify_sub(d, reify_sub(a, s))
        case _:
            return s


def call_with_empty_state(g: Goal) -> Stream:
    return g(State.empty())


def run(n: int, f_fresh_vars: Callable[[Var, ...], Goal]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = (Var(i) for i in range(n_vars))
    state = State([], [], n_vars)
    return reify(take(n, f_fresh_vars(*fresh_vars)(state)))


def run_all(f_fresh_vars: Callable[[Var, ...], Goal]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = (Var(i) for i in range(n_vars))
    state = State([], [], n_vars)
    return reify(take_all(f_fresh_vars(*fresh_vars)(state)))


def appendo(xs: Cons | Var, ys: Cons | Var, zs: Cons | Var) -> Goal:
    return disj(
        nullo(xs) & eq(ys, zs),
        fresh(
            lambda a, d, res: conj(
                conso(a, d, xs),
                conso(a, res, zs),
                appendo(d, ys, res),
            )
        ),
    )


def membero(x, xs):
    return fresh(
        lambda a, d: conj(
            conso(a, d, xs),
            disj(eq(a, x), membero(x, d)),
        )
    )


def nullo(x):
    return eq(x, Nil())


def notnullo(x):
    return neq(x, Nil())


def conso(a, d, ls):
    return eq(Cons(a, d), ls)


def caro(a, xs):
    return fresh(lambda d: conso(a, d, xs))


def cdro(d, xs):
    return fresh(lambda a: conso(a, d, xs))


def listo(xs):
    return nullo(xs) | fresh(lambda d: cdro(d, xs) & listo(d))


def inserto(x, ys, zs):
    def _inserto(state: State) -> Stream:
        return disj(
            appendo(Cons(x, Nil()), ys, zs),
            fresh(
                lambda a, d, res: conj(
                    eq((a, d), ys),
                    eq((a, res), zs),
                    inserto(x, d, res),
                ),
            ),
        )(state)

    return goal(_inserto)


def assoco(x, xs, y):
    return ifte(
        eq(xs, Nil()),
        fail(),
        fresh(
            lambda key, val, rest: disj(
                conj(
                    conso((key, val), rest, xs),
                    disj(
                        eq(key, x) & eq((key, val), y),
                        assoco(x, rest, y),
                    ),
                )
            ),
        ),
    )


def rembero(x, xs, out):
    def _rembero(state: State) -> Stream:
        return disj(
            nullo(xs) & nullo(out),
            fresh(
                lambda a, d, res: disj(
                    conso(x, d, xs) & eq(d, out),
                    conj(
                        neq(a, x),
                        conso(a, d, xs),
                        conso(a, res, out),
                        rembero(x, d, res),
                    ),
                )
            ),
        )(state)

    return goal(_rembero)


def joino(prefix, suffix, sep, out):
    return disj(
        nullo(prefix) & eq(suffix, out),
        nullo(suffix) & eq(prefix, out),
        fresh(
            lambda tmp: notnullo(prefix)
            & notnullo(suffix)
            & appendo(prefix, sep, tmp)
            & appendo(tmp, suffix, out)
        ),
    )
