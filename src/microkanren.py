"""
TODO: maybe add a const_string or symbol type to avoid overhead for strings that won't
be combined/mutated

TODO: FD constraints could probably be stored as an upper and lower bound instead of a
set of the entire range
"""

from dataclasses import dataclass
from functools import reduce, update_wrapper, wraps
from typing import Union, Tuple, TypeAlias, Any, Callable, Optional, TypeVar


@dataclass(slots=True)
class nil:
    pass


@dataclass(slots=True)
class Char:
    i: int

    def __repr__(self):
        return f"Char({self.i})"


@dataclass(slots=True, repr=False)
class cons:
    head: Any
    tail: Any = nil()
    is_proper: bool = False

    def __init__(self, head, tail=nil()):
        self.head = head
        self.tail = tail
        match tail:
            case nil():
                self.is_proper = True
            case cons(_, _) as d if d.is_proper:
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
        accum = nil()
        for x in reversed(ls):
            accum = cons(x, accum)
        return accum

    def _to_str(self):
        return "".join(chr(char.i) for char in self._to_list())

    def _to_list(self):
        ls = [to_python(self.head)]
        cons = self.tail
        while cons != nil():
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
    return cons.from_python([Char(ord(c)) for c in cs])


def from_python(obj):
    if isinstance(obj, list):
        return cons.from_python(list(map(from_python, obj)))
    elif isinstance(obj, dict):
        return cons.from_python(
            [(from_python(k), from_python(v)) for k, v in obj.items()]
        )
    elif isinstance(obj, tuple):
        return tuple(from_python(x) for x in obj)
    elif isinstance(obj, str):
        return string(obj)
    return obj


def to_python(obj):
    if obj == nil():
        return []
    elif isinstance(obj, cons):
        if obj.is_proper:
            return obj.to_python()
        return cons(to_python(obj.head), to_python(obj.tail))
    elif isinstance(obj, tuple):
        return tuple(to_python(x) for x in obj)
    else:
        return obj


@dataclass(slots=True)
class Var:
    i: int


@dataclass(slots=True)
class ReifiedVar:
    i: int

    def __repr__(self):
        return f"_.{self.i}"


Cons: TypeAlias = Union[cons, nil]
Value: TypeAlias = Union[Var, int, str, bool, Tuple["Value", ...], Cons]
Substitution: TypeAlias = list[tuple[Var, Value]]
NeqStore: TypeAlias = list[list[tuple[Var, Value]]]
DomainStore: TypeAlias = list[tuple[Value, set[int]]]
Constraint: TypeAlias = Callable[["State"], Optional["State"]]
ConstraintStore: TypeAlias = list[Constraint]


@dataclass(slots=True)
class State:
    sub: Substitution
    neqs: NeqStore
    domains: DomainStore
    fd_cs: ConstraintStore
    var_count: int

    @classmethod
    def empty(cls):
        return cls([], [], [], [], 0)

    def update(
        self, *, sub=None, neqs=None, domains=None, fd_cs=None, var_count=None
    ) -> "State":
        state = {
            "sub": self.sub if sub is None else sub,
            "neqs": self.neqs if neqs is None else neqs,
            "domains": self.domains if domains is None else domains,
            "fd_cs": self.fd_cs if fd_cs is None else fd_cs,
            "var_count": self.var_count if var_count is None else var_count,
        }
        return self.__class__(**state)

    def get_domain(self, x: Value) -> tuple[Value, set[int]] | None:
        return find(x, self.domains)

    def get_relevant_neqs(self, x: Var):
        return [c for c in self.neqs if find(x, c) is not None]


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


def extend_domain(x: Value, fd: set[int], d: DomainStore) -> DomainStore:
    return [(x, fd), *d]


def occurs(x, v, s):
    v = walk(v, s)
    match v:
        case Var(_) as var:
            return var == x
        case (a, d):
            return occurs(x, a, s) or occurs(x, d, s)
        case _:
            return False


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

    # There are new associations, so we need to run the disequality constraints
    remaining_neqs = run_disequality_constraints(state.neqs, [], new_sub)
    if remaining_neqs is None:
        # A constraint was violated by the new association
        return mzero

    state_1 = state.update(sub=new_sub, neqs=remaining_neqs)
    prefix = get_sub_prefix(new_sub, state.sub)
    state_2 = process_prefix_fd(prefix, state_1.fd_cs)(state_1)
    return mzero if state_2 is None else unit(state_2)


def run_disequality_constraints(
    existing_constraints: NeqStore,
    remaining_constraints: NeqStore,
    sub: Substitution,
) -> NeqStore | None:
    if existing_constraints == []:
        return remaining_constraints
    match unify_all(existing_constraints[0], sub):
        case None:
            # Unification failed, the constraint holds, discard it and continue
            return run_disequality_constraints(
                existing_constraints[1:], remaining_constraints, sub
            )
        case new_sub if new_sub == sub:
            # No new associations were made as a result of unification
            # therefore the the values were equal, violating constraints
            return None
        case new_sub:
            # New associations were added to the sub - constraints were simplified,
            # made more concrete, etc. Keep the simplified constraints and continue
            constraints = get_sub_prefix(new_sub, sub)
            return run_disequality_constraints(
                constraints[1:], [constraints, *remaining_constraints], sub
            )


def unify_all(
    constraint: list[tuple[Var, Value]], sub: Substitution
) -> Substitution | None:
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
    pair: Tuple[Value, Value], sub: Substitution | None
) -> Substitution | None:
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
        # If unification of any of the pairs fails, that pair can never be unified,
        # so the constraint will always hold
        # e.g. neq((x,1), (y,2)) in []: both unifications succeed, so the constraints
        # must be kept
        # e.g. neq((x,1), (y,2)) in [(x,2)]: the unification of (x,1) will fail,
        # meaning this unification will never succeed, so this constraint cannot be
        # violated
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
        return unit(state.update(neqs=[constraint, *state.neqs]))


@dataclass(slots=True)
class FdConstraint:
    func: Constraint
    operands: list[Value]

    def __call__(self, state: State) -> State | None:
        return self.func(state)


def any_relevant_vars(operands, values):
    match operands:
        case Var(_) as v:
            return v in values
        case [first, *rest]:
            return any_relevant_vars(first, values) or any_relevant_vars(rest, values)
        case _:
            return False


def infd(x: Value, domain: set[int]) -> Goal:
    def _infd(state: State) -> Stream:
        match process_domain(x, domain)(state):
            case None:
                return mzero
            case _ as s1:
                return unit(s1)
        return unit(state)

    return goal(_infd)


def process_domain(x: Value, domain: set[int]):
    def _process_domain(state):
        match walk(x, state.sub):
            case Var(_):
                # x is not associated with any concrete value, update its domain
                return update_var_domain(x, domain, state)
            case val if val in domain:
                # We already have a concrete value, check if it's in the domain
                return state
            case _:
                return None

    return _process_domain


def update_var_domain(x: Var, domain: set[int], state: State) -> State | None:
    match state.get_domain(x):
        case (_, fd):
            i = domain & fd
            if i:
                return resolve_storable_domain(i, x, state)
            else:
                return None
        case _:
            return resolve_storable_domain(domain, x, state)


def resolve_storable_domain(domain: set[int], x: Var, state: State) -> State | None:
    if len(domain) == 1:
        n = domain.copy().pop()
        next_state = state.update(sub=extend_substitution(x, n, state.sub))
        return run_constraints([x], state.fd_cs)(next_state)
    return state.update(domains=extend_domain(x, domain, state.domains))


def identity_constraint(state):
    return state


def compose_constraints(f, g):
    def _compose_constraints(state):
        maybe_state = f(state)
        return g(maybe_state) if maybe_state else None

    return _compose_constraints


def run_constraints(xs: list, constraints: ConstraintStore):
    match constraints:
        case []:
            return identity_constraint
        case [first, *rest]:
            if any_relevant_vars(first, xs):
                return compose_constraints(
                    remove_and_run(first),
                    run_constraints(xs, rest),
                )
            else:
                return run_constraints(xs, rest)


def remove_and_run(constraint: FdConstraint) -> Constraint:
    def _remove_and_run(state: State) -> State | None:
        if constraint in state.fd_cs:
            fd_cs = [x for x in state.fd_cs if x != constraint]
            return constraint(state.update(fd_cs=fd_cs))
        else:
            return state

    return _remove_and_run


def process_prefix_fd(
    prefix: Substitution, constraints: ConstraintStore
) -> Callable[[State], State | None]:
    if not prefix:
        return identity_constraint
    (x, v), *rest_prefix = prefix
    t = compose_constraints(
        run_constraints([x], constraints), process_prefix_fd(rest_prefix, constraints)
    )

    def _process_prefix_fd(state: State) -> State | None:
        pair = state.get_domain(x)
        if pair is not None:
            _, domain_x = pair
            return compose_constraints(process_domain(v, domain_x), t)(state)
        return t(state)

    return _process_prefix_fd


def enforce_constraints_fd(x: Var) -> Goal:
    def _enforce_constraints(state: State) -> Stream:
        bound_vars = [x[0] for x in state.domains]
        # verify_all_bound(state.fd_cs, bound_vars)
        return onceo(force_answer(bound_vars))(state)

    return conj(force_answer(x), _enforce_constraints)


def force_answer(x: Var | list[Var]) -> Goal:
    def _force_answer(state: State) -> Stream:
        match walk(x, state.sub):
            case Var(_) as var if (d := state.get_domain(var)) is not None:
                _, domain = d
                return map_sum(lambda val: eq(x, val), domain)(state)
            case [first, *rest]:
                return disj(force_answer(first), force_answer(rest))(state)
            case _:
                return succeed()(state)

    return _force_answer


A = TypeVar("A")


def map_sum(goal_constructor: Callable[[A], Goal], xs: list[A]) -> Goal:
    return reduce(lambda accum, x: disj(accum, goal_constructor(x)), xs, fail())


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


def onceo(g: Goal) -> Goal:
    def _onceo(state: State) -> Stream:
        def onceo_loop(stream: Stream) -> Stream:
            match stream:
                case ():
                    return mzero
                case (s1, _):
                    return unit(s1)
                case _:
                    return onceo_loop(stream())

        return onceo_loop(g(state))

    return _onceo


def fresh(fp: Callable) -> Goal:
    n = fp.__code__.co_argcount

    def _fresh(state: State) -> Stream:
        i = state.var_count
        vs = (Var(j) for j in range(i, i + n))
        return fp(*vs)(state.update(var_count=i + n))

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


def reify_state(state: State, i: int = 0) -> Value:
    v = walk_all(Var(i), state.sub)
    v = walk_all(v, reify_sub(v, []))
    return to_python(v)


def walk_all(v: Value, s: Substitution) -> Value:
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            return v
        case cons(a, d):
            return cons(walk_all(a, s), walk_all(d, s))
        case tuple(xs):
            return tuple(walk_all(x, s) for x in xs)
        case _:
            return v


def reify_sub(v: Value, s: Substitution) -> Substitution:
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            return [(v, ReifiedVar(len(s))), *s]
        case (a, d):
            return reify_sub(d, reify_sub(a, s))
        case _:
            return s


def call_with_empty_state(g: Goal) -> Stream:
    return g(State.empty())


def run(n: int, f_fresh_vars: Callable[[Var, ...], Goal]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = (Var(i) for i in range(n_vars))
    state = State.empty().update(var_count=n_vars)
    enforce_constraints = enforce_constraints_fd(Var(0))
    return reify(take(n, (f_fresh_vars(*fresh_vars) & enforce_constraints)(state)))


def run_all(f_fresh_vars: Callable[[Var, ...], Goal]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = (Var(i) for i in range(n_vars))
    state = State.empty().update(var_count=n_vars)
    enforce_constraints = enforce_constraints_fd(Var(0))
    return reify(take_all((f_fresh_vars(*fresh_vars) & enforce_constraints)(state)))


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
    return eq(x, nil())


def notnullo(x):
    return neq(x, nil())


def conso(a, d, ls):
    return eq(cons(a, d), ls)


def caro(a, xs):
    return fresh(lambda d: conso(a, d, xs))


def cdro(d, xs):
    return fresh(lambda a: conso(a, d, xs))


def listo(xs):
    return nullo(xs) | fresh(lambda d: cdro(d, xs) & listo(d))


def inserto(x, ys, zs):
    def _inserto(state: State) -> Stream:
        return disj(
            appendo(cons(x, nil()), ys, zs),
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
        eq(xs, nil()),
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
