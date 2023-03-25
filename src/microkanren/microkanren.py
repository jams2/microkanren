"""
TODO: maybe add a const_string or symbol type to avoid overhead for strings that won't
be combined/mutated

TODO: can we improve memory efficiency of domains by storing min/max of intervals only?

TODO: maybe store constraints as an inverse mapping of operands to constraints for more
efficient access
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import reduce, update_wrapper, wraps
from typing import Any, Optional, Protocol, TypeAlias, TypeVar

from pyrsistent import PClass, field, pmap
from pyrsistent.typing import PMap

from microkanren.cons import Cons, cons, to_python
from microkanren.utils import identity, partition

NOT_FOUND = object()


@dataclass(slots=True, frozen=True)
class Var:
    i: int


@dataclass(slots=True, frozen=True)
class ReifiedVar:
    i: int

    def __repr__(self):
        return f"_.{self.i}"


Value: TypeAlias = Var | int | str | bool | tuple["Value", ...] | list["Value"] | Cons
Substitution: TypeAlias = PMap[Var, Value]
NeqStore: TypeAlias = list[list[tuple[Var, Value]]]
DomainStore: TypeAlias = PMap[Var, set[int]]
ConstraintFunction: TypeAlias = Callable[["State"], Optional["State"]]
ConstraintStore: TypeAlias = list["Constraint"]
StreamThunk: TypeAlias = Callable[[], "Stream"]


def empty_sub() -> Substitution:
    return pmap()


def empty_domain_store() -> DomainStore:
    return pmap()


def empty_constraint_store() -> ConstraintStore:
    return []


class State(PClass):
    sub = field(mandatory=True)
    domains = field(mandatory=True)
    constraints = field(mandatory=True)
    var_count = field(mandatory=True)

    @staticmethod
    def empty():
        return State(
            sub=empty_sub(),
            domains=empty_domain_store(),
            constraints=empty_constraint_store(),
            var_count=0,
        )

    def get_domain(self, x: Var) -> set[int] | None:
        return self.domains.get(x, None)


class ConstraintProto(Protocol):
    def __call__(self, *args: Any) -> ConstraintFunction:
        ...


@dataclass(slots=True)
class Constraint:
    func: ConstraintProto
    operands: list[Value]

    def __call__(self, state: State) -> State | None:
        return self.func(*self.operands)(state)


def make_domain(*values: int) -> set[int]:
    return set(values)


def mkrange(start: int, end: int) -> set[int]:
    return make_domain(*range(start, end + 1))


Stream: TypeAlias = tuple[()] | Callable[[], "Stream"] | tuple[State, "Stream"]


class GoalProto(Protocol):
    def __call__(self, state: State) -> StreamThunk:
        ...


class GoalConstructorProto(Protocol):
    def __call__(self, *args: Value) -> GoalProto:
        ...


class Goal:
    def __init__(self, goal: GoalProto):
        update_wrapper(self, goal)
        self.goal = goal

    def __call__(self, state: State) -> StreamThunk:
        return lambda: self.goal(state)

    def __or__(self, other):
        # Use disj and conj instead of _disj and _conj, as the former delay their goals
        return disj(self, other)

    def __and__(self, other):
        return conj(self, other)


def goal_from_constraint(constraint: ConstraintFunction) -> GoalProto:
    def _goal(state: State) -> StreamThunk:
        match constraint(state):
            case None:
                return lambda: mzero
            case _ as next_state:
                return lambda: unit(next_state)

    return Goal(_goal)


class InvalidStream(Exception):
    pass


mzero = ()


def unit(state: State) -> Stream:
    return (state, mzero)


def mplus(s1: Stream, s2: Stream) -> StreamThunk:
    match s1:
        case ():
            return lambda: s2
        case f if callable(f):
            return lambda: mplus(s2, f())
        case (head, tail):
            return lambda: (head, mplus(s2, tail))
        case _:
            raise InvalidStream


def bind(stream: Stream, g: GoalProto) -> StreamThunk:
    match stream:
        case ():
            return lambda: mzero
        case f if callable(f):
            return lambda: bind(f(), g)
        case (s1, s2):
            return lambda: mplus(g(s1), bind(s2, g))
        case _:
            raise InvalidStream


def walk(u: Value, s: Substitution) -> Value:
    if isinstance(u, Var):
        bound = s.get(u, NOT_FOUND)
        if bound is NOT_FOUND:
            return u
        return walk(bound, s)
    return u


def extend_substitution(x: Var, v: Value, s: Substitution) -> Substitution:
    return s.set(x, v)


def extend_domain_store(x: Var, fd: set[int], d: DomainStore) -> DomainStore:
    return d.set(x, fd)


def extend_constraint_store(
    constraint: Constraint, c: ConstraintStore
) -> ConstraintStore:
    return [constraint, *c]


def occurs(x, v, s):
    v = walk(v, s)
    match v:
        case Var(_) as var:
            return var == x
        case (a, d) | cons(a, d):
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
        case (_, *_) as xs, (_, *_) as ys if (
            len(xs) == len(ys) and type(xs) == type(ys)
        ):
            for x, y in zip(xs, ys):
                s = unify(x, y, s)
                if s is None:
                    return None
            return s
        case x, y if x == y:
            return s
        case _:
            return None


def succeed() -> GoalProto:
    def _succeed(state):
        return eq(True, True)(state)

    return Goal(_succeed)


def fail() -> GoalProto:
    def _fail(state):
        return eq(False, True)(state)

    return Goal(_fail)


def snooze(g: GoalConstructorProto, *args: Value) -> GoalProto:
    def delayed_goal(state) -> StreamThunk:
        return lambda: g(*args)(state)

    return Goal(delayed_goal)


def delay(g: GoalProto) -> GoalProto:
    return Goal(lambda state: lambda: g(state))


def disj(g: GoalProto, *goals: GoalProto) -> GoalProto:
    if goals == ():
        return delay(g)
    g2, *rest = goals
    return _disj(delay(g), disj(g2, *rest))


def _disj(g1: GoalProto, g2: GoalProto) -> GoalProto:
    def __disj(state: State) -> StreamThunk:
        return lambda: mplus(g1(state), g2(state))

    return Goal(__disj)


def conj(g: GoalProto, *goals: GoalProto) -> GoalProto:
    if goals == ():
        return delay(g)
    g2, *rest = goals
    return _conj(delay(g), conj(g2, *rest))


def _conj(g1: GoalProto, g2: GoalProto) -> GoalProto:
    def __conj(state: State) -> StreamThunk:
        return lambda: bind(g1(state), g2)

    return Goal(__conj)


def eq(u: Value, v: Value) -> GoalProto:
    def _eqc(state: State) -> State | None:
        new_sub = unify(u, v, state.sub)
        if new_sub is None:
            # Unification failed
            return None
        elif new_sub == state.sub:
            # Unification succeeded without new associations
            return state
        else:
            prefix = get_sub_prefix(new_sub, state.sub)
            return process_prefix(prefix, state.constraints)(state.set(sub=new_sub))

    return goal_from_constraint(_eqc)


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
    pair: tuple[Value, Value], sub: Substitution | None
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


def neq(*pairs) -> GoalProto:
    return goal_from_constraint(neqc(pairs))


def neqc(pairs: tuple[tuple[Value, Value], ...]) -> ConstraintFunction:
    def _neqc(state: State) -> State | None:
        (u, v), *rest = pairs
        new_sub = reduce(flip(maybe_unify), rest, unify(u, v, state.sub))
        if new_sub is None:
            return state
        elif new_sub == state.sub:
            return None
        prefix = get_sub_prefix(new_sub, state.sub)
        remaining_pairs = list(prefix.items())
        return state.set(
            constraints=extend_constraint_store(
                Constraint(neqc, [remaining_pairs]), state.constraints
            )
        )

    return _neqc


def any_relevant_vars(operands, values):
    match operands:
        case Var(_) as v:
            return v in values
        case [first, *rest]:
            return any_relevant_vars(first, values) or any_relevant_vars(rest, values)
        case _:
            return False


def domfd(x: Value, domain: set[int]) -> GoalProto:
    return goal_from_constraint(domfdc(x, domain))


def domfdc(x: Value, domain: set[int]) -> ConstraintFunction:
    def _domfdc(state: State) -> State | None:
        process_domain_goal = process_domain(x, domain)
        return process_domain_goal(state)

    return _domfdc


def infd(values: tuple[Value], domain, /) -> GoalProto:
    infdc = reduce(
        lambda c, v: compose_constraints(c, domfdc(v, domain)),
        values,
        identity,
    )
    return goal_from_constraint(infdc)


def ltefd(u: Value, v: Value) -> GoalProto:
    return goal_from_constraint(ltefdc(u, v))


def ltefdc(u: Value, v: Value) -> ConstraintFunction:
    def _ltefdc(state: State) -> State | None:
        _u = walk(u, state.sub)
        _v = walk(v, state.sub)
        dom_u = state.get_domain(_u) if isinstance(_u, Var) else make_domain(_u)
        dom_v = state.get_domain(_v) if isinstance(_v, Var) else make_domain(_v)
        next_state = state.set(
            constraints=extend_constraint_store(
                Constraint(ltefdc, [_u, _v]), state.constraints
            )
        )
        if dom_u and dom_v:
            max_v = max(dom_v)
            min_u = min(dom_u)
            return compose_constraints(
                process_domain(_u, make_domain(*(i for i in dom_u if i <= max_v))),
                process_domain(_v, make_domain(*(i for i in dom_v if i >= min_u))),
            )(next_state)
        return state

    return _ltefdc


def plusfd(u: Value, v: Value, w: Value):
    return goal_from_constraint(plusfdc(u, v, w))


def plusfdc(u: Value, v: Value, w: Value) -> ConstraintFunction:
    def _plusfdc(state: State) -> State | None:
        _u = walk(u, state.sub)
        _v = walk(v, state.sub)
        _w = walk(w, state.sub)
        dom_u = state.get_domain(_u) if isinstance(_u, Var) else make_domain(_u)
        dom_v = state.get_domain(_v) if isinstance(_v, Var) else make_domain(_v)
        dom_w = state.get_domain(_w) if isinstance(_w, Var) else make_domain(_w)
        next_state = state.set(
            constraints=extend_constraint_store(
                Constraint(plusfdc, [_u, _v, _w]), state.constraints
            )
        )
        if dom_u and dom_v and dom_w:
            min_u = min(dom_u)
            max_u = max(dom_u)
            min_v = min(dom_v)
            max_v = max(dom_v)
            min_w = min(dom_w)
            max_w = max(dom_w)
            return compose_constraints(
                process_domain(_w, mkrange(min_u + min_v, max_u + max_v)),
                compose_constraints(
                    process_domain(_u, mkrange(min_w - max_v, max_w - min_v)),
                    process_domain(_v, mkrange(min_w - max_u, max_w - min_u)),
                ),
            )(next_state)
        return state

    return _plusfdc


def neqfd(u: Value, v: Value) -> GoalProto:
    return goal_from_constraint(neqfdc(u, v))


def neqfdc(u: Value, v: Value) -> ConstraintFunction:
    def _neqfdc(state: State) -> State | None:
        _u = walk(u, state.sub)
        _v = walk(v, state.sub)
        dom_u = state.get_domain(_u) if isinstance(_u, Var) else make_domain(_u)
        dom_v = state.get_domain(_v) if isinstance(_v, Var) else make_domain(_v)
        if dom_u is None or dom_v is None:
            return state.set(
                constraints=extend_constraint_store(
                    Constraint(neqfdc, [_u, _v]), state.constraints
                )
            )
        elif len(dom_u) == 1 and len(dom_v) == 1 and dom_u == dom_v:
            return None
        elif dom_u.isdisjoint(dom_v):
            return state

        next_state = state.set(
            constraints=extend_constraint_store(
                Constraint(neqfdc, [_u, _v]), state.constraints
            )
        )
        if len(dom_u) == 1:
            return process_domain(_v, dom_v - dom_u)(next_state)
        elif len(dom_v) == 1:
            return process_domain(_u, dom_u - dom_v)(next_state)
        else:
            return next_state

    return _neqfdc


def alldifffd(*vs: Value) -> GoalProto:
    return goal_from_constraint(alldifffdc(*vs))


def alldifffdc(*vs: Value) -> ConstraintFunction:
    def _alldifffdc(state: State) -> State | None:
        unresolved, values = partition(lambda v: isinstance(v, Var), vs)
        unresolved = list(unresolved)
        values = list(values)
        values_domain = make_domain(*values)
        if len(values) == len(values_domain):
            return alldifffdc_resolve(unresolved, values_domain)(state)
        return None

    return _alldifffdc


def alldifffdc_resolve(unresolved: list[Var], values: set[Value]) -> ConstraintFunction:
    def _alldifffdc_resolve(state: State) -> State | None:
        nonlocal values
        values = values.copy()
        remains_unresolved = []
        for var in unresolved:
            v = walk(var, state.sub)
            if isinstance(v, Var):
                remains_unresolved.append(v)
            elif is_domain_member(v, values):
                return None
            else:
                values.add(v)

        next_state = state.set(
            constraints=extend_constraint_store(
                Constraint(alldifffdc_resolve, [remains_unresolved, values]),
                state.constraints,
            )
        )
        return exclude_from_domains(remains_unresolved, values)(next_state)

    return _alldifffdc_resolve


def exclude_from_domains(vs: list[Var], values: set[Value]) -> ConstraintFunction:
    """
    For each Var in vs, remove all values in values from its domain.
    """

    def _exclude_from_domains(state: State) -> State | None:
        with_domains = (
            (var, dom) for var in vs if (dom := state.get_domain(var)) is not None
        )
        constraint = reduce(
            compose_constraints,
            (process_domain(var, dom - values) for var, dom in with_domains),
            identity,
        )
        return constraint(state)

    return _exclude_from_domains


def is_domain_member(v: Value, dom: set[int]) -> bool:
    return isinstance(v, int) and v in dom


def process_domain(x: Value, domain: set[int]) -> ConstraintFunction:
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
        case fd if isinstance(fd, set):
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
        next_state = state.set(sub=extend_substitution(x, n, state.sub))
        return run_constraints([x], state.constraints)(next_state)
    return state.set(domains=extend_domain_store(x, domain, state.domains))


def bind_constraints(f, g) -> ConstraintFunction:
    def _bind_constraints(state: State) -> State | None:
        maybe_state = f(state)
        return g(maybe_state) if maybe_state is not None else None

    return _bind_constraints


def compose_constraints(f, g) -> ConstraintFunction:
    return bind_constraints(f, g)


def run_constraints(
    xs: list[Var] | set[Var], constraints: ConstraintStore
) -> ConstraintFunction:
    match constraints:
        case []:
            return identity
        case [first, *rest]:
            if any_relevant_vars(first.operands, xs):
                return compose_constraints(
                    remove_and_run(first),
                    run_constraints(xs, rest),
                )
            else:
                return run_constraints(xs, rest)
        case _:
            raise ValueError("Invalid constraint store")


def remove_and_run(constraint: Constraint) -> ConstraintFunction:
    def _remove_and_run(state: State) -> State | None:
        if constraint in state.constraints:
            constraints = [x for x in state.constraints if x != constraint]
            return constraint(state.set(constraints=constraints))
        else:
            return state

    return _remove_and_run


def process_prefix_neq(
    prefix: Substitution, constraints: ConstraintStore
) -> ConstraintFunction:
    prefix_vars = {x for x in prefix.keys() if isinstance(x, Var)}
    prefix_vars.update({x for x in prefix.values() if isinstance(x, Var)})
    return run_constraints(prefix_vars, constraints)


def enforce_constraints_neq(_: Var) -> GoalProto:
    return lambda state: lambda: unit(state)


def reify_constraints_neq(_: Var, __: Substitution) -> ConstraintFunction:
    def _reify_constraints_neq(state: State):
        return state

    return _reify_constraints_neq


def process_prefix_fd(
    prefix: Substitution, constraints: ConstraintStore
) -> ConstraintFunction:
    if not prefix:
        return identity
    (x, v), *_ = prefix.items()
    t = compose_constraints(
        run_constraints([x], constraints),
        process_prefix_fd(prefix.remove(x), constraints),
    )

    def _process_prefix_fd(state: State):
        domain_x = state.get_domain(x)
        if domain_x is not None:
            # We have a new association for x (as x is in prefix), and we found an
            # existing domain for x. Check that the new association does not violate
            # the fd constraint
            return compose_constraints(process_domain(v, domain_x), t)(state)
        return t(state)

    return _process_prefix_fd


def enforce_constraints_fd(x: Var) -> GoalProto:
    def _enforce_constraints(state: State) -> StreamThunk:
        bound_vars = state.domains.keys()
        verify_all_bound(state.constraints, bound_vars)
        return lambda: onceo(force_answer(bound_vars))(state)

    return conj(force_answer(x), Goal(_enforce_constraints))


# TODO
def reify_constraints_fd(_: Var, __: Substitution) -> ConstraintFunction:
    def _reify_constraints_fd(state: State) -> State | None:
        return state
        # raise UnboundVariables()

    return _reify_constraints_fd


class UnboundConstrainedVariable(Exception):
    pass


class UnboundVariables(Exception):
    pass


def verify_all_bound(constraints: ConstraintStore, bound_vars: list[Var]):
    if len(constraints) > 0:
        first, *rest = constraints
        var_operands = [x for x in first.operands if isinstance(x, Var)]
        for var in var_operands:
            if var not in bound_vars:
                raise UnboundConstrainedVariable(
                    f"Constrained variable {var} has no domain"
                )
        verify_all_bound(rest, bound_vars)


def force_answer(x: Var | list[Var]) -> GoalProto:
    def _force_answer(state: State) -> StreamThunk:
        match walk(x, state.sub):
            case Var(_) as var if (d := state.get_domain(var)) is not None:
                return lambda: map_sum(lambda val: eq(x, val), d)(state)
            case (first, *rest):
                return lambda: conj(force_answer(first), force_answer(rest))(state)
            case _:
                return lambda: succeed()(state)

    return _force_answer


A = TypeVar("A")


def map_sum(goal_constructor: Callable[[A], GoalProto], xs: list[A]) -> GoalProto:
    return reduce(lambda accum, x: disj(accum, goal_constructor(x)), xs, fail())


def get_sub_prefix(new_sub: Substitution, old_sub: Substitution) -> Substitution:
    return pmap({k: v for k, v in new_sub.items() if k not in old_sub})


def conda(*cases) -> GoalProto:
    _cases = []
    for case in cases:
        if isinstance(case, list | tuple):
            _cases.append((case[0], succeed) if len(case) < 2 else case)
        else:
            _cases.append((case, succeed()))

    def _conda(state: State) -> Stream:
        return starfoldr(ifte, _cases, fail())(state)

    return Goal(_conda)


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


def ifte(g1, g2, g3=fail()) -> GoalProto:
    def _ifte(state: State) -> StreamThunk:
        def ifte_loop(stream: Stream) -> StreamThunk:
            match stream:
                case ():
                    return lambda: g3(state)
                case (_, _):
                    return lambda: bind(stream, g2)
                case _:
                    return lambda: ifte_loop(stream())

        return ifte_loop(g1(state))

    return Goal(_ifte)


def onceo(g: GoalProto) -> GoalProto:
    def _onceo(state: State):
        stream = g(state)
        while stream:
            match stream:
                case (s1, _):
                    return lambda: unit(s1)
                case _:
                    stream = stream()
        return lambda: mzero

    return _onceo


def fresh(fp: Callable) -> GoalProto:
    n = fp.__code__.co_argcount

    def _fresh(state: State) -> Stream:
        i = state.var_count
        vs = (Var(j) for j in range(i, i + n))
        return fp(*vs)(state.set(var_count=i + n))

    return Goal(_fresh)


def freshn(n: int, fp: Callable) -> GoalProto:
    def _fresh(state: State) -> Stream:
        i = state.var_count
        vs = (Var(j) for j in range(i, i + n))
        return fp(*vs)(state.set(var_count=i + n))

    return Goal(_fresh)


def pull(s: Stream):
    while callable(s):
        s = s()
    return s


def take_all(s: Stream) -> list[State]:
    result = []
    rest = s
    while (s := pull(rest)) != ():
        first, rest = s
        result.append(first)
    return result


def take(n, s: Stream) -> list[State]:
    i = 0
    result = []
    rest = s
    while i < n and (s := pull(rest)) != ():
        first, rest = s
        result.append(first)
        i += 1
    return result


def itake(s: Stream) -> list[State]:
    rest = s
    while (s := pull(rest)) != ():
        first, rest = s
        yield first


def reify(states: list[State], *top_level_vars: Var):
    if len(top_level_vars) == 1:
        return [reify_state(s, top_level_vars[0]) for s in states]
    return [tuple(reify_state(s, var) for var in top_level_vars) for s in states]


def ireify(states: Generator[State, None, None], *top_level_vars: Var):
    if len(top_level_vars) == 1:
        yield from (reify_state(s, top_level_vars[0]) for s in states)
    else:
        yield from (
            tuple(reify_state(s, var) for var in top_level_vars) for s in states
        )


def reify_state(state: State, v: Var) -> Value:
    v = walk_all(v, state.sub)
    r = reify_sub(v, empty_sub())
    v = walk_all(v, r)
    return to_python(v)
    if len(state.constraints) == 0:
        return to_python(v)
    return to_python(reify_constraints(v, r)(state))


def walk_all(v: Value, s: Substitution) -> Value:
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            return v
        case cons(a, d):
            return cons(walk_all(a, s), walk_all(d, s))
        case (first, *rest) as xs:
            if isinstance(xs, list):
                return [walk_all(first, s), *walk_all(rest, s)]
            return (walk_all(first, s), *walk_all(rest, s))
        case _:
            return v


def reify_sub(v: Value, s: Substitution) -> Substitution:
    # Allows us to control how fresh Vars are reified
    v = walk(v, s)
    match v:
        case _ if isinstance(v, Var):
            return extend_substitution(v, ReifiedVar(len(s)), s)
        case (a, *d):
            return reify_sub(d, reify_sub(a, s))
        case _:
            return s


def call_with_empty_state(g: GoalProto) -> Stream:
    return g(State.empty())


def run(n: int, f_fresh_vars: Callable[..., GoalProto]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        *map(enforce_constraints, fresh_vars),
    )
    return reify(take(n, goal(state)), *fresh_vars)


def run_all(f_fresh_vars: Callable[..., GoalProto]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        *map(enforce_constraints, fresh_vars),
    )
    return reify(take_all(goal(state)), *fresh_vars)


def irun(f_fresh_vars: Callable[..., GoalProto]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        *map(enforce_constraints, fresh_vars),
    )
    return ireify(itake(goal(state, identity)), *fresh_vars)


def default_process_prefix(*_):
    return identity


def default_enforce_constraints(*_):
    return succeed()


def default_reify_constraints(*_):
    return succeed()


__PROCESS_PREFIX__: Callable[
    [Substitution, ConstraintStore],
    ConstraintFunction,
] = default_process_prefix
__ENFORCE_CONSTRAINTS__: Callable[[Var], GoalProto] = default_enforce_constraints
__REIFY_CONSTRAINTS__: Callable[
    [Var, Substitution],
    GoalProto,
] = default_reify_constraints


def set_process_prefix(constraint_function):
    global __PROCESS_PREFIX__
    __PROCESS_PREFIX__ = constraint_function


def set_enforce_constraints(goal):
    global __ENFORCE_CONSTRAINTS__
    __ENFORCE_CONSTRAINTS__ = goal


def set_reify_constraints(goal):
    global __REIFY_CONSTRAINTS__
    __REIFY_CONSTRAINTS__ = goal


def process_prefix(
    prefix: Substitution, constraints: ConstraintStore
) -> ConstraintFunction:
    global __PROCESS_PREFIX__
    return __PROCESS_PREFIX__(prefix, constraints)


def enforce_constraints(x: Var) -> GoalProto:
    global __ENFORCE_CONSTRAINTS__
    return __ENFORCE_CONSTRAINTS__(x)


def reify_constraints(x: Var, s: Substitution) -> GoalProto:
    global __REIFY_CONSTRAINTS__
    return __REIFY_CONSTRAINTS__(x, s)
