"""
TODO: maybe add a const_string or symbol type to avoid overhead for strings that won't
be combined/mutated

TODO: can we improve memory efficiency of domains by storing min/max of intervals only?

TODO: maybe store constraints as an inverse mapping of operands to constraints for more
efficient access

TODO: look at trampolining
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce, update_wrapper, wraps
from itertools import filterfalse, tee
from typing import Any, Optional, TypeAlias, TypeVar

from pyrsistent import PClass, field, pmap
from pyrsistent.typing import PMap

NOT_FOUND = object()


def identity(x):
    return x


def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filter(pred, t1), filterfalse(pred, t2)


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
    tail: Any
    is_proper: bool = False

    def __init__(self, head, tail):
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
        return f"{self.__class__.__name__}({self.head!r}, {self.tail!r})"

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


@dataclass(slots=True, frozen=True)
class Var:
    i: int


@dataclass(slots=True, frozen=True)
class ReifiedVar:
    i: int

    def __repr__(self):
        return f"_.{self.i}"


Cons: TypeAlias = cons | nil
Value: TypeAlias = Var | int | str | bool | tuple["Value", ...] | Cons
Substitution: TypeAlias = PMap[Var, Value]
NeqStore: TypeAlias = list[list[tuple[Var, Value]]]
DomainStore: TypeAlias = PMap[Var, set[int]]
ConstraintFunction: TypeAlias = Callable[["State"], Optional["State"]]
ConstraintStore: TypeAlias = list["Constraint"]


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


@dataclass(slots=True)
class Constraint:
    func: Callable[[Value, ...], ConstraintFunction]
    operands: list[Value]

    def __call__(self, state: State) -> State | None:
        return self.func(*self.operands)(state)


def make_domain(*values: int) -> set[int]:
    return set(values)


def mkrange(start: int, end: int) -> set[int]:
    return make_domain(*range(start, end + 1))


Stream: TypeAlias = tuple[()] | Callable[[], "Stream"] | tuple[State, "Stream"]
Goal: TypeAlias = Callable[[State], Stream]


def goal_from_constraint(constraint: ConstraintFunction) -> Goal:
    def _goal(state: State, continuation):
        match constraint(state):
            case None:
                return lambda: continuation(mzero)
            case _ as next_state:
                return lambda: continuation(unit(next_state))

    return goal(_goal)


@dataclass(slots=True)
class Thunk:
    func: Callable
    args: list

    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __call__(self):
        return self.func(*self.args)


class ImmatureStream(Thunk):
    ...


def trampoline_goal(goal, state=None, continuation=None):
    if state is None and continuation is None:
        result = goal()
    else:
        result = goal(state, continuation)
    while type(result) is Thunk:
        result = result()
    return result


class InvalidStream(Exception):
    pass


mzero = ()


def unit(state: State) -> Stream:
    return (state, mzero)


def mplus(s1: Stream, s2: Stream, continuation):
    match s1:
        case ():
            return lambda: continuation(s2)
        case f if callable(f):
            return lambda: lambda: mplus(s2, f(), continuation)
        case (head, tail):
            return lambda: continuation((head, mplus(s2, tail, identity)))
        case _:
            raise InvalidStream


def bind(stream: Stream, g: Goal, continuation):
    match stream:
        case ():
            return lambda: continuation(mzero)
        case f if callable(f):
            return lambda: lambda: bind(f(), g, continuation)
        case (s1, s2):
            return lambda: mplus(g(s1), bind(s2, g), identity)
        case _:
            raise InvalidStream


def walk(u: Value, s: Substitution) -> Value:
    if isinstance(u, Var):
        bound = s.get(u, NOT_FOUND)
        if bound is NOT_FOUND:
            return u
        return walk(bound, s)
    return u


def find(x: Var, s: Substitution) -> tuple[Var, Value] | None:
    for var, value in s:
        if var == x:
            return (var, value)
    return None


def extend_substitution(x: Var, v: Value, s: Substitution) -> Substitution:
    return s.set(x, v)


def extend_domain_store(x: Value, fd: set[int], d: DomainStore) -> DomainStore:
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
        case (x, *rest_x) as xs, (y, *rest_y) as ys if len(xs) == len(ys):
            for x, y in zip(xs, ys):
                s = unify(x, y, s)
                if s is None:
                    return None
            return s
        case x, y if x == y:
            return s
        case _:
            return None


def succeed() -> Goal:
    def _succeed(state, continuation):
        return eq(True, True)(state, continuation)

    return goal(_succeed)


def fail() -> Goal:
    def _fail(state, continuation):
        return eq(False, True)(state, continuation)

    return goal(_fail)


class goal:
    def __init__(self, goal_func: Goal):
        update_wrapper(self, goal_func)
        self.f = goal_func

    def __call__(self, state, continuation):
        return lambda: self.f(state, continuation)

    def __or__(self, other):
        # Use disj and conj instead of _disj and _conj, as the former delay their goals
        return disj(self, other)

    def __and__(self, other):
        return conj(self, other)


class DelayedGoal(goal):
    def __init__(self, goal_func: Goal, *args: Value):
        self.f = goal_func
        self.args = args

    def __call__(self, state, continuation):
        return Thunk(
            self.f, *self.args, lambda goal_func: Thunk(goal_func, state, continuation)
        )


def snooze(g: Goal, *args: Value):
    return goal(lambda state, continuation: lambda: g(*args)(state, continuation))


def delay(g: Goal) -> Goal:
    return goal(lambda state, continuation: lambda: g(state, continuation))


def disj(g: Goal, *goals: Goal) -> Goal:
    if goals == ():
        return delay(g)
    g2, *rest = goals
    return _disj(delay(g), disj(g2, *rest))


def _disj(g1: Goal, g2: Goal) -> Goal:
    def __disj(state: State, continuation) -> Stream:
        return lambda: g2(
            state,
            lambda g2_res: lambda: g1(
                state, lambda g1_res: lambda: mplus(g1_res, g2_res, continuation)
            ),
        )

    return goal(__disj)


def conj(g: Goal, *goals: Goal) -> Goal:
    if goals == ():
        return delay(g)
    g2, *rest = goals
    return _conj(delay(g), conj(g2, *rest))


def _conj(g1: Goal, g2: Goal) -> Goal:
    def __conj(state: State, continuation) -> Stream:
        return lambda: g1(
            state, lambda g1_result: lambda: bind(g1_result, g2, continuation)
        )

    return goal(__conj)


def eq(u: Value, v: Value) -> Goal:
    def _eqc(state: State) -> State | None:
        new_sub = unify(u, v, state.sub)
        if new_sub is None:
            # Unification failed
            return None
        elif new_sub == state.sub:
            # Unification succeeded without new associations
            return unit(state)
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


def neq(*pairs) -> Goal:
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
        remaining_pairs = [x for x in prefix.items()]
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


def domfd(x: Value, domain: set[int]) -> Goal:
    def _domfd(state: State):
        process_domain_goal = process_domain(x, domain)
        return process_domain_goal(state)

    return goal_from_constraint(_domfd)


def infd(values: tuple[Value], domain, /) -> Goal:
    infdc = reduce(
        lambda c, v: compose_constraints(c, domfd(v, domain)),
        values,
        identity_constraint,
    )
    return goal_from_constraint(infdc)


def ltefd(u: Value, v: Value) -> Goal:
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


def neqfd(u: Value, v: Value) -> Goal:
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


def alldifffd(*vs: Value) -> Goal:
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
            identity_constraint,
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


def identity_constraint(state):
    return state


def bind_constraints(f, g) -> ConstraintFunction:
    def _bind_constraints(state: State) -> State | None:
        maybe_state = f(state)
        return g(maybe_state) if maybe_state is not None else None

    return _bind_constraints


def compose_constraints(f, g) -> ConstraintFunction:
    return bind_constraints(f, g)


def run_constraints(xs: list, constraints: ConstraintStore) -> ConstraintFunction:
    match constraints:
        case []:
            return identity_constraint
        case [first, *rest]:
            if any_relevant_vars(first.operands, xs):
                return compose_constraints(
                    remove_and_run(first),
                    run_constraints(xs, rest),
                )
            else:
                return run_constraints(xs, rest)


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


def enforce_constraints_neq(_) -> Stream:
    return lambda state, continuation: lambda: continuation(unit(state))


def reify_constraints_neq(x: Var, r: Substitution) -> Goal:
    def _reify_constraints_neq(state: State):
        return state

    return _reify_constraints_neq


def process_prefix_fd(
    prefix: Substitution, constraints: ConstraintStore
) -> ConstraintFunction:
    if not prefix:
        return identity_constraint
    (x, v), *_ = prefix.items()
    t = compose_constraints(
        run_constraints([x], constraints),
        process_prefix_fd(prefix.remove(x), constraints),
    )

    def _process_prefix_fd(state: State):
        domain_x = state.get_domain(x)
        if domain_x is not None:
            # We have a new association for x (as x is in prefix), and we found an existing
            # domain for x. Check that the new association does not violate the fd constraint
            return compose_constraints(process_domain(v, domain_x), t)(state)
        return t(state)

    return _process_prefix_fd


def enforce_constraints_fd(x: Var) -> Goal:
    def _enforce_constraints(state: State, continuation):
        bound_vars = state.domains.keys()
        verify_all_bound(state.constraints, bound_vars)
        return onceo(force_answer(bound_vars))(state, continuation)

    return conj(force_answer(x), _enforce_constraints)


# TODO
def reify_constraints_fd(x: Var, r: Substitution) -> ConstraintFunction:
    def _reify_constraints_fd(state: State) -> Stream:
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


def force_answer(x: Var | list[Var]) -> Goal:
    def _force_answer(state: State, continuation):
        match walk(x, state.sub):
            case Var(_) as var if (d := state.get_domain(var)) is not None:
                return lambda: map_sum(lambda val: eq(x, val), d)(state, continuation)
            case (first, *rest):
                return lambda: force_answer(rest)(
                    state,
                    lambda value: lambda: force_answer(first)(
                        state,
                        lambda value_2: lambda: conj(value, value_2)(
                            state, continuation
                        ),
                    ),
                )
            case _:
                return lambda: succeed()(state, continuation)

    return _force_answer


A = TypeVar("A")


def map_sum(goal_constructor: Callable[[A], Goal], xs: list[A]) -> Goal:
    return reduce(lambda accum, x: disj(accum, goal_constructor(x)), xs, fail())


def get_sub_prefix(new_sub: Substitution, old_sub: Substitution) -> Substitution:
    return pmap({k: v for k, v in new_sub.items() if k not in old_sub})


def conda(*cases) -> Goal:
    _cases = []
    for case in cases:
        if isinstance(case, list | tuple):
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
    def _onceo(state: State, continuation):
        stream = g(state, continuation)
        while stream:
            match stream:
                case (s1, _):
                    return lambda: continuation(unit(s1))
                case _:
                    stream = stream()
        return lambda: continuation(mzero)

    return _onceo


def fresh(fp: Callable) -> Goal:
    n = fp.__code__.co_argcount

    def _fresh(state: State, continuation) -> Stream:
        i = state.var_count
        vs = (Var(j) for j in range(i, i + n))
        return fp(*vs)(state.set(var_count=i + n), continuation)

    return goal(_fresh)


def freshn(n: int, fp: Callable) -> Goal:
    def _fresh(state: State) -> Stream:
        i = state.var_count
        vs = (Var(j) for j in range(i, i + n))
        return fp(*vs)(state.set(var_count=i + n))

    return goal(_fresh)


def pull(s: Stream):
    while True:
        if type(s) is Thunk:
            s = trampoline_goal(s)
        elif callable(s):
            s = s()
        else:
            break
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


def reify(states: list[State], *top_level_vars: Var):
    if len(top_level_vars) == 1:
        return [reify_state(s, top_level_vars[0]) for s in states]
    return [tuple(reify_state(s, var) for var in top_level_vars) for s in states]


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


def call_with_empty_state(g: Goal) -> Stream:
    return g(State.empty())


def run(n: int, f_fresh_vars: Callable[[Var, ...], Goal]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        # *map(enforce_constraints, fresh_vars),
    )
    return reify(take(n, goal(state, identity)), *fresh_vars)


def run_all(f_fresh_vars: Callable[[Var, ...], Goal]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        # *map(enforce_constraints, fresh_vars),
    )
    return reify(take_all(goal(state, identity)), *fresh_vars)


def process_prefix(
    prefix: Substitution, constraints: ConstraintStore
) -> ConstraintFunction:
    return compose_constraints(
        process_prefix_neq(prefix, constraints),
        process_prefix_fd(prefix, constraints),
    )


def enforce_constraints(x: Var):
    return conj(
        enforce_constraints_neq(x),
        enforce_constraints_fd(x),
    )


def reify_constraints(x: Value, s: Substitution):
    return goal_from_constraint(
        compose_constraints(
            reify_constraints_neq(x, s),
            reify_constraints_fd(x, s),
        )
    )
