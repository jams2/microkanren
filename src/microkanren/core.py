"""
TODO: maybe add a const_string or symbol type to avoid overhead for strings that won't
be combined/mutated

TODO: can we improve memory efficiency of domains by storing min/max of intervals only?

TODO: maybe store constraints as an inverse mapping of operands to constraints for more
efficient access
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import reduce, wraps
from typing import Any, Optional, Protocol, TypeAlias, TypeVar

import immutables
from fastcons import cons, nil
from pyrsistent import PClass, field

from microkanren.utils import foldr, identity

NOT_FOUND = object()


@dataclass(slots=True, frozen=True)
class Var:
    i: int


@dataclass(slots=True, frozen=True)
class ReifiedVar:
    i: int

    def __repr__(self):
        return f"_.{self.i}"


Value: TypeAlias = (
    Var | int | str | bool | tuple["Value", ...] | list["Value"] | cons | nil
)
Substitution: TypeAlias = immutables.Map[Var, Value]
NeqStore: TypeAlias = list[list[tuple[Var, Value]]]
DomainStore: TypeAlias = immutables.Map[Var, set[int]]
ConstraintFunction: TypeAlias = Callable[["State"], Optional["State"]]
ConstraintStore: TypeAlias = list["Constraint"]


def empty_sub() -> Substitution:
    return immutables.Map()


def empty_domain_store() -> DomainStore:
    return immutables.Map()


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


@dataclass(slots=True, frozen=True)
class Constraint:
    func: ConstraintProto
    operands: tuple[Value]

    def __call__(self, state: State) -> State | None:
        return self.func(*self.operands)(state)


Stream: TypeAlias = tuple[()] | Callable[[], "Stream"] | tuple[State, "Stream"]


class GoalProto(Protocol):
    def __call__(self, state: State) -> Stream:
        ...


class GoalConstructorProto(Protocol):
    def __call__(self, *args: Value) -> GoalProto:
        ...


class Goal:
    def __init__(self, goal: GoalProto):
        self.goal = goal

    def __call__(self, state: State) -> Stream:
        # Inverse η-delay happens here
        return lambda: self.goal(state)

    def __or__(self, other):
        return disj(self, other)

    def __and__(self, other):
        return conj(self, other)


def goal_from_constraint(constraint: ConstraintFunction) -> GoalProto:
    def _goal(state: State) -> Stream:
        match constraint(state):
            case None:
                return mzero
            case _ as next_state:
                return unit(next_state)

    return Goal(_goal)


class InvalidStream(Exception):
    pass


mzero = ()


def unit(state: State) -> Stream:
    return (state, mzero)


def mplus(s1: Stream, s2: Stream) -> Stream:
    match s1:
        case ():
            return lambda: s2
        case f if callable(f):
            return lambda: mplus(s2, f())
        case (head, tail):
            return lambda: (head, mplus(s2, tail))
        case _:
            raise InvalidStream


def bind(stream: Stream, g: GoalProto) -> Stream:
    match stream:
        case ():
            return mzero
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
            for x, y in zip(xs, ys):  # NOQA: B905
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
    def delayed_goal(state) -> Stream:
        return g(*args)(state)

    return Goal(delayed_goal)


def delay(g: GoalProto) -> GoalProto:
    return Goal(lambda state: g(state))


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
            return Hooks.process_prefix(prefix, state.constraints)(
                state.set(sub=new_sub)
            )

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


def pairs(xs):
    _xs = iter(xs)
    while True:
        try:
            a = next(_xs)
        except StopIteration:
            break
        try:
            b = next(_xs)
            yield (a, b)
        except StopIteration as err:
            raise ValueError("got sequence with uneven length") from err


def unpairs(xs):
    for a, b in xs:
        yield a
        yield b


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


def neq(u, v, /, *rest) -> GoalProto:
    return goal_from_constraint(neqc(u, v, *rest))


def neqc(u, v, *rest) -> ConstraintFunction:
    def _neqc(state: State) -> State | None:
        new_sub = reduce(flip(maybe_unify), pairs(rest), unify(u, v, state.sub))
        if new_sub is None:
            return state
        elif new_sub == state.sub:
            return None
        prefix = get_sub_prefix(new_sub, state.sub)
        remaining_pairs = tuple(prefix.items())
        return state.set(
            constraints=extend_constraint_store(
                Constraint(neqc, tuple(unpairs(remaining_pairs))), state.constraints
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
    return lambda state: unit(state)


def reify_constraints_neq(_: Var, __: Substitution) -> ConstraintFunction:
    return identity


class UnboundVariables(Exception):
    pass


def force_answer(x: Var | list[Var]) -> GoalProto:
    def _force_answer(state: State) -> Stream:
        match walk(x, state.sub):
            case Var(_) as var if (d := state.get_domain(var)) is not None:
                return map_sum(lambda val: eq(x, val), d)(state)
            case (first, *rest) | cons(first, rest):
                return conj(force_answer(first), force_answer(rest))(state)
            case _:
                return succeed()(state)

    return _force_answer


A = TypeVar("A")


def map_sum(goal_constructor: Callable[[A], GoalProto], xs: list[A]) -> GoalProto:
    return reduce(lambda accum, x: disj(accum, goal_constructor(x)), xs, fail())


def get_sub_prefix(new_sub: Substitution, old_sub: Substitution) -> Substitution:
    mutation = new_sub.mutate()
    for k in new_sub:
        if k in old_sub:
            del mutation[k]
    return mutation.finish()


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
    reified_sub = reify_sub(v, empty_sub())
    v = walk_all(v, reified_sub)
    return Hooks.reify_value(Hooks.reify_constraints(v, reified_sub, state))


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
            return extend_substitution(v, Hooks.reify_var(v, s), s)
        case (a, *d) | cons(a, d):
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
        *map(Hooks.enforce_constraints, fresh_vars),
    )
    return reify(take(n, goal(state)), *fresh_vars)


def run_all(f_fresh_vars: Callable[..., GoalProto]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        *map(Hooks.enforce_constraints, fresh_vars),
    )
    return reify(take_all(goal(state)), *fresh_vars)


def irun(f_fresh_vars: Callable[..., GoalProto]):
    n_vars = f_fresh_vars.__code__.co_argcount
    fresh_vars = tuple(Var(i) for i in range(n_vars))
    state = State.empty().set(var_count=n_vars)
    goal = conj(
        f_fresh_vars(*fresh_vars),
        *map(Hooks.enforce_constraints, fresh_vars),
    )
    return ireify(itake(goal(state)), *fresh_vars)


def default_process_prefix(*_):
    return identity


def default_enforce_constraints(*_):
    return succeed()


def default_reify_constraints(value, *_):
    return value


def default_reify_var(_, substitution):
    return ReifiedVar(len(substitution))


def default_reify_value(value):
    return value


class HooksMeta(type):
    def _set_hook(cls, name):
        def update_hooks(hook_function):
            if not hasattr(cls, name):
                raise AttributeError(
                    f"Cannot set unknown hook '{name}' on registry '{cls.__name__}'"
                )
            setattr(cls, name, hook_function)

        return update_hooks

    def __getattribute__(cls, attr):
        if attr.startswith("set_"):
            return cls._set_hook(attr[4:])
        return super().__getattribute__(attr)


class Hooks(metaclass=HooksMeta):
    process_prefix: Callable[
        [Substitution, ConstraintStore], ConstraintFunction
    ] = default_process_prefix
    enforce_constraints: Callable[[Var], GoalProto] = default_enforce_constraints
    reify_constraints: Callable[
        [Value, Substitution, State], Any
    ] = default_reify_constraints
    reify_var: Callable[[Var, Substitution], Any] = default_reify_var
    reify_value: Callable[[Any], Any] = default_reify_value
