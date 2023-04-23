from functools import reduce

from microkanren.core import (
    Constraint,
    ConstraintFunction,
    ConstraintStore,
    Goal,
    GoalProto,
    State,
    Stream,
    Substitution,
    Value,
    Var,
    compose_constraints,
    conj,
    extend_constraint_store,
    extend_domain_store,
    extend_substitution,
    force_answer,
    goal_from_constraint,
    run_constraints,
    walk,
)
from microkanren.goals import onceo
from microkanren.utils import identity, partition


class UnboundConstrainedVariable(Exception):
    pass


def make_domain(*values: int) -> set[int]:
    return set(values)


def mkrange(start: int, end: int) -> set[int]:
    return make_domain(*range(start, end + 1))


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


def ltfd(u: Value, v: Value) -> GoalProto:
    return goal_from_constraint(ltfdc(u, v))


def ltfdc(u: Value, v: Value) -> ConstraintFunction:
    def _ltfdc(state: State) -> State | None:
        _u = walk(u, state.sub)
        _v = walk(v, state.sub)
        dom_u = state.get_domain(_u) if isinstance(_u, Var) else make_domain(_u)
        dom_v = state.get_domain(_v) if isinstance(_v, Var) else make_domain(_v)

        next_state = state.set(
            constraints=extend_constraint_store(
                Constraint(ltfdc, (_u, _v)), state.constraints
            )
        )
        if not dom_u or not dom_v:
            return next_state

        max_v = max(dom_v)
        min_u = min(dom_u)
        return compose_constraints(
            process_domain(_u, make_domain(*(i for i in dom_u if i < max_v))),
            process_domain(_v, make_domain(*(i for i in dom_v if i > min_u))),
        )(next_state)

    return _ltfdc


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
                Constraint(ltefdc, (_u, _v)), state.constraints
            )
        )
        if not dom_u or not dom_v:
            return next_state

        max_v = max(dom_v)
        min_u = min(dom_u)
        return compose_constraints(
            process_domain(_u, make_domain(*(i for i in dom_u if i <= max_v))),
            process_domain(_v, make_domain(*(i for i in dom_v if i >= min_u))),
        )(next_state)

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
                Constraint(plusfdc, (_u, _v, _w)), state.constraints
            )
        )
        if not all((dom_u, dom_v, dom_w)):
            return next_state

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
                    Constraint(neqfdc, (_u, _v)), state.constraints
                )
            )
        elif len(dom_u) == 1 and len(dom_v) == 1 and dom_u == dom_v:
            return None
        elif dom_u.isdisjoint(dom_v):
            return state

        next_state = state.set(
            constraints=extend_constraint_store(
                Constraint(neqfdc, (_u, _v)), state.constraints
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
        unresolved = tuple(unresolved)
        values = tuple(values)
        values_domain = make_domain(*values)
        if len(values) == len(values_domain):
            return alldifffdc_resolve(unresolved, values_domain)(state)
        return None

    return _alldifffdc


def alldifffdc_resolve(
    unresolved: tuple[Var], values: set[Value]
) -> ConstraintFunction:
    def _alldifffdc_resolve(state: State) -> State | None:
        nonlocal values
        values = set(values)
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
                Constraint(
                    alldifffdc_resolve, (tuple(remains_unresolved), tuple(values))
                ),
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


def process_prefix_fd(
    prefix: Substitution, constraints: ConstraintStore
) -> ConstraintFunction:
    if not prefix:
        return identity
    (x, v), *_ = prefix.items()
    t = compose_constraints(
        run_constraints([x], constraints),
        process_prefix_fd(prefix.delete(x), constraints),
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
    def _enforce_constraints(state: State) -> Stream:
        bound_vars = state.domains.keys()
        verify_all_bound(state.constraints, bound_vars)
        return onceo(force_answer(bound_vars))(state)

    return conj(force_answer(x), Goal(_enforce_constraints))


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


# TODO
def reify_constraints_fd(_: Var, __: Substitution) -> ConstraintFunction:
    def _reify_constraints_fd(state: State) -> State | None:
        return state
        # raise UnboundVariables()

    return _reify_constraints_fd
