# microkanren

`microkanren` is an implementation of a miniKanren style relational programming language, embedded in Python. The solver is implemented in the style of μKanren[^1]. It provides a framework for extending the language with constraints, as well as a basic implementation of disequality and finite domain constraints, in the style of cKanren[^2].

Due to the differences between Python and the reference implementation languages (Scheme, Racket), some divergences from the typical miniKanren API are necessary. It is a goal to capture the spirit of the miniKanren language family, but not the exact API.

* [Installation](#installation)
* [Usage](#usage)
  + [Basic usage](#basic-usage)
  + [Conjunction and disjunction](#conjunction-and-disjunction)
  + [The result type and multiple top-level variables](#the-result-type-and-multiple-top-level-variables)
  + [Defining goal constructors](#defining-goal-constructors)
    - [Recursive goal constructors and `snooze` (Zzz)](#recursive-goal-constructors-and--snooze---zzz-)
* [Developing microkanren](#developing-microkanren)

## Installation

``` bash
pip install microkanren
```

## Usage

### Basic usage

The basic goal constructor is `eq`. `eq` takes two terms as arguments, and returns a goal that will succeed if the terms can be unified, and fails otherwise.

``` python-console
>>> from microkanren import eq
>>> eq("🍕", "🍕")
<microkanren.core.Goal object at 0x7f07d85cced0>
```

To run a goal, use one of the provided interfaces: `run`, `run_all`, or `irun`. `run` takes two arguments:

1. an integer, the maximum number of results to return; and
2. a callable with positional-only arguments, each of which will receive a fresh logic variable.

`run_all` and `irun` take a single argument, the fresh-var-receiver.

``` python-console
>>> from microkanren import run
>>> run(1, lambda x: eq(x, "🍕"))
['🍕']
```

The return type of `run` and `run_all` is a (possibly-empty) list of results. If the list is empty, there are no solutions that satisfy the goal. `irun` returns a generator that yields single results.

``` python-console
>>> from microkanren import irun
>>> rs = irun(lambda x: eq(x, "😁"))
>>> next(rs)
'😁'
>>> next(rs)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

### Conjunction and disjunction

Conjunction and disjunction are provided by the vararg `conj` and `disj` functions. `Goal` objects support combination using `|` and `&` operators, which map to `conj` and `disj`.

``` python-console
>>> from microkanren import run_all
>>> run_all(lambda x: disj(eq(x, "α"), eq(x, "β"), eq(x, "δ")))
['α', 'β', 'δ']
>>> run_all(lambda x: eq(x, "α") | eq(x, "β") | eq(x, "δ"))
['α', 'β', 'δ']
>>> run_all(lambda x: eq(x, "ω") & eq(x, "ω"))
['ω']
>>> run_all(lambda x: conj(eq(x, "ω"), eq(x, "ω")))
['ω']
```

### The result type and multiple top-level variables

If the fresh-var-receiver provided to an interface has arity 1, results will be single elements. If it has arity > 1, the results will be a tuple of values, each mapping position-wise to the receiver's arguments.

``` python-console
>>> run_all(lambda x, y: eq(x, "foo") & eq(y, "bar") | eq(x, "hello") & eq(y, "world"))
[('foo', 'bar'), ('hello', 'world')]
```

### Defining goal constructors

Calling goal constructors in your top-level program quickly becomes unwieldy. To mitigate this, you can define your own goal constructors.

A goal constructor is a function that takes zero or more arguments, and returns a `Goal` (or some object that implements the `GoalProto`).

A `Goal` is a callable that takes a `State` and returns a `Stream` of `State` objects.

A `Stream` is either:
- empty (`mzero`);
- a callable of no arguments that returns a `Stream` (a thunk); or
- a tuple, `(State, Stream)`.

``` python-console
>>> def likes_pizza(person, out):
...     return eq(out, (person, "likes 🍕"))
...
>>> run_all(lambda q: likes_pizza("Jane", q) | likes_pizza("Bill", q))
[('Jane', 'likes 🍕'), ('Bill', 'likes 🍕')]
```

As shown in the above example, it can be convenient to define goals in terms of the combination of other goals. However, if you require access to the current state, you can define the goal returned by your goal constructor explicitly.

``` python
def my_constructor(x):
    def _my_constructor(state):
        if there_is_something_about(x):
            return unit(state)
        return mzero

    return Goal(_my_constructor)
```

Wrapping your goal with `Goal` means it will be combinable with other goals using `|` and `&`.

#### Recursive goal constructors and `snooze` (Zzz)

If your goal constructor is directly recursive, it will never terminate.

``` python-console
>>> def always_pizza(x):
...     return eq(x, "🍕") | always_pizza(x)
...
>>> run(1, lambda x: always_pizza(x))
...
RecursionError: maximum recursion depth exceeded while calling a Python object
```

We provide `snooze` to delay the construction of a goal until it is needed. Using `snooze` we can fix `always_pizza` to return an infinite stream of pizza[^3].

``` python-console
>>> def always_pizza(x):
...     return eq(x, "🍕") | snooze(always_pizza, x)
...
>>> rs = irun(lambda x: always_pizza(x))
>>> next(rs)
'🍕'
>>> next(rs)
'🍕'
>>> next(rs)
'🍕'
>>> next(rs)
'🍕'
```

## Developing microkanren

`microkanren` currently requires Python 3.11.

1. `git clone git@github.com:jams2/microkanren.git`
2. `pip install -e .[dev,testing]`

Run the tests with `pytest`.

Format code with `black` and `ruff`:

``` bash
black .
ruff check --fix src tests
```

[^1]: [μKanren: A Minimal Functional Core for Relational Programming (Hemann & Friedman, 2013)](http://webyrd.net/scheme-2013/papers/HemannMuKanren2013.pdf)
[^2]: [cKanren: miniKanren with constraints (Alvis et al, 2011)](http://www.schemeworkshop.org/2011/papers/Alvis2011.pdf)
[^3]: original example `fives` from the μKanren paper altered here to provide more pizza
