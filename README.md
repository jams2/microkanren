# microkanren

`microkanren` is an implementation of a miniKanren style relational programming language. The solver is implemented in the style of μKanren[^1], with a Hy-based interface that closely follows the traditional miniKanren API. The core implementation is in Python and can be used directly for advanced use cases.

## Installation

```bash
pip install microkanren[hy]  # For the Hy interface
pip install microkanren      # For just the Python core
```

## Usage with Hy

The main interface is provided through Hy macros and functions in `microkanren.lang`. Here's a basic example:

```clojure
(require microkanren.lang *)
(import microkanren.lang *)

;; Basic unification
(run* [q]
  (== q 5))  ; Returns [#(5)]

;; Conjunction with fresh variables
(fresh [x y]
  (== x 1)
  (== y 2)
  (== q [x y]))

;; Disjunction with conde
(conde
  [(== q 'apple)]
  [(== q 'orange)])

;; Basic pattern-matching support with defne
(defne ancestor° [x y]
  ([[x y]]
   (parent° x y))
  ([[x y]]
   (fresh [z]
     (parent° x z)
     (ancestor° z y))))
```

### Tabling Support

Relations can be tabled, using the `tabled` decorator to improve performance, and in some cases avoid divergence:

```clojure
(defn [core.tabled] path° [x y]
  (conde
    [(arc° x y)]
    [(fresh [z]
       (arc° x z)
       (path° z y))]))
```

## Python Core API

The core implementation is available in `microkanren.core` for direct use from Python:

```python
from microkanren.core import eq, State, empty_state, reify

# Create a goal that unifies two terms
goal = eq("x", "x")

# Run the goal with an empty state
stream = goal(empty_state())

# Process results
state = next(stream)
```

## Developing microkanren

Requirements:
- Python 3.12
- Hy 1.0+

1. `git clone git@github.com:jams2/microkanren.git`
2. `pip install -e '.[dev,hy]'`

Run tests with `pytest`.

Format code:
```bash
ruff check --fix src tests
ruff format src tests
```

[^1]: [μKanren: A Minimal Functional Core for Relational Programming (Hemann & Friedman, 2013)](http://webyrd.net/scheme-2013/papers/HemannMuKanren2013.pdf)
