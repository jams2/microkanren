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

## Tutorial: Building a Pizza Recommendation System

Let's build a simple pizza recommendation system to demonstrate how relational programming works:

```clojure
(require microkanren.lang *)
(import microkanren.lang *)

;; Define our pizza database
(defn likes° [person topping]
  (conde
    [(== person 'alice) (== topping 'mushroom)]
    [(== person 'alice) (== topping 'olive)]
    [(== person 'bob) (== topping 'pepper)]
    [(== person 'bob) (== topping 'mushroom)]))

;; Find toppings that both people like
(defn common-topping° [p1 p2 topping]
  (fresh []
    (likes° p1 topping)
    (likes° p2 topping)))

;; Query examples:
;; What does Alice like?
(run* [q]
  (likes° 'alice q))
;; => [#(mushroom) #(olive)]

;; Who likes mushrooms?
(run* [q]
  (likes° q 'mushroom))
;; => [#(alice) #(bob)]

;; What toppings do Alice and Bob have in common?
(run* [q]
  (common-topping° 'alice 'bob q))
;; => [#(mushroom)]
```

This example shows how to:
1. Define facts using `conde` for alternatives
2. Create relations between entities (people and toppings)
3. Compose relations to find common preferences
4. Query the system in different ways

## Python Core API

The core implementation is available in `microkanren.core` for direct use from Python:

```python
from microkanren.core import eq, State, empty_state, reify

# Create a goal that unifies two terms
goal = eq("x", "x")

# Run the goal with an empty state
stream = goal(empty_state())

# Process results
five_states = take(5, stream)
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
