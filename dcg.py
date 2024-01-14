import string
from functools import partial, wraps

from fastcons import cons, nil

from microkanren import conj, disj, eq, fresh
from microkanren.goals import conso


def _(*xs):
    return cons.from_xs(xs)


def s(string):
    return cons.from_xs(c for c in string)


def qs(x):
    return disj(
        eq(x, nil()),
        fresh(
            lambda a, d: conj(
                eq(cons(a, d), x),
                eq(a, "q"),
                qs(d),
            ),
        ),
    )


def append(xs, ys):
    match xs:
        case nil():
            return ys
        case cons(a, d):
            return cons(a, append(d, ys))


def appendo(xs, ys, zs):
    return disj(
        eq(xs, nil()) & eq(ys, zs),
        fresh(
            lambda a, d, res: conj(
                eq(cons(a, d), xs),
                eq(cons(a, res), zs),
                appendo(d, ys, res),
            ),
        ),
    )


def reverse(xs):
    def _reverse(xs, acc):
        match xs:
            case nil():
                return acc
            case cons(a, d):
                return _reverse(d, cons(a, acc))

    return _reverse(xs, nil())


def same_lengtho(xs, ys):
    return disj(
        eq(xs, nil()) & eq(ys, nil()),
        fresh(
            lambda ax, dx, ay, dy: conj(
                eq(cons(ax, dx), xs),
                eq(cons(ay, dy), ys),
                same_lengtho(dx, dy),
            )
        ),
    )


def reverso(xs, sx):
    def _reverso(xs, acc, sx):
        return disj(
            eq(xs, nil()) & eq(acc, sx),
            fresh(
                lambda a, d: conj(
                    eq(cons(a, d), xs),
                    _reverso(d, cons(a, acc), sx),
                ),
            ),
        )

    return same_lengtho(xs, sx) & _reverso(xs, nil(), sx)


"""
    S → NP VP
    NP → Noun
    NP → Article Noun
    NP → Article Adjective Noun
    VP → Verb NP
"""


# def noun_phrase(xs, ys):
#     return disj(
#         noun(xs, ys),
#         fresh(
#             lambda _1: conj(
#                 article(xs, _1),
#                 noun(_1, ys),
#             )
#         ),
#         fresh(
#             lambda _1, _2: conj(
#                 article(xs, _1),
#                 adjective(_1, _2),
#                 noun(_2, ys),
#             )
#         ),
#     )


# def noun(xs, ys):
#     return disj(
#         conso("dog", ys, xs),
#         conso("cat", ys, xs),
#         conso("mouse", ys, xs),
#     )


# def article(xs, ys):
#     return disj(
#         conso("the", ys, xs),
#         conso("a", ys, xs),
#     )


# def adjective(xs, ys):
#     return disj(
#         conso("small", ys, xs),
#         conso("big", ys, xs),
#     )


# def verb_phrase(xs, ys):
#     return fresh(lambda _1: conj(verb(xs, _1), noun_phrase(_1, ys)))


# def verb(xs, ys):
#     return disj(
#         conso("chases", ys, xs),
#         conso("eats", ys, xs),
#     )


# def sentence(xs):
#     return fresh(lambda _1: conj(noun_phrase(xs, _1), verb_phrase(_1, nil())))


# Combinators


def exact(s):
    def _exact(xs, ys):
        return conso(s, ys, xs)

    return _exact


def one_of(*words):
    def _one_of(xs, ys):
        return disj(*(conso(word, ys, xs) for word in words))

    return _one_of


def chain(*rules):
    def _chain(rules, xs, ys):
        match rules:
            case []:
                return eq(xs, ys)
            case (first,):
                return first(xs, ys)
            case first, *rest:
                return fresh(lambda _1: first(xs, _1) & _chain(rest, _1, ys))

    return partial(_chain, rules)


def maybe(rule):
    def _maybe(xs, ys):
        return disj(
            rule(xs, ys),
            eq(xs, ys),
        )

    return _maybe


def star(rule):
    def _star(xs, ys):
        return disj(
            eq(xs, ys),
            chain(rule, lambda xs, ys: _star(xs, ys))(xs, ys),
        )

    return _star


def plus(rule):
    return chain(rule, star(rule))


def rule(*choices):
    def _rule(xs, ys):
        return disj(*(chain(*choice)(xs, ys) for choice in choices))

    return _rule


"""
    S → NP VP
    NP → Noun
    NP → Article Noun
    NP → Article Adjective Noun
    VP → Verb NP
"""


noun = one_of("dog", "cat", "mouse")
article = one_of("the", "a")
verb = one_of("eats", "chases")
adjective = one_of("big", "small")
noun_phrase = rule(
    [noun],
    [article, maybe(adjective), noun],
)
verb_phrase = rule(
    [verb, noun_phrase],
)
sentence = rule([noun_phrase, verb_phrase])


def parse(words):
    return sentence(words, nil())


"""
E -> aF | b
F -> cE | d
"""


def E(xs, ys):
    return rule(
        [exact("a"), F],
        [exact("b")],
    )(xs, ys)


def F(xs, ys):
    return rule(
        [exact("c"), E],
        [exact("d")],
    )(xs, ys)


"""
Q -> Qa | b
F -> cE | d
"""


def Q(xs, ys):
    return rule(
        [Q, exact("a")],
        [exact("b")],
    )(xs, ys)


"""
E -> x
  | λx.E
  | E E
"""


variable = one_of("x", "y", "z")
lam = exact("λ")
dot = exact(".")
lparen = exact("(")
rparen = exact(")")
space = exact(" ")


def Term(xs, ys):
    return rule(
        [variable],
        [lam, variable, dot, Term],
        [Term, space, Term],
    )(xs, ys)


"""
S -> NP VP
NP -> Det N | NP PP | Adj NP | N
VP -> V NP | VP PP
PP -> P NP

Det -> 'the' | 'a'
N -> 'man' | 'telescope' | 'hill'
Adj -> 'red' | 'old'
V -> 'saw' | 'met'
P -> 'with' | 'on'
"""


def Sentence(xs, ys):
    return rule([NP, VP])(xs, ys)


def NP(xs, ys):
    return rule(
        [N],
        [Det, N],
        [Adj, NP],
        [NP, PP],
    )(xs, ys)


def VP(xs, ys):
    return rule(
        [V, NP],
        [VP, PP],
    )(xs, ys)


def PP(xs, ys):
    return rule([P, NP])(xs, ys)


Det = one_of("the", "a")
N = one_of("man", "telescope", "hill", "moon")
Adj = one_of("old", "pretty")
V = one_of("saw", "met")
P = one_of("with", "on")


"""
Left recursive grammars
E → E + 'a'
E → 'a'
"""

"""
E -> x
  | λx.E
  | E E


Term → Term Term
Term → 'λ' Var '.' Term
Term → Var
Var → 'x'


Term  → Var Term'
Term  → 'λ' Var '.' Term
Term' → Term Term'
Term' → ε
Var → 'x'
"""


def parser(f):
    @wraps(f)
    def _f(xs, ys):
        return f()(xs, ys)

    return _f


var = exact("x")


@parser
def term():
    return rule(
        [var, termhat],
        [exact("λ"), var, exact("."), term],
    )


@parser
def termhat():
    return rule(
        [],
        [term, termhat],
    )


"""
Grammar for grammars

Grammar → Rules
Rules → Rule Rules
Rules → ε

Rule → Nonterminal '→' Body

Body → Symbol Body
Body → ε

Symbol → Nonterminal
Symbol → Terminal

Nonterminal → Chars
Chars → Char Chars | Char

Terminal → ''' Chars '''

Char → [a-zA-Z]
"""


def word(xs, ys, out):
    def in_(x, cs):
        return disj(*(eq(x, c) for c in cs))

    def _word(xs, ys, acc, out):
        return disj(
            eq(xs, nil()) & eq(acc, out),
            fresh(
                lambda a, d: conj(
                    conso(a, d, xs),
                    disj(
                        in_(a, string.whitespace) & eq(xs, ys),
                        in_(a, string.printable) & eq,
                    ),
                )
            ),
        )

    return _word(xs, ys, nil(), out)


"""
E -> x
  | λx.E
  | E E


1.
Term → Term Term
Term → 'λ' Var '.' Term
Term → Var
Var → 'x'

2.
Term  → Var Term'
Term  → 'λ' Var '.' Term
Term' → Term Term'
Term' → ε
Var → 'x'

3.
Term → Abstraction | Application | Var
Abstraction  → 'λ' Var '.' Term
Application  → Var Application'
Application' → Term Application'
Application' → ε
Var → 'x'
"""


def var(xs, ys, tree):
    return conj(
        conso("x", ys, xs),
        eq(tree, cons("var", "x")),
    )
