(require microkanren.lang *)
(import microkanren.lang *)
(import microkanren.core [Symbol])

(defreader ?
  `(hy.I.microkanren.core.Symbol ~f"_.{(hy.eval (.parse-one-form &reader))}"))

(defn test-unifies-self []
  (defne goal [x]
    ([x]))

  (assert (= (run* [q] (goal q))
             [#(#? 0)])))

(defn test-unifies-ground-val []
  (defne goal [x]
    ([5]))

  (assert (= (run* [q] (goal q))
             [#(5)])))

(defn test-unifies-ground-val-alternatives []
  (defne goal [x]
    ([5])
    ([6]))

  (assert (= (run* [q] (goal q))
             [#(5) #(6)])))

(defn test-unifies-named-fresh-var []
  (defne goal [x]
    ([a]))

  (assert (= (run* [q] (goal q))
             [#(#? 0)])))

(defn test-unifies-anon-fresh-var []
  (defne goal [x]
    ([_]))

  (assert (= (run* [q] (goal q))
             [#(#? 0)])))

(defn test-unifies-varieties []
  (defne goal [x]
    ([5])
    ([6])
    ([a])
    ([_]))

  (assert (= (run* [q] (goal q))
             [#(5) #(6) #(#? 0) #(#? 0)])))

(defn test-unifies-multiple-args []
  (defne goal [x y]
    ([5 6])
    ([a b])
    ([_ _])
    ([_ 6]))

  (assert (= (run* [a b] (goal a b))
             [#(5 6) #(#? 0 #? 1) #(#? 0 #? 1) #(#? 0 6)])))

(defn test-unifies-nested-structures []
  (defne goal [x]
    ([[1 [2 3]]]))

  (assert (= (run* [q] (goal q))
             [#([1 [2 3]])])))
