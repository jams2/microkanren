(require microkanren.lang *)
(import microkanren.lang *)

(defn fives [x]
  (conde
    [(== x 5)]
    [(Zzz (fives x))]))

(defn test-run*-eq []
  (assert (= (run* [q] (== q 5))
             [#(5)])))

(defn test-run-positive-n []
  (assert (= (len (run 5 [q] (fives q)))
             5)))

(defn test-run-zero-n []
  (assert (= (len (run 0 [q] (fives q)))
             0)))
