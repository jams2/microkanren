(require microkanren.lang *)
(import microkanren.lang *)

(defn test-run*-eq []
  (assert (= (run* [q] (== q 5))
             [#(5)])))
