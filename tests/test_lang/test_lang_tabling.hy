(require microkanren.lang *)
(import microkanren.lang *)
(import microkanren [core])

;; Examples from Will Byrd's thesis.

(defn test-badbadbad []
  (defn path° [x y]
    (conde
      [(arc° x y)]
      [(fresh [z]
         (conj+ (arc° x z) (path° z y)))]))

  (defn arc° [x y]
    (conde
      [(== 'a x) (== 'b y)]
      [(== 'b x) (== 'a y)]
      [(== 'b x) (== 'd y)]))

  (setv [b a d] '[b a d])
  (setv result (run 9 [q] (path° 'a q)))
  (assert (= result
             [#(b) #(a) #(d) #(b) #(a) #(d) #(b) #(a) #(d)])))

(defn test-badbadbad-tabled []
  (defn [core.tabled] path° [x y]
    (conde
      [(arc° x y)]
      [(fresh [z]
         (conj+ (arc° x z) (path° z y)))]))

  (defn arc° [x y]
    (conde
      [(== 'a x) (== 'b y)]
      [(== 'b x) (== 'a y)]
      [(== 'b x) (== 'd y)]))

  (setv [b a d] '[b a d])
  (setv result (run* [q] (path° 'a q)))
  (assert (= result
             [#(b) #(a) #(d)])))

(defn test-mutually-recursive-no-table []
  (defn f° [x]
    (conde
      [(== 0 x)]
      [(g° x)]))

  (defn g° [x]
    (conde
      [(== 1 x)]
      [(f° x)]))

  (assert (= (run 5 [q] (f° q))
             [#(0) #(1) #(0) #(1) #(0)])))

(defn test-mutually-recursive-table-f° []
  (defn [core.tabled] f° [x]
    (conde
      [(== 0 x)]
      [(g° x)]))

  (defn g° [x]
    (conde
      [(== 1 x)]
      [(f° x)]))

  (assert (= (run 5 [q] (f° q))
             [#(0) #(1)])))

(defn test-mutually-recursive-table-both []
  (defn [core.tabled] f° [x]
    (conde
      [(== 0 x)]
      [(g° x)]))

  (defn [core.tabled] g° [x]
    (conde
      [(== 1 x)]
      [(f° x)]))

  (assert (= (run 5 [q] (f° q))
             [#(0) #(1)]))

  (assert (= (run 5 [q] (g° q))
             [#(1) #(0)])))

;; Other examples.

(defn test-same-generation° []
  (defn [core.tabled] same-generation° [x y]
    (conde
      [(== x y)]
      [(fresh [x1 y1]
         ;; Eventually diverges without tabling, as the first conde
         ;; case keeps succeeding, then we try the alt, recur,
         ;; succeed, recur, succeed, etc.
         (same-generation° x1 y1)
         (parent° x x1)
         (parent° y y1))]))

  (defn parent° [x y]
    (conde
      [(== 'john x) (== 'mary y)]
      [(== 'jane x) (== 'mary y)]
      [(== 'mary x) (== 'sam y)]))

  (setv [john jane] '[john jane])
  (assert (= (run* [q] (same-generation° q john))
             [#(john) #(jane)])))
