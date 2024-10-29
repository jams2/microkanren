(import microkanren [core])
(import hy)
(require hyrule.destructure [setv+])

(setv call/fresh core.call-fresh)
(setv == core.eq)

(defmacro Zzz [g]
  `(fn [s/c] (fn [] (~g s/c))))

(defmacro conj+ [goal #* goals]
  (if (not goals)
      `(Zzz ~goal)
      `(core.conj (Zzz ~goal) (conj+ ~@goals))))

(defmacro disj+ [goal #* goals]
  (if (not goals)
      `(Zzz ~goal)
      `(core.disj (Zzz ~goal) (disj+ ~@goals))))

(defmacro conde [#* goals]
  (match goals
    [] 'core.fail
    [gs] `(conj+ ~@gs)
    [gs #* gs^] `(disj+ (conj+ ~@gs) ~@(map (fn [xs] `(conj+ ~@xs)) gs^))))

(defmacro fresh [lvars #* goals]
  (match lvars
    [] `(conj+ ~@goals)
    [v #* vs] `(call/fresh (fn [~v] (fresh ~vs ~@goals)))))

(defmacro exist [lvars #* goals]
  `(fresh ~lvars (&& #* goals)))

(defmacro run [n lvars #* goals]
  `(lfor state (core.take ~n ((fresh ~lvars ~@goals)(core.empty-state)))
         (core.reify (tuple (gfor i (range ~(len lvars)) (core.Var i))) state.sub)))

(defmacro run* [lvars #* goals]
  (hy.macroexpand `(run -1 ~lvars ~@goals)))

(defmacro defne [name args #* body]
  "Accept list patterns only, that match the arity of `args'."
  `(defn ~name ~args
     (disj+
       ~@(map (fn [case]
                (when (not (isinstance case hy.models.Expression))
                  (raise (ValueError "defne case must be a hy.models.Expression")))
                (when (not case)
                  (raise (ValueError "defne case must be non-empty")))
                (setv+ [head rest] case)
                (print (type head))
                `(~head ~@rest))
              body))))
