(import microkanren [core])
(import hy)
(import hyrule.collections [prewalk])
(require hyrule.argmove *)

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
    [] (if goals `(conj+ ~@goals) 'core.succeed)
    [v #* vs] `(call/fresh (fn [~v] (fresh ~vs ~@goals)))))

(defmacro exist [lvars #* goals]
  `(fresh ~lvars (conj+ ~@goals)))

(defmacro run [n lvars #* goals]
  `(lfor
     x
     (core.take ~n ((fresh ~lvars ~@goals (fn [s] (core.unit (core.reify ~(tuple lvars) s.sub))))
                     (core.empty-state)))
     x))

(defmacro run* [lvars / #* goals]
  `(run -1 ~lvars ~@goals))

(defmacro unless [test / #* consequents]
  `(when (not ~test) ~@consequents))

(defmacro defne [name subject / #* cases]
  (unless (isinstance name hy.models.Symbol)
    (raise
      (ValueError
        "First positional argument to defne must be an identifier, the name of the defined relation")))
  (unless (isinstance subject hy.models.List)
    (raise
      (ValueError
        "Second positional argument to defne must be a list, the parameter list of the defined relation")))

  (defn -collect-free-vars [head bound]
    (match head
      [first #* rest] [#* (-collect-free-vars first bound) #* (-collect-free-vars rest bound)]
      x :if (and (isinstance x hy.models.Symbol) (not-in x bound)) [x]
      _ []))

  (defn -replace-anons [head]
    (prewalk (fn [x] (if (= x '_) (hy.gensym) x)) head))

  `(defn ~name ~subject
     (disj+
       ~@(lfor [head #* rest] cases
               (let [-head (-replace-anons head)]
                 `(fresh ~(-collect-free-vars -head subject)
                    (== ~-head ~subject)
                    ~@rest))))))
