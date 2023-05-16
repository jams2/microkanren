#ifndef mkcoremodule_h
#define mkcoremodule_h

#include <Python.h>

typedef struct {
    PyTypeObject *goal_type;
    PyTypeObject *eq_goal_type;
    PyTypeObject *disj_goal_type;
    PyTypeObject *conj_goal_type;
    PyTypeObject *fresh_goal_type;
} mkcore_state;

typedef struct {
    PyObject_HEAD PyObject *lhs;
    PyObject *rhs;
} GoalObject;

PyObject *
GoalObject_new(PyTypeObject *, PyObject *, PyObject *);

static int
GoalObject_traverse(GoalObject *, visitproc, void *);

static int
GoalObject_clear(PyObject *);

#define Goal_LHS(obj) ((GoalObject *)obj)->lhs
#define Goal_RHS(obj) ((GoalObject *)obj)->rhs

typedef struct {
  PyObject_HEAD PyObject *goal;
} FreshGoalObject;

#endif
