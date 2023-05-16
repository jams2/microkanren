#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <structmember.h>
#include "mkcore_module.h"

/* Base Goal type */

PyObject *
Goal_create(PyTypeObject *goal_type, PyObject *lhs, PyObject *rhs)
{
    GoalObject *self = PyObject_GC_New(GoalObject, goal_type);
    if (self == NULL)
        return NULL;

    self->lhs = lhs;
    Py_INCREF(lhs);
    self->rhs = rhs;
    Py_INCREF(rhs);
    PyObject_GC_Track(self);
    return (PyObject *)self;
}

PyObject *
GoalObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    mkcore_state *state = PyType_GetModuleState(type);
    if (state == NULL)
        return NULL;

    static char *kwlist[] = {"lhs", "rhs", NULL};
    PyObject *lhs = NULL, *rhs = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &lhs, &rhs))
        return NULL;

    PyTypeObject *GoalType = state->goal_type;

    if (!PyObject_IsInstance(lhs, (PyObject *)GoalType) || !PyCallable_Check(lhs)) {
        PyErr_SetString(PyExc_TypeError,
                        "lhs must be a Goal or a callable that returns a Goal");
        return NULL;
    }
    else if (!PyObject_IsInstance(rhs, (PyObject *)GoalType) || !PyCallable_Check(rhs)) {
        PyErr_SetString(PyExc_TypeError,
                        "rhs must be a Goal or a callable that returns a Goal");
        return NULL;
    }

    return Goal_create(GoalType, lhs, rhs);
}

static int
GoalObject_traverse(GoalObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(self->lhs);
    Py_VISIT(self->rhs);
    return 0;
}

static int
GoalObject_clear(PyObject *self)
{
    Py_CLEAR(Goal_LHS(self));
    Py_CLEAR(Goal_RHS(self));
    return 0;
}

void
GoalObject_dealloc(GoalObject *self)
{
    PyObject_GC_UnTrack(self);
    GoalObject_clear((PyObject *)self);
    Py_TYPE(self)->tp_free(self);
}

PyDoc_STRVAR(goal_lhs_doc, "");
PyDoc_STRVAR(goal_rhs_doc, "");

static PyMemberDef GoalObject_members[] = {
    {"lhs", T_OBJECT_EX, offsetof(GoalObject, lhs), READONLY, goal_lhs_doc},
    {"rhs", T_OBJECT_EX, offsetof(GoalObject, rhs), READONLY, goal_rhs_doc},
};

PyDoc_STRVAR(goal_doc, "Base Goal class, should not be used directly");

static PyType_Slot GoalType_Slots[] = {
    {Py_tp_doc, (void *)goal_doc},
    {Py_tp_dealloc, GoalObject_dealloc},
    {Py_tp_new, GoalObject_new},
    {Py_tp_members, GoalObject_members},
    {Py_tp_traverse, GoalObject_traverse},
    {Py_tp_clear, GoalObject_clear},
    {0, NULL},
};

static PyType_Spec GoalType_Spec = {
    .name = "mkcore.Goal",
    .basicsize = sizeof(GoalObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC |
             Py_TPFLAGS_BASETYPE,
    .slots = GoalType_Slots,
};

/* Eq Goal type */

PyObject *
EqGoalObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    mkcore_state *state = PyType_GetModuleState(type);
    if (state == NULL)
        return NULL;

    static char *kwlist[] = {"lhs", "rhs", NULL};
    PyObject *lhs = NULL, *rhs = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &lhs, &rhs))
        return NULL;

    PyTypeObject *GoalType = state->goal_type;
    return Goal_create(GoalType, lhs, rhs);
}

PyDoc_STRVAR(eq_goal_doc,
             "EqGoal(lhs, rhs)\n\n\
A goal that succeeds if lhs and rhs can be unified");

static PyType_Slot EqGoalType_Slots[] = {
    {Py_tp_doc, (void *)eq_goal_doc},    {Py_tp_new, EqGoalObject_new},
    {Py_tp_dealloc, GoalObject_dealloc}, {Py_tp_traverse, GoalObject_traverse},
    {Py_tp_clear, GoalObject_clear},     {0, NULL},
};

static PyType_Spec EqGoalType_Spec = {
    .name = "mkcore.EqGoal",
    .basicsize = sizeof(GoalObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC,
    .slots = EqGoalType_Slots,
};

/* Disj Goal type */

PyDoc_STRVAR(disj_goal_doc,
             "DisjGoal(lhs, rhs)\n\n\
A goal that succeeds if either lhs or rhs succeeds");

static PyType_Slot DisjGoalType_Slots[] = {
    {Py_tp_doc, (void *)disj_goal_doc},
    {Py_tp_dealloc, GoalObject_dealloc},
    {Py_tp_traverse, GoalObject_traverse},
    {Py_tp_clear, GoalObject_clear},
    {0, NULL},
};

static PyType_Spec DisjGoalType_Spec = {
    .name = "mkcore.DisjGoal",
    .basicsize = sizeof(GoalObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC,
    .slots = DisjGoalType_Slots,
};

/* Conj Goal type */

PyDoc_STRVAR(conj_goal_doc,
             "ConjGoal(lhs, rhs)\n\n\
A goal that succeeds if both lhs and rhs succeed");

static PyType_Slot ConjGoalType_Slots[] = {
    {Py_tp_doc, (void *)conj_goal_doc},
    {Py_tp_dealloc, GoalObject_dealloc},
    {Py_tp_traverse, GoalObject_traverse},
    {Py_tp_clear, GoalObject_clear},
    {0, NULL},
};

static PyType_Spec ConjGoalType_Spec = {
    .name = "mkcore.ConjGoal",
    .basicsize = sizeof(GoalObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC,
    .slots = ConjGoalType_Slots,
};

/* Fresh Goal type */

PyObject *
FreshGoalObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    mkcore_state *state = PyType_GetModuleState(type);
    if (state == NULL)
        return NULL;

    static char *kwlist[] = {"goal", NULL};
    PyObject *goal = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &goal))
        return NULL;

    if (!PyCallable_Check(goal)) {
        PyErr_SetString(PyExc_TypeError, "goal must be callable");
        return NULL;
    }

    PyTypeObject *fresh_goal_type = state->fresh_goal_type;
    FreshGoalObject *self = PyObject_GC_New(FreshGoalObject, fresh_goal_type);
    if (self == NULL)
        return NULL;

    self->goal = goal;
    Py_INCREF(goal);
    PyObject_GC_Track(self);
    return (PyObject *)self;
}

static int
FreshGoalObject_traverse(FreshGoalObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(self->goal);
    return 0;
}

static int
FreshGoalObject_clear(PyObject *self)
{
    Py_CLEAR(((FreshGoalObject *)self)->goal);
    return 0;
}

void
FreshGoalObject_dealloc(FreshGoalObject *self)
{
    PyObject_GC_UnTrack(self);
    FreshGoalObject_clear((PyObject *)self);
    Py_TYPE(self)->tp_free(self);
}

PyDoc_STRVAR(fresh_goal_goal_doc, "");

static PyMemberDef FreshGoalObject_members[] = {
    {"goal", T_OBJECT_EX, offsetof(FreshGoalObject, goal), READONLY, fresh_goal_goal_doc},
    {NULL},
};

PyDoc_STRVAR(fresh_goal_doc, "");

static PyType_Slot FreshGoalType_Slots[] = {
    {Py_tp_doc, (void *)fresh_goal_doc},
    {Py_tp_dealloc, (void *)FreshGoalObject_dealloc},
    {Py_tp_traverse, (void *)FreshGoalObject_traverse},
    {Py_tp_clear, (void *)FreshGoalObject_clear},
    {Py_tp_members, FreshGoalObject_members},
    {Py_tp_new, (void *)FreshGoalObject_new},
    {0, NULL},
};

static PyType_Spec FreshGoalType_Spec = {
    .name = "mkcore.FreshGoal",
    .basicsize = sizeof(FreshGoalObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = FreshGoalType_Slots,
};

/* Module init */
static int
mkcore_module_exec(PyObject *m)
{
    mkcore_state *state = PyModule_GetState(m);
    if (state == NULL)
        return -1;

    state->goal_type = (PyTypeObject *)PyType_FromModuleAndSpec(m, &GoalType_Spec, NULL);
    if (state->goal_type == NULL)
        return -1;
    if (PyModule_AddType(m, state->goal_type) < 0)
        return -1;

    state->eq_goal_type = (PyTypeObject *)PyType_FromModuleAndSpec(
        m, &EqGoalType_Spec, (PyObject *)state->goal_type);
    if (state->eq_goal_type == NULL)
        return -1;
    if (PyModule_AddType(m, state->eq_goal_type) < 0)
        return -1;

    state->disj_goal_type = (PyTypeObject *)PyType_FromModuleAndSpec(
        m, &DisjGoalType_Spec, (PyObject *)state->goal_type);
    if (state->disj_goal_type == NULL)
        return -1;
    if (PyModule_AddType(m, state->disj_goal_type) < 0)
        return -1;

    state->conj_goal_type = (PyTypeObject *)PyType_FromModuleAndSpec(
        m, &ConjGoalType_Spec, (PyObject *)state->goal_type);
    if (state->conj_goal_type == NULL)
        return -1;
    if (PyModule_AddType(m, state->conj_goal_type) < 0)
        return -1;

    state->fresh_goal_type = (PyTypeObject *)PyType_FromModuleAndSpec(
        m, &FreshGoalType_Spec, (PyObject *)state->goal_type);
    if (state->fresh_goal_type == NULL)
        return -1;
    if (PyModule_AddType(m, state->fresh_goal_type) < 0)
        return -1;

    return 0;
}

static PyModuleDef_Slot mkcore_module_slots[] = {
    {Py_mod_exec, mkcore_module_exec},
    {0, NULL},
};

static int
mkcore_module_traverse(PyObject *m, visitproc visit, void *arg)
{
    mkcore_state *state = PyModule_GetState(m);
    if (state == NULL)
        return -1;
    Py_VISIT(state->goal_type);
    return 0;
}

static int
mkcore_module_clear(PyObject *m)
{
    mkcore_state *state = PyModule_GetState(m);
    if (state == NULL)
        return -1;
    Py_CLEAR(state->goal_type);
    return 0;
}

static struct PyModuleDef mkcore_module_def = {
    PyModuleDef_HEAD_INIT,          .m_name = "mkcore",
    .m_doc = "mkcore module",       .m_size = sizeof(mkcore_state),
    .m_slots = mkcore_module_slots, .m_traverse = mkcore_module_traverse,
    .m_clear = mkcore_module_clear};

PyMODINIT_FUNC
PyInit_mkcore(void)
{
    return PyModuleDef_Init(&mkcore_module_def);
}
