#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <structmember.h>
#include "mkcore_module.h"

/* Base Goal type */

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
    GoalObject *self = PyObject_GC_New(GoalObject, GoalType);

    if (!PyObject_IsInstance(lhs, (PyObject *)GoalType)) {
        PyErr_SetString(PyExc_TypeError, "lhs must be a Goal");
        return NULL;
    }
    else if (!PyObject_IsInstance(rhs, (PyObject *)GoalType)) {
        PyErr_SetString(PyExc_TypeError, "rhs must be a Goal");
        return NULL;
    }

    self->lhs = lhs;
    Py_INCREF(lhs);
    self->rhs = rhs;
    Py_INCREF(rhs);
    PyObject_GC_Track(self);
    return (PyObject *)self;
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
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC,
    .slots = GoalType_Slots,
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

    PyObject *match_args = PyTuple_New(2);
    if (match_args == NULL)
        return -1;

    PyTuple_SET_ITEM(match_args, 0, PyUnicode_FromString("lhs"));
    PyTuple_SET_ITEM(match_args, 1, PyUnicode_FromString("rhs"));
    if (PyDict_SetItemString(((PyTypeObject *)state->goal_type)->tp_dict, "__match_args__",
                             match_args) < 0) {
        Py_DECREF(match_args);
        return -1;
    }
    Py_DECREF(match_args);

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
