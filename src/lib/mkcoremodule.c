#include <stdlib.h>
#include "Python.h"

static inline PyObject *
pull(PyObject *stream)
{
    while (PyFunction_Check(stream)) {
        stream = PyObject_CallNoArgs(stream);
        if (!stream)
            return NULL;
    }
    return stream;
}

static PyObject *
mkcore_take(PyObject *module, PyObject *args)
{
    long n = 0;
    PyObject *stream = NULL;
    if (!PyArg_ParseTuple(args, "lO:take", &n, &stream))
        return NULL;

    int capacity = 32;
    PyObject **results = realloc(NULL, capacity * sizeof(PyObject *));
    if (results == NULL) {
        PyErr_Format(PyExc_OSError, "mkcore_take: realloc failed with size %d", capacity);
        return NULL;
    }

    PyObject **next = results, **end = results + capacity;
    PyObject *empty = PyTuple_New(0);
    if (!empty)
        return NULL;

    PyObject *head = NULL, **tmp = NULL;
    int len;
    for (int i = 0; i < n; i++) {
        stream = pull(stream);
        if (!stream)
            goto error;
        if (Py_Is(stream, empty))
            break;
        head = PyTuple_GET_ITEM(stream, 0);

        if (next == end) {
            len = next - results;
            capacity += capacity;
            tmp = realloc(results, capacity * sizeof(PyObject *));
            if (tmp == NULL) {
                PyErr_Format(PyExc_OSError, "mkcore_take: realloc failed with size %d",
                             capacity);
                goto error;
            }
            results = tmp;
            next = results + len;
            end = results + capacity;
        }
        *(next++) = head;
        stream = PyTuple_GET_ITEM(stream, 1);
    }

    len = next - results;
    PyObject *py_results = PyList_New(len);
    if (!py_results)
        goto error;

    for (int i = 0; i < len; i++) {
        next = results + i;
        Py_INCREF(*next);
        PyList_SET_ITEM(py_results, i, *next);
    }

    return py_results;
error:
    free(results);
    Py_DECREF(empty);
    return NULL;
}

PyDoc_STRVAR(mkcore_take_doc, "");

static PyMethodDef mkcoremodule_methods[] = {
    {"take", mkcore_take, METH_VARARGS, mkcore_take_doc},
    {NULL, NULL},
};

static int
mkcoremodule_exec(PyObject *m)
{
    return 0;
}

static PyModuleDef_Slot mkcoremodule_slots[] = {
    {Py_mod_exec, mkcoremodule_exec},
    {0, NULL},
};

PyDoc_STRVAR(mkcoremodule_doc, "Extensions for microkanren core");

static struct PyModuleDef mkcoremodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "mkcore",
    .m_doc = mkcoremodule_doc,
    .m_size = 0,
    .m_methods = mkcoremodule_methods,
    .m_slots = mkcoremodule_slots,
};

PyMODINIT_FUNC
PyInit_mkcore()
{
    return PyModuleDef_Init(&mkcoremodule);
}
