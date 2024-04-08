#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

int initializePython() {
    Py_Initialize();
    return 0;
}

int finalizePython() {
    Py_Finalize();
    return 0;
}

char* callPythonCode(char *pythonCode) {
     // Инициализируем Python
    Py_Initialize();

    // Создаем объекты Python для перехвата вывода
    PyObject *module = PyImport_ImportModule("io");
    PyObject *stdout = PySys_GetObject("stdout");
    PyObject *buffer = PyObject_CallMethod(module, "StringIO", NULL);
    PySys_SetObject("stdout", buffer);

    // Выполняем переданный код Python
    PyRun_SimpleString(pythonCode);

    // Получаем строку из буфера
    PyObject *value = PyObject_CallMethod(buffer, "getvalue", NULL);
    const char *outputString = PyUnicode_AsUTF8(value);

    // Освобождаем ресурсы
    Py_DECREF(module);
    Py_DECREF(buffer);
    Py_DECREF(value);

    // Освобождаем Python
    Py_Finalize();

    // Копируем строку в динамически выделенную память
    char* result = strdup(outputString);

    // Возвращаем результат
    return result;
}
