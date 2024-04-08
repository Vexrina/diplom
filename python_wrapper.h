#ifndef PYTHON_WRAPPER_H
#define PYTHON_WRAPPER_H

int initializePython();
int finalizePython();
char* callPythonCode(char *pythonCode);

#endif
