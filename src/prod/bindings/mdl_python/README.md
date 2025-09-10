# Source code for the Python binding

The Python binding is also provided as pre-generated source code such that it
can be compiled for the Python version of your choice.

For example, building it on Linux can be as simple as

    g++ -fPIC -I../../../include -c mdl_python.cpp -o mdl_python.o
    g++ -fPIC -I../../../include -c mdl_python_swig.cpp -o mdl_python_swig.o -I/usr/include/python3.11
    g++ -shared -o _pymdlsdk.so mdl_python.o mdl_python_swig.o -lpython3.11
