::*****************************************************************************
:: Copyright 2022 NVIDIA Corporation. All rights reserved.
::*****************************************************************************

@IGNORE_THE_FOLLOWING_COMMENT@
:: This file is generated during CMake Configuration.
:: If you want to regenerate it, delete this file and run CMake Configure again.

:: Library path to load the MDL SDK and plugins from
set @ENV_LIB_PATH@=@ENV_MDL_SDK_PATH_BAT@/@_CONFIG@;@ENV_FREEIMAGE_PATH_BAT@/@_CONFIG@;%@ENV_LIB_PATH@%

:: Path from which python modules are loaded from (i.e. pymdlsdk)
set PYTHONPATH=@BINDING_MODULE_PATH@/@_CONFIG@;%PYTHONPATH%

:: Path of the examples to access the example content
set MDL_SAMPLES_ROOT=@CMAKE_CURRENT_SOURCE_DIR@/../

:: Path of the python binary matching the version used for building the bindings
set PYTHON_BINARY=@MDL_DEPENDENCY_PYTHON_DEV_EXE@
