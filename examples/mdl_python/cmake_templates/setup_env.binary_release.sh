#*****************************************************************************
# Copyright 2022 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

@IGNORE_THE_FOLLOWING_COMMENT@
# This file is generated during CMake Configuration.
# If you want to regenerate it, delete this file and run CMake Configure again.

# Library path to load the MDL SDK and plugins from
@ENV_LIB_PATH@=@CMAKE_CURRENT_SOURCE_DIR@/../../@MI_PLATFORM_NAME@/lib${@ENV_LIB_PATH@:+:${@ENV_LIB_PATH@}}
export @ENV_LIB_PATH@

# Path from which python modules are loaded from (i.e. pymdlsdk)
PYTHONPATH=@BINDING_MODULE_PATH@/@_CONFIG@${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

# Path of the examples to access the example content
MDL_SAMPLES_ROOT=@CMAKE_CURRENT_SOURCE_DIR@/../
export MDL_SAMPLES_ROOT

# Path of the python binary matching the version used for building the bindings
PYTHON_BINARY=@MDL_DEPENDENCY_PYTHON_DEV_EXE@
export PYTHON_BINARY
