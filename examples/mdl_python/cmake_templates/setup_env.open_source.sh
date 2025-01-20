#*****************************************************************************
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

@IGNORE_THE_FOLLOWING_COMMENT@
# This file is generated during CMake Configuration.
# If you want to regenerate it, delete this file and run CMake Configure again.

# Library path to load the MDL SDK and plugins from
@ENV_LIB_PATH@=@ENV_MDL_SDK_PATH@/@_CONFIG@:@ENV_OPENIMAGEIO_PATH@/@_CONFIG@:@ENV_DDS_PATH@/@_CONFIG@:@ENV_MDLDISTILLER_PATH@/@_CONFIG@${@ENV_LIB_PATH@:+:${@ENV_LIB_PATH@}}
export @ENV_LIB_PATH@

# Path from which python modules are loaded from (i.e. pymdlsdk)
PYTHONPATH=@BINDING_MODULE_PATH@/@_CONFIG@${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

# Path of the examples to access the example content
MDL_SAMPLES_ROOT=@CMAKE_CURRENT_SOURCE_DIR@/../
export MDL_SAMPLES_ROOT

# Path of the examples to access the common modules
MDL_SRC_SHADERS_MDL=@MDL_SRC_FOLDER@/shaders/mdl/
export MDL_SRC_SHADERS_MDL

# Path of the python binary matching the version used for building the bindings
PYTHON_BINARY=@MDL_DEPENDENCY_PYTHON_DEV_EXE@
export PYTHON_BINARY
