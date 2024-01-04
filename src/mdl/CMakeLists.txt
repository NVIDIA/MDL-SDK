#*****************************************************************************
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/llvm)

add_subdirectory(${MDL_SRC_FOLDER}/mdl/codegenerators/generator_code)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/codegenerators/generator_dag)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/coco)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/compiler_glsl)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/compiler_hlsl)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/compilercore)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/distiller/dist)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/integration/i18n)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/integration/mdlnr)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/libbsdf)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/libmdlrt)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/devlib)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/generator_jit)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/no_jit/generator_stub)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/runtime)

set(LINKER_DEPENDENCIES_MDL_JIT
    mdl::mdl-codegenerators-generator_code
    mdl::mdl-codegenerators-generator_dag
    mdl::mdl-compiler-compiler_glsl
    mdl::mdl-compiler-compiler_hlsl
    mdl::mdl-compiler-compilercore
    mdl::mdl-distiller-dist
    mdl::mdl-integration-i18n
    mdl::mdl-integration-mdlnr
    mdl::mdl-jit-generator_jit
    mdl::mdl-runtime
    CACHE INTERNAL "" FORCE)
