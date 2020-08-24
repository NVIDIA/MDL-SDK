#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

# name of the target and the resulting library
set(PROJECT_NAME mdl-jit-generator_jit)

# Files to generate
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES 
    ${_GENERATED_DIR}/libmdlrt_bitcode.h
    ${_GENERATED_DIR}/glue_libdevice.h
    ${_GENERATED_DIR}/generator_jit_intrinsic_func.i
    ${_GENERATED_DIR}/libbsdf_multiscatter_tables.cpp
    )

set(DF_HANDLE_SLOT_MODES "hsmp" "hsmn" "hsm1" "hsm2" "hsm4" "hsm8")
foreach(HS ${DF_HANDLE_SLOT_MODES})
    list(APPEND _GENERATED_SOURCES
        ${_GENERATED_DIR}/libbsdf_bitcode_${HS}.h
    )
endforeach()

# scripts/files used to generate the sources above
set(_GENERATOR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrinsic_func.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf_multiscatter_tables.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_libdevice.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_libmdlrt.py
    )

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_HEADERS
    "generator_jit.h"
    "generator_jit_ast_compute.h"
    "generator_jit_cns_pass.h"
    "generator_jit_code_printer.h"
    "generator_jit_context.h"
    "generator_jit_generated_code.h"
    "generator_jit_generated_code_value_layout.h"
    "generator_jit_hlsl_depgraph.h"
    "generator_jit_hlsl_function.h"
    "generator_jit_hlsl_writer.h"
    "generator_jit_llvm.h"
    "generator_jit_llvm_passes.h"
    "generator_jit_libbsdf_data.h"
    "generator_jit_opt_pass_gate.h"
    "generator_jit_res_manager.h"
    "generator_jit_streams.h"
    "generator_jit_type_map.h"
    )

set(PROJECT_SOURCES 
    "generator_jit.cpp"
    "generator_jit_ast_compute.cpp"
    "generator_jit_cns_pass.cpp"
    "generator_jit_code_printer.cpp"
    "generator_jit_context.cpp"
    "generator_jit_generated_code.cpp"
    "generator_jit_hlsl.cpp"
    "generator_jit_hlsl_writer.cpp"
    "generator_jit_libbsdf.cpp"
    "generator_jit_libbsdf_data.cpp"
    "generator_jit_libdevice.cpp"
    "generator_jit_libmdlrt.cpp"
    "generator_jit_mangle.cpp"
    "generator_jit_llvm.cpp"
    "generator_jit_llvm_df.cpp"
    "generator_jit_llvm_intrinsics.cpp"
    "generator_jit_llvm_passes.cpp"
    "generator_jit_type_map.cpp"
    ${_GENERATED_SOURCES}
    ${_GENERATOR_FILES}
    ${PROJECT_HEADERS}
    )

# mark files as generated to disable the check for existence during configure
# mark them also as headers since generated .cpp files are included in other sources
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)

# add the generator files to the solution
set_source_files_properties(${_GENERATOR_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("generator" FILES ${_GENERATOR_FILES})


# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    ADDITIONAL_INCLUDE_DIRS 
        ${_GENERATED_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/../../compiler/compiler_hlsl/generated
)

# additional definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        # LLVM include files require this
        "__STDC_CONSTANT_MACROS"
        "__STDC_FORMAT_MACROS"
        "__STDC_LIMIT_MACROS"

        # to get M_PI
        "_USE_MATH_DEFINES" 
    )

if(NOT MSVC)
    # LLVM does not use RTTI
    target_compile_options(${PROJECT_NAME} 
        PRIVATE 
            "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>"
        )
    target_compile_definitions(${PROJECT_NAME} 
        PRIVATE 
            "$<$<COMPILE_LANGUAGE:CXX>:NO_RTTI>"
        )
endif()

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
        mdl-jit-libbsdf
        mdl-jit-libmdlrt
    )

# add cuda headers, only required for CUDA_VERSION define
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        llvm
    NO_LINKING
    NO_RUNTIME_COPY
    )

# dependency only for build order
add_dependencies(${PROJECT_NAME} 
    mdl-compiler-compiler_hlsl
)

# -------------------------------------------------------------------------------------------------
# Code Generation Step 
# -------------------------------------------------------------------------------------------------

# get tools
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL devlib)

# mdl
set(_STANDARD_MDL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/stdmodule)
set(_STANDARD_MDL
    ${_STANDARD_MDL_DIR}/std.mdl
    ${_STANDARD_MDL_DIR}/limits.mdl
    ${_STANDARD_MDL_DIR}/anno.mdl
    ${_STANDARD_MDL_DIR}/state.mdl
    ${_STANDARD_MDL_DIR}/math.mdl
    ${_STANDARD_MDL_DIR}/tex.mdl
    ${_STANDARD_MDL_DIR}/noise.mdl
    ${_STANDARD_MDL_DIR}/df.mdl
    ${_STANDARD_MDL_DIR}/builtins.mdl
    )

# create a target, PRE_BUILD commands only work for visual studio, other platforms interpret it as PRE_LINK, which is to late
add_custom_command(
    OUTPUT ${_GENERATED_DIR}/generator_jit_intrinsic_func.i
    COMMAND ${CMAKE_COMMAND} -E echo "Generating generator_jit_intrinsic_func.i ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrinsic_func.py ${_STANDARD_MDL_DIR} ${_GENERATED_DIR}/generator_jit_intrinsic_func.i
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrinsic_func.py
        ${_STANDARD_MDL}
    VERBATIM
    )

foreach(HS ${DF_HANDLE_SLOT_MODES})
    add_custom_command(
        OUTPUT ${_GENERATED_DIR}/libbsdf_bitcode_${HS}.h
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
        COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf.py ${CMAKE_CURRENT_BINARY_DIR}/../libbsdf/generated/libbsdf_${HS}.bc ${_GENERATED_DIR}
        DEPENDS 
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf.py
            ${CMAKE_CURRENT_BINARY_DIR}/../libbsdf/generated/libbsdf_${HS}.bc
        VERBATIM
        )
endforeach()

add_custom_command(
    OUTPUT ${_GENERATED_DIR}/libmdlrt_bitcode.h
    COMMAND ${CMAKE_COMMAND} -E echo "Generating libmdlrt_bitcode.h ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_libmdlrt.py ${CMAKE_CURRENT_BINARY_DIR}/../libmdlrt/generated/libmdlrt.bc ${_GENERATED_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_libmdlrt.py
        ${CMAKE_CURRENT_BINARY_DIR}/../libmdlrt/generated/libmdlrt.bc
    VERBATIM
    )

add_custom_command(
    OUTPUT ${_GENERATED_DIR}/glue_libdevice.h
    COMMAND ${CMAKE_COMMAND} -E echo "Generating glue_libdevice.h ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_libdevice.py ${devlib_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${_GENERATED_DIR}
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_libdevice.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../libdevice/libdevice.10.bc # try to replace this with the cuda sdk file
    VERBATIM
    )

set(_TABLES
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/beckmann_smith.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/beckmann_vc.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/phong_vc.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/disk_bs.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/ggx_smith.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/ggx_vc.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/sink_vc.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/../libbsdf/multiscatter_tables/ward_gm.bin
    )

add_custom_command(
    OUTPUT ${_GENERATED_DIR}/libbsdf_multiscatter_tables.cpp
    COMMAND ${CMAKE_COMMAND} -E echo "Generating libbsdf_multiscatter_tables.cpp ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf_multiscatter_tables.py ${_GENERATED_DIR}/libbsdf_multiscatter_tables.cpp ${_TABLES}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf_multiscatter_tables.py
        ${_TABLES}
    VERBATIM
    )