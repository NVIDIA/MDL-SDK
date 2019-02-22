#*****************************************************************************
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME mdl-jit-llvm)

# create a project to get folder references in cmake/add_llvm.cmake
project(${PROJECT_NAME})
target_print_log_header(
    TARGET ${PROJECT_NAME}
    TYPE "DEPENDENCY"
    )

# list of llvm Libraries we need
set(_LLVM_LIB_NAMES 
    LLVMAnalysis
    LLVMAsmParser
    LLVMAsmPrinter
    LLVMBitReader
    LLVMBitWriter
    LLVMCodeGen
    LLVMCore
    LLVMDebugInfo
    LLVMExecutionEngine
    LLVMInstCombine
    LLVMInstrumentation
    LLVMipa
    LLVMipo
    LLVMIRReader
    LLVMJIT
    LLVMLinker
    LLVMMC
    LLVMMCDisassembler
    LLVMMCJIT
    LLVMMCParser
    LLVMNVPTXAsmPrinter
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMObject
    LLVMOption
    LLVMRuntimeDyld
    LLVMScalarOpts
    LLVMSelectionDAG
    LLVMSupport
    LLVMTableGen
    LLVMTarget
    LLVMTransformUtils
    LLVMVectorize
    LLVMX86AsmParser
    LLVMX86AsmPrinter
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Disassembler
    LLVMX86Info
    LLVMX86Utils
    )

# -------------------------------------------------------------------------------------------------
# External LLVM Build Target 
# -------------------------------------------------------------------------------------------------

# disable unneeded stuff
# Here, we define the options before they are created within the dist/CMakeLists.txt.
# This allows to change the default values without altering the code inside the dist folder.
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX" CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
option(LLVM_INCLUDE_TESTS "Generate build targets for the LLVM unit tests." OFF)
option(LLVM_INCLUDE_EXAMPLES "Generate build targets for the LLVM examples" OFF)
option(LLVM_INCLUDE_DOCS "Generate build targets for llvm documentation." OFF)
option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." OFF)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." OFF)
option(LLVM_ENABLE_TIMESTAMPS "Enable embedding timestamp information in build" OFF)
option(LLVM_ENABLE_TERMINFO "Use terminfo database if available." OFF)
option(LLVM_ENABLE_ZLIB "Use zlib for compression/decompression if available." OFF)

# build variant settings
option(LLVM_ENABLE_ASSERTIONS "Enable assertions" OFF)
set(LLVM_DISABLE_ASSERTIONS_LOGIC ON)

if(WINDOWS)
    # use static runtime
    set(LLVM_USE_CRT_DEBUG MTd)
    set(LLVM_USE_CRT_RELEASE MT)
    set(LLVM_USE_CRT_MINSIZEREL MT)
    set(LLVM_USE_CRT_RELWITHDEBINFO MT)

elseif(MACOSX)
    # For MacOS build we must set the lowest supported OS version by the default triple.
    # This is baked into LLVM und later used to decide which runtime functions are
    # use the same version as for the 'mmacosx-version-min' flag
    set(LLVM_DEFAULT_TARGET_TRIPLE x86_64-apple-darwin10.10.0 CACHE STRING "")

    # disable mem-sanitizer even if the header was found (because the symbol can be missing)
    set(HAVE_SANITIZER_MSAN_INTERFACE_H 0)
endif()

# add boost
find_package(Boost QUIET REQUIRED)
set(DLLVM_BOOST_INC ${Boost_INCLUDE_DIRS})

# python path
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
set(PYTHON_EXECUTABLE ${python_PATH})

# pass our setup to llvm (with minor changes)
function(LLVM_TARGET_SETUP)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(LLVM_TARGET_SETUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # options depending on the target type
    get_target_property(_TARGET_TYPE ${LLVM_TARGET_SETUP_TARGET} TYPE)

    # very simple set of flags depending on the compiler instead of the combination of compiler, OS, ...
    # for more complex scenarios, replace that 

    # GENERAL 
    #---------------------------------------------------------------------------------------
    target_compile_definitions(${LLVM_TARGET_SETUP_TARGET} 
        PRIVATE
            "$<$<CONFIG:DEBUG>:_DEBUG>"
            "MI_BUILDING_LLVM"
            "__USE_XOPEN2K8"
            "_REENTRANT"
            "__STDC_CONSTANT_MACROS"
            "__STDC_FORMAT_MACROS"
            "__STDC_LIMIT_MACROS"
            ${MDL_ADDITIONAL_COMPILER_DEFINES}   # additional user defines
        )

    target_compile_options(${LLVM_TARGET_SETUP_TARGET} 
        PRIVATE
            ${MDL_ADDITIONAL_COMPILER_OPTIONS}   # additional user options
        )


    # WINDOWS
    #---------------------------------------------------------------------------------------
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${LLVM_TARGET_SETUP_TARGET} 
            PRIVATE
                "/MP"
            )
    endif()


    # LINUX
    #---------------------------------------------------------------------------------------
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${LLVM_TARGET_SETUP_TARGET} 
            PRIVATE
                "-fPIC"
                "-Wno-misleading-indentation"
                "-Wno-unused-function"
                "-Wno-nonnull-compare"
                "-Wno-implicit-fallthrough"
            )
    endif()


    # MACOSX
    #---------------------------------------------------------------------------------------
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(${LLVM_TARGET_SETUP_TARGET} 
            PRIVATE
                "-mmacosx-version-min=10.10"
                "-fPIC"
                "-m64"
                "-stdlib=libc++"
                "$<$<COMPILE_LANGUAGE:CXX>:-std=c++11>"
                "-fvisibility-inlines-hidden"
                "-fdiagnostics-fixit-info"
                "-fdiagnostics-parseable-fixits"
                "-Wno-unused-parameter"
                "-Wno-inconsistent-missing-override"
                "-Wno-unnamed-type-template-args"
                "-Wno-invalid-offsetof"
                "-Wno-long-long"
                "-Wwrite-strings"
                "-Wmissing-field-initializers"
                "-Wcovered-switch-default"
                "-Wnon-virtual-dtor"
                "-fdiagnostics-fixit-info"
                "-fdiagnostics-parseable-fixits"
            )
    endif()

endfunction()

# go down into the llvm project
add_subdirectory(dist)

message(STATUS "LLVM_USE_OPROFILE: ${LLVM_USE_OPROFILE}")
message(STATUS "LLVM_USE_INTEL_JITEVENTS : ${LLVM_USE_INTEL_JITEVENTS}")

# add llvm projects into a proper solution folder 
set(_IDE_FOLDER "src/mdl/jit/llvm")

# restore compiler flags
set(CMAKE_C_FLAGS ${_SAVED_CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS ${_SAVED_CMAKE_CXX_FLAGS})

foreach(_LIB ${_LLVM_LIB_NAMES})
    llvm_target_setup(TARGET ${_LIB}) 
    set_target_properties(${_LIB} PROPERTIES        # place the project in the right folder 
        FOLDER "${_IDE_FOLDER}/libs"
        )                          
endforeach()

# -------------------------------------------------------------------------------------------------
# set options for all other projects that are build in addition to the libs we need
set(ADDITIONAL_LLVM_TARGETS llvm-tblgen count FileCheck FileUpdate not yaml-bench)
foreach(_ADD_LLVM ${ADDITIONAL_LLVM_TARGETS})
    message(STATUS "setting flags for ${_ADD_LLVM}")
    llvm_target_setup(TARGET ${_ADD_LLVM})
    set_target_properties(${_ADD_LLVM} PROPERTIES 
        FOLDER "${_IDE_FOLDER}/additional"
    )
endforeach()

set_target_properties(intrinsics_gen NVPTXCommonTableGen X86CommonTableGen
    PROPERTIES 
        FOLDER "${_IDE_FOLDER}/additional"
    )

if(MSVC AND MSVC_IDE)
    set_target_properties(llvm_headers_do_not_build
        PROPERTIES 
            EXCLUDE_FROM_ALL 1 
            EXCLUDE_FROM_DEFAULT_BUILD 1
            FOLDER "${_IDE_FOLDER}/additional"
        )
endif()
