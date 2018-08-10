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


# For MacOS build we must set the lowest supported OS version by the default triple.
# This is baked into LLVM und later used to decide which runtime functions are
# available ...
# Set support for OS X 10.5 Leopard (as we set in Premake)
if(MACOSX) #TODO and clang?
    set(LLVM_DEFAULT_TARGET_TRIPLE x86_64-apple-darwin9.0.0)
endif()

# -------------------------------------------------------------------------------------------------
# global settings used by llvm

# add boost
find_package(Boost QUIET REQUIRED)
set(DLLVM_BOOST_INC ${Boost_INCLUDE_DIRS})

if(LINUX)

elseif(WINDOWS)
    set(LLVM_USE_CRT_DEBUG MTd)
    set(LLVM_USE_CRT_RELEASE MT)
    set(LLVM_USE_CRT_MINSIZEREL MT)
    set(LLVM_USE_CRT_RELWITHDEBINFO MT)
elseif(MACOSX)

endif()

# python path
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
set(PYTHON_EXECUTABLE ${python_PATH})

# pass our setup to llvm (with minor changes)
function(LLVM_TARGET_SETUP)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(LLVM_TARGET_SETUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # very simple set of flags depending on the compiler instead of the combination of compiler, OS, ...
    # for more complex scenarios, replace that 
    target_compile_options(${LLVM_TARGET_SETUP_TARGET} 
        PUBLIC
            # general 
            "$<$<CONFIG:DEBUG>:-D_DEBUG>"
            "$<$<CONFIG:RELEASE>:-DNDEBUG>"
            "-DMISTD=std"                       # Use system std lib rather than a custom one
            "-DMI_BUILDING_LLVM"
            "-D__USE_XOPEN2K8"
            "$<$<CXX_COMPILER_ID:MSVC>:/MP>"

        PRIVATE 
             "$<$<CXX_COMPILER_ID:GNU>:-Wno-misleading-indentation>"
             "$<$<CXX_COMPILER_ID:GNU>:-Wno-unused-function>"
             "$<$<CXX_COMPILER_ID:GNU>:-Wno-nonnull-compare>"
             "$<$<CXX_COMPILER_ID:GNU>:-Wno-implicit-fallthrough>"
        )

endfunction()

# go down into the llvm project
add_subdirectory(dist)

# restore compiler flags
set(CMAKE_C_FLAGS ${_SAVED_CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS ${_SAVED_CMAKE_CXX_FLAGS})

foreach(_LIB ${_LLVM_LIB_NAMES})
    llvm_target_setup(TARGET ${_LIB}) 
    set_target_properties(${_LIB} PROPERTIES        # place the project in the right folder 
        FOLDER "src/mdl/jit/llvm/libs"
        )                          
endforeach()

# -------------------------------------------------------------------------------------------------
# set options for all other projects that are build in addition to the libs we need
set(ADDITIONAL_LLVM_TARGETS llvm-tblgen count FileCheck FileUpdate not yaml-bench)
foreach(_ADD_LLVM ${ADDITIONAL_LLVM_TARGETS})
    message(STATUS "setting flags for ${_ADD_LLVM}")
    llvm_target_setup(TARGET ${_ADD_LLVM})
    set_target_properties(${_ADD_LLVM} PROPERTIES 
        FOLDER "src/mdl/jit/llvm/additional"
    )
endforeach()

set_target_properties(intrinsics_gen NVPTXCommonTableGen X86CommonTableGen
    PROPERTIES 
        FOLDER "src/mdl/jit/llvm/additional"
    )

if(MSVC AND MSVC_IDE)
    set_target_properties(llvm_headers_do_not_build
        PROPERTIES 
            EXCLUDE_FROM_ALL 1 
            EXCLUDE_FROM_DEFAULT_BUILD 1
            FOLDER "src/mdl/jit/llvm/additional"
        )
endif()