# name of the target and the resulting library
set(PROJECT_NAME mdl-jit-generator_jit)

# Files to generate
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES 
    ${_GENERATED_DIR}/libbsdf_bitcode.h
    ${_GENERATED_DIR}/glue_libdevice.h
    ${_GENERATED_DIR}/generator_jit_intrinsic_func.i
    )

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_HEADERS
    "generator_jit.h"
    "generator_jit_code_printer.h"
    "generator_jit_context.h"
    "generator_jit_generated_code.h"
    "generator_jit_llvm.h"
    "generator_jit_llvm_passes.h"
    "generator_jit_res_manager.h"
    "generator_jit_type_map.h"
    )

set(PROJECT_SOURCES 
    "generator_jit.cpp"
    "generator_jit_code_printer.cpp"
    "generator_jit_context.cpp"
    "generator_jit_generated_code.cpp"
    "generator_jit_libbsdf.cpp"
    "generator_jit_libdevice.cpp"
    "generator_jit_mangle.cpp"
    "generator_jit_llvm.cpp"
    "generator_jit_llvm_df.cpp"
    "generator_jit_llvm_intrinsics.cpp"
    "generator_jit_llvm_passes.cpp"
    "generator_jit_type_map.cpp"
    ${_GENERATED_SOURCES}
    ${PROJECT_HEADERS}
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    ADDITIONAL_INCLUDE_DIRS 
        ${_GENERATED_DIR}
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
        mdl::mdl-jit-libbsdf
    )

# add cuda headers, only required for CUDA_VERSION define
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        llvm
    NO_LINKING
    NO_RUNTIME_COPY
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

add_custom_command(
    OUTPUT ${_GENERATED_DIR}/libbsdf_bitcode.h
    COMMAND ${CMAKE_COMMAND} -E echo "Generating libbsdf_bitcode.h ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf.py ${CMAKE_CURRENT_BINARY_DIR}/../libbsdf/libbsdf.bc ${_GENERATED_DIR}
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_libbsdf.py
        ${CMAKE_CURRENT_BINARY_DIR}/../libbsdf/libbsdf.bc
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