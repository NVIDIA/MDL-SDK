# name of the target and the resulting library
set(PROJECT_NAME mdl-compiler-compilercore)

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

# generated Files
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES 
    ${_GENERATED_DIR}/Parser.h
    ${_GENERATED_DIR}/Parser.cpp
    ${_GENERATED_DIR}/Scanner.h
    ${_GENERATED_DIR}/Scanner.cpp
    ${_GENERATED_DIR}/mdl_module.h
    ${_GENERATED_DIR}/mdl_module.cpp
    ${_GENERATED_DIR}/compilercore_intrinsic_eval.i
    )

# collect sources
set(PROJECT_SOURCES 
    "compilercore_analysis.cpp"
    "compilercore_annotations.cpp"
    "compilercore_archiver.cpp"
    "compilercore_auto_type.cpp"
    "compilercore_assert.cpp"
    "compilercore_builder.cpp"
    "compilercore_call_graph.cpp"
    "compilercore_checker.cpp"
    "compilercore_debug_tools.cpp"
    "compilercore_declarations.cpp"
    "compilercore_def_table.cpp"
    "compilercore_dependency_graph.cpp"
    "compilercore_errors.cpp"
    "compilercore_expressions.cpp"
    "compilercore_fatal.cpp"
    "compilercore_file_utils.cpp"
    "compilercore_file_resolution.cpp"
    "compilercore_function_instance.cpp"
    "compilercore_intrinsic_eval.cpp"
    "compilercore_manifest.cpp"
    "compilercore_messages.cpp"
    "compilercore_mdl.cpp"
    "compilercore_mangle.cpp"
    "compilercore_malloc_allocator.cpp"
    "compilercore_memory_arena.cpp"
    "compilercore_modules.cpp"
    "compilercore_names.cpp"
    "compilercore_overload.cpp"
    "compilercore_positions.cpp"
    "compilercore_statements.cpp"
    "compilercore_types.cpp"
    "compilercore_optimizer.cpp"
    "compilercore_options.cpp"
    "compilercore_printers.cpp"
    "compilercore_sema_analysis.cpp"
    "compilercore_serializer.cpp"
    "compilercore_streams.cpp"
    "compilercore_symbols.cpp"
    "compilercore_thread_context.cpp"
    "compilercore_values.cpp"
    "compilercore_visitor.cpp"
    "compilercore_wchar_support.cpp"
    ${_GENERATED_SOURCES}
    )

# mark files as generated to disable the check for existance during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    ADDITIONAL_INCLUDE_DIRS ${_GENERATED_DIR}
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )

# -------------------------------------------------------------------------------------------------
# Code Generation Step 
# -------------------------------------------------------------------------------------------------

# get and run coco
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL coco)
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/Parser.h
        ${_GENERATED_DIR}/Parser.cpp
        ${_GENERATED_DIR}/Scanner.h
        ${_GENERATED_DIR}/Scanner.cpp
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Files with COCO ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${coco_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/compilercore_parser.atg -no_misplaced_resolver -frames . -trace FX -o ${_GENERATED_DIR}
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/Scanner.frame 
        ${CMAKE_CURRENT_SOURCE_DIR}/Parser.frame 
        ${CMAKE_CURRENT_SOURCE_DIR}/Copyright.frame
    VERBATIM
    )


# mdl
set(_STANDARD_MDL
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/std.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/limits.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/anno.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/state.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/math.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/tex.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/noise.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/df.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/builtins.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/debug.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule/nvidia_df.mdl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../shaders/mdl/base/base.mdl
    )

target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
add_custom_command(
    OUTPUT ${_GENERATED_DIR}/compilercore_intrinsic_eval.i
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Standard Module Header ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/generate_stdmodule.py -E P4ssW0rT -e -d ${_GENERATED_DIR}/mdl_module.cpp ${_STANDARD_MDL}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrinsic_eval.py ${CMAKE_CURRENT_SOURCE_DIR}/../stdmodule ${_GENERATED_DIR}/compilercore_intrinsic_eval.i
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_stdmodule.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrinsic_eval.py
        ${_STANDARD_MDL}
    VERBATIM
    )
