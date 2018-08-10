# name of the target and the resulting library
set(PROJECT_NAME mdl-codegenerators-generator_dag)

# collect sources
set(PROJECT_SOURCES 
    "generator_dag.cpp"
    "generator_dag_builder.cpp"
    "generator_dag_code_printer.cpp"
    "generator_dag_dep_graph.cpp"
    "generator_dag_dumper.cpp"
    "generator_dag_generated_dag.cpp"
    "generator_dag_ir.cpp"
    "generator_dag_ir_checker.cpp"
    "generator_dag_lambda_function.cpp"
    "generator_dag_serializer.cpp"
    "generator_dag_thin_walled.cpp"
    "generator_dag_type_collector.cpp"
    "generator_dag_opacity.cpp"
    "generator_dag_walker.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )

# additional options
target_compile_options(${PROJECT_NAME} 
    PRIVATE 
        "-D_USE_MATH_DEFINES" # to get M_PI
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )