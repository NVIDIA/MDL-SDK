# name of the target and the resulting library
set(PROJECT_NAME mdl-codegenerators-generator_dag)

# collect sources
set(PROJECT_HEADERS 
    "generator_dag.h"
    "generator_dag_builder.h"
    "generator_dag_code_printer.h"
    "generator_dag_dep_graph.h"
    "generator_dag_derivatives.h"
    "generator_dag_dumper.h"
    "generator_dag_generated_dag.h"
    "generator_dag_ir.h"
    "generator_dag_ir_checker.h"
    "generator_dag_lambda_function.h"
    "generator_dag_serializer.h"
    "generator_dag_tools.h"
    "generator_dag_type_collector.h"
    "generator_dag_walker.h"
    )

set(PROJECT_SOURCES 
    "generator_dag.cpp"
    "generator_dag_builder.cpp"
    "generator_dag_code_printer.cpp"
    "generator_dag_dep_graph.cpp"
    "generator_dag_derivatives.cpp"
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
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )

# additional definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        "_USE_MATH_DEFINES" # to get M_PI
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )