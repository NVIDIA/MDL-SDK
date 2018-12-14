# name of the target and the resulting library
set(PROJECT_NAME prod-bin-mdlc)

# collect sources
set(PROJECT_HEADERS
    "getopt.h"
    "mdlc.h"
    "search_path.h"
    )

set(PROJECT_SOURCES
    "mdlc.cpp"
    "getopt.cpp"
    "search_path.cpp"
    "main.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    OUTPUT_NAME "mdlc"
    SOURCES ${PROJECT_SOURCES}
)

# add mdl and other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
        llvm
        ${LINKER_START_GROUP}
        mdl::mdl-compiler-compilercore
        mdl::mdl-codegenerators-generator_dag
        mdl::mdl-codegenerators-generator_code
        mdl::mdl-runtime
        mdl::mdl-no_jit-generator_stub
        mdl::mdl-no_glsl-generator_stub
        mdl::base-lib-libzip
        mdl::base-lib-zlib
        mdl::base-system-version
        ${LINKER_END_GROUP}
    )
    