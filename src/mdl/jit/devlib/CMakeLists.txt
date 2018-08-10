# name of the target and the resulting library
set(PROJECT_NAME mdl-jit-devlib)

# collect sources
set(PROJECT_SOURCES
    "main.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    SOURCES ${PROJECT_SOURCES}
)

# additional options
target_compile_options(${PROJECT_NAME} 
    PUBLIC 
        # LLVM include files require this
        "-D__STDC_CONSTANT_MACROS"
        "-D__STDC_FORMAT_MACROS"
        "-D__STDC_LIMIT_MACROS"

        # LLVM does not use RTTI
        "$<$<CXX_COMPILER_ID:GNU>:-DNO_RTTI>"
        "$<$<CXX_COMPILER_ID:Clang>:-DNO_RTTI>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-fno-rtti>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-fno-rtti>"
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        llvm
    )

# add system dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} DEPENDS system
    COMPONENTS
        ld threads
    )