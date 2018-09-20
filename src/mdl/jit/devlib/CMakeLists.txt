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

# additional definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        # LLVM include files require this
        "__STDC_CONSTANT_MACROS"
        "__STDC_FORMAT_MACROS"
        "__STDC_LIMIT_MACROS"
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
        llvm
        system
    )
