# name of the target and the resulting example
set(PROJECT_NAME mdl_sdk_example-execution_cuda)

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_SOURCES
    "example_execution_cuda.cpp"
    ${CMAKE_CURRENT_BINARY_DIR}/example_execution_cuda.ptx
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk_examples
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk_examples::mdl_sdk_shared
        opengl
        imgui
        cuda
        )

# link system libraries
target_add_dependencies(TARGET ${PROJECT_NAME} DEPENDS system
    COMPONENTS
        ld
        threads
    )


# -------------------------------------------------------------------------------------------------
# PTX Target 
# -------------------------------------------------------------------------------------------------
target_add_cuda_ptx_rule(TARGET ${PROJECT_NAME} 
    DEPENDS 
        mdl::mdl_sdk
        mdl_sdk_examples::mdl_sdk_shared
    CUDA_SOURCES
        "example_execution_cuda.cu"
    )