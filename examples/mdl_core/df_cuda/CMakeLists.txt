# name of the target and the resulting example
set(PROJECT_NAME mdl_core_example-df_cuda)

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_SOURCES
    "example_df_cuda.cpp"
    ${CMAKE_CURRENT_BINARY_DIR}/example_df_cuda.ptx
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_core_examples
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_core
        mdl_core_examples::mdl_core_shared
        opengl
        imgui
        cuda
        freeimage
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
        mdl::mdl_core
        mdl_core_examples::mdl_core_shared
    CUDA_SOURCES
        "example_df_cuda.cu"
    )