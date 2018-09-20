# name of the target and the resulting library
set(PROJECT_NAME mdl_core_shared)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_SOURCES
    "example_cuda_shared.h"
    "example_shared.h"
    "texture_support_cuda.h"
    ${DUMMY_CPP}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE mdl_core_examples
    SOURCES ${PROJECT_SOURCES}
)

# inject the MDL_SAMPLES_ROOT path
target_compile_definitions(${PROJECT_NAME} 
    PUBLIC
        "MDL_SAMPLES_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/../..\""
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        mdl::mdl_core    
    NO_RUNTIME_COPY # header only -> the consumer will also add these dependency
    )

# add include directories
target_include_directories(${PROJECT_NAME}
    INTERFACE
        # the template makes non public headers private by default
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
    )
