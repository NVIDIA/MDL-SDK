# IDE Setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # Generate folders for IDE targets

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run not moc automatically (Qt)
# We select the files manually and add the generated files to the project and the IDEs. 
# set(CMAKE_AUTOMOC ON)

# set platform variable
set(WINDOWS FALSE)
set(LINUX FALSE)
set(MACOSX FALSE)

if(WIN32 AND NOT UNIX)
    set(WINDOWS TRUE)
elseif(UNIX AND NOT WIN32 AND NOT APPLE)
    set(LINUX TRUE)
elseif(APPLE AND UNIX)
    set(MACOSX TRUE)
else()
    MESSAGE(AUTHOR_WARNING "System is currently not supported explicitly. Assuming Linux.")
    set(LINUX TRUE)
endif()

# remove the /MD flag cmake sets by default
set(CompilerFlags CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE)
foreach(CompilerFlag ${CompilerFlags})
  string(REPLACE "/MDd" "" ${CompilerFlag} "${${CompilerFlag}}")
  string(REPLACE "/MD" "" ${CompilerFlag} "${${CompilerFlag}}")
endforeach()

# mark interally used variables as advanced (since the are not supposed to be changed in CMake-Gui)
mark_as_advanced(MDL_BASE_FOLDER)
mark_as_advanced(MDL_INCLUDE_FOLDER)
mark_as_advanced(MDL_SRC_FOLDER)
mark_as_advanced(MDL_EXAMPLES_FOLDER)

# check for dependencies
# pre-declare all options that are used
# in order to show them in CMake-Gui, even the script stops because of an error.
option(MDL_ENABLE_CUDA_EXAMPLES "Enable examples that require CUDA." ON)
option(MDL_ENABLE_OPENGL_EXAMPLES "Enable examples that require OpenGL." ON)
option(MDL_ENABLE_QT_EXAMPLES "Enable examples that require Qt." ON) 

include(${MDL_BASE_FOLDER}/cmake/find/find_cuda_ext.cmake)
include(${MDL_BASE_FOLDER}/cmake/find/find_opengl_ext.cmake)
find_opengl_ext()

include(${MDL_BASE_FOLDER}/cmake/find/find_qt_ext.cmake)
find_qt_ext()

# examples could potentially use FreeImage directly
if(EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_freeimage_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_freeimage_ext.cmake)
    find_freeimage_ext()
endif()

if(EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    find_boost_ext()
endif()
