# -------------------------------------------------------------------------------------------------
# script expects the following variables:
    # - TARGET_ADD_TOOL_DEPENDENCY_TARGET
    # - TARGET_ADD_TOOL_DEPENDENCY_TOOL
# -------------------------------------------------------------------------------------------------

# first we need to build the devlib, so we add the dependency manually
add_dependencies(${TARGET_ADD_TOOL_DEPENDENCY_TARGET} mdl-jit-devlib)

# set the path
# note, that this is an generator expression, that will be evaluated by the build system (not during configuration)
set(devlib_PATH "$<TARGET_FILE:mdl-jit-devlib>" CACHE INTERNAL "")
