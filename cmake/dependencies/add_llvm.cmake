# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# assuming we can use a non-modified version some day

# up till then we add a build dependency here
# add includes link dependencies

# list of llvm Libraries we need
set(_LLVM_LIB_NAMES 
    LLVMAnalysis
    LLVMAsmParser
    LLVMAsmPrinter
    LLVMBitReader
    LLVMBitWriter
    LLVMCodeGen
    LLVMCore
    LLVMDebugInfo
    LLVMExecutionEngine
    LLVMInstCombine
    LLVMInstrumentation
    LLVMipa
    LLVMipo
    LLVMIRReader
    LLVMJIT
    LLVMLinker
    LLVMMC
    LLVMMCDisassembler
    LLVMMCJIT
    LLVMMCParser
    LLVMNVPTXAsmPrinter
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMObject
    LLVMOption
    LLVMRuntimeDyld
    LLVMScalarOpts
    LLVMSelectionDAG
    LLVMSupport
    LLVMTableGen
    LLVMTarget
    LLVMTransformUtils
    LLVMVectorize
    LLVMX86AsmParser
    LLVMX86AsmPrinter
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Disassembler
    LLVMX86Info
    LLVMX86Utils
    )
    
target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
    PRIVATE
        ${mdl-jit-llvm_SOURCE_DIR}/dist/include
        ${mdl-jit-llvm_BINARY_DIR}/dist/include
    )

if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)

    # avoid the transitive dependencies here to be able to create a linker group
    # since we simply link all llvm libs, the transitive dependencies are covered
    foreach(_LIB ${_LLVM_LIB_NAMES})
        list(APPEND _STATIC_LIB_FILE_LIST $<TARGET_FILE:${_LIB}>)
        add_dependencies(${__TARGET_ADD_DEPENDENCY_TARGET} ${_LIB}) # add dependency manually
    endforeach()

    target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${LINKER_START_GROUP}
            ${_STATIC_LIB_FILE_LIST}
            ${LINKER_END_GROUP}
        )
endif()
