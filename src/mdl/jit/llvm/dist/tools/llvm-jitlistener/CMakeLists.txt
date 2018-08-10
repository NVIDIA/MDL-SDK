# This tool is excluded from the CMake build if Intel JIT events are disabled.

link_directories( ${LLVM_INTEL_JITEVENTS_LIBDIR} )
include_directories( ${LLVM_INTEL_JITEVENTS_INCDIR} )

if( LLVM_INCLUDE_INTERPRETER )
set(LLVM_LINK_COMPONENTS
  asmparser
  bitreader
  debuginfo
  inteljitevents
  interpreter
  irreader
  jit
  mcjit
  nativecodegen
  object
  selectiondag
  )
else()
set(LLVM_LINK_COMPONENTS
  asmparser
  bitreader
  debuginfo
  inteljitevents
  irreader
  jit
  mcjit
  nativecodegen
  object
  selectiondag
  )
endif( LLVM_INCLUDE_INTERPRETER )

add_llvm_tool(llvm-jitlistener
  llvm-jitlistener.cpp
  )
