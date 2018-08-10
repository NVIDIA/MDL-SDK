##===- lib/IR/Makefile -------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../..
LIBRARYNAME = LLVMCore
BUILD_ARCHIVE = 1

BUILT_SOURCES = $(PROJ_OBJ_ROOT)/include/llvm/IR/Intrinsics.gen

include $(LEVEL)/Makefile.common

GENFILE:=$(PROJ_OBJ_ROOT)/include/llvm/IR/Intrinsics.gen

INTRINSICTD  := $(PROJ_SRC_ROOT)/include/llvm/IR/Intrinsics.td
INTRINSICTDS := $(wildcard $(PROJ_SRC_ROOT)/include/llvm/IR/Intrinsics*.td)

$(ObjDir)/Intrinsics.gen.tmp: $(ObjDir)/.dir $(INTRINSICTDS) $(LLVM_TBLGEN)
	$(call EchoOrQuietMessage, Building Intrinsics.gen.tmp from Intrinsics.td,GEN)
	$(Verb) $(LLVMTableGen) $(call SYSPATH, $(INTRINSICTD)) -o $(call SYSPATH, $@) -gen-intrinsic

$(GENFILE): $(ObjDir)/Intrinsics.gen.tmp $(PROJ_OBJ_ROOT)/include/llvm/IR/.dir
	$(Verb) $(CMP) -s $@ $< || ( $(CP) $< $@  \
	  $(if $(quiet)$(silent),, && $(EchoCmd) Updated Intrinsics.gen because Intrinsics.gen.tmp changed significantly.))

GENFILE_DEST := $(DESTDIR)$(PROJ_includedir)/llvm/IR/Intrinsics.gen

install-local:: $(GENFILE_DEST)

$(GENFILE_DEST): $(GENFILE)
	$(call EchoOrQuietMessage, Installing $@,)
	$(Verb) $(DataInstall) $< $@
