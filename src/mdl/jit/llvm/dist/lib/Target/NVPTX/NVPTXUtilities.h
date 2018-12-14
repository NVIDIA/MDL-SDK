//===-- NVPTXUtilities - Utilities -----------------------------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVVM specific utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXUTILITIES_H
#define NVPTXUTILITIES_H

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include <cstdarg>
#include <set>
#include <string>
#include <vector>

namespace llvm {

#define NVCL_IMAGE2D_READONLY_FUNCNAME "__is_image2D_readonly"
#define NVCL_IMAGE3D_READONLY_FUNCNAME "__is_image3D_readonly"

class NVVMAnnotations {
    typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
    typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;
    typedef std::map<const Module *, global_val_annot_t> per_module_annot_t;

public:
    NVVMAnnotations() {}

    bool findOneNVVMAnnotation(const llvm::GlobalValue *, std::string, unsigned &) const;
    bool findAllNVVMAnnotation(const llvm::GlobalValue *, std::string,
                               std::vector<unsigned> &) const;

    bool isTexture(const llvm::Value &) const;
    bool isSurface(const llvm::Value &) const;
    bool isSampler(const llvm::Value &) const;
    bool isImage(const llvm::Value &) const;
    bool isImageReadOnly(const llvm::Value &) const;
    bool isImageWriteOnly(const llvm::Value &) const;
    bool isImageReadWrite(const llvm::Value &) const;
    bool isManaged(const llvm::Value &) const;

    bool getMaxNTIDx(const llvm::Function &, unsigned &) const;
    bool getMaxNTIDy(const llvm::Function &, unsigned &) const;
    bool getMaxNTIDz(const llvm::Function &, unsigned &) const;

    bool getReqNTIDx(const llvm::Function &, unsigned &) const;
    bool getReqNTIDy(const llvm::Function &, unsigned &) const;
    bool getReqNTIDz(const llvm::Function &, unsigned &) const;

    bool getMinCTASm(const llvm::Function &, unsigned &) const;
    bool isKernelFunction(const llvm::Function &) const;

    bool getAlign(const llvm::Function &, unsigned index, unsigned &) const;

private:
    static void cacheAnnotationFromMD(const MDNode *, key_val_pair_t &);
    void cacheAnnotationFromMD(const Module *, const GlobalValue *) const;

private:
    mutable per_module_annot_t Cache;
};

bool getAlign(const llvm::CallInst &, unsigned index, unsigned &);

std::string getTextureName(const llvm::Value &);
std::string getSurfaceName(const llvm::Value &);
std::string getSamplerName(const llvm::Value &);

bool isBarrierIntrinsic(llvm::Intrinsic::ID);
bool isMemorySpaceTransferIntrinsic(Intrinsic::ID id);
const Value *skipPointerTransfer(const Value *V, bool ignore_GEP_indices);
const Value *
skipPointerTransfer(const Value *V, std::set<const Value *> &processed);
BasicBlock *getParentBlock(Value *v);
Function *getParentFunction(Value *v);
void dumpBlock(Value *v, char *blockName);
Instruction *getInst(Value *base, char *instName);
void dumpInst(Value *base, char *instName);
void dumpInstRec(Value *v, std::set<Instruction *> *visited);
void dumpInstRec(Value *v);
void dumpParent(Value *v);

}

#endif
