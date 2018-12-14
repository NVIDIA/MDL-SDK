//===- NVPTXUtilities.cpp - Utility Functions -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "NVPTX.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/InstIterator.h"

namespace llvm {

void NVVMAnnotations::cacheAnnotationFromMD(const MDNode *md, key_val_pair_t &retval) {
  assert(md && "Invalid mdnode for annotation");
  assert((md->getNumOperands() % 2) == 1 && "Invalid number of operands");
  // start index = 1, to skip the global variable key
  // increment = 2, to skip the value for each property-value pairs
  for (unsigned i = 1, e = md->getNumOperands(); i != e; i += 2) {
    // property
    const MDString *prop = dyn_cast<MDString>(md->getOperand(i));
    assert(prop && "Annotation property not a string");

    // value
    ConstantInt *Val = dyn_cast<ConstantInt>(md->getOperand(i + 1));
    assert(Val && "Value operand not a constant int");

    std::string keyname = prop->getString().str();
    if (retval.find(keyname) != retval.end())
      retval[keyname].push_back(Val->getZExtValue());
    else {
      std::vector<unsigned> tmp;
      tmp.push_back(Val->getZExtValue());
      retval[keyname] = tmp;
    }
  }
}

void NVVMAnnotations::cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) const {
  NamedMDNode *NMD = m->getNamedMetadata(NamedMDForAnnotations);
  if (!NMD)
    return;
  key_val_pair_t tmp;
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *elem = NMD->getOperand(i);

    Value *entity = elem->getOperand(0);
    // entity may be null due to DCE
    if (!entity)
      continue;
    if (entity != gv)
      continue;

    // accumulate annotations for entity in tmp
    cacheAnnotationFromMD(elem, tmp);
  }

  if (tmp.empty()) // no annotations for this gv
    return;

  if (Cache.find(m) != Cache.end())
    Cache[m][gv] = tmp;
  else {
    global_val_annot_t tmp1;
    tmp1[gv] = tmp;
    Cache[m] = tmp1;
  }
}

bool NVVMAnnotations::findOneNVVMAnnotation(const GlobalValue *gv, std::string prop,
                                            unsigned &retval) const {
  const Module *m = gv->getParent();
  if (Cache.find(m) == Cache.end())
    cacheAnnotationFromMD(m, gv);
  else if (Cache[m].find(gv) == Cache[m].end())
    cacheAnnotationFromMD(m, gv);
  if (Cache[m][gv].find(prop) == Cache[m][gv].end())
    return false;
  retval = Cache[m][gv][prop][0];
  return true;
}

bool NVVMAnnotations::findAllNVVMAnnotation(const GlobalValue *gv, std::string prop,
                                            std::vector<unsigned> &retval) const {
  const Module *m = gv->getParent();
  if (Cache.find(m) == Cache.end())
    cacheAnnotationFromMD(m, gv);
  else if (Cache[m].find(gv) == Cache[m].end())
    cacheAnnotationFromMD(m, gv);
  if (Cache[m][gv].find(prop) == Cache[m][gv].end())
    return false;
  retval = Cache[m][gv][prop];
  return true;
}

bool NVVMAnnotations::isTexture(const Value &val) const {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (findOneNVVMAnnotation(
            gv, PropertyAnnotationNames[PROPERTY_ISTEXTURE],
            annot)) {
      assert((annot == 1) && "Unexpected annotation on a texture symbol");
      return true;
    }
  }
  return false;
}

bool NVVMAnnotations::isSurface(const Value &val) const {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (findOneNVVMAnnotation(
            gv, PropertyAnnotationNames[PROPERTY_ISSURFACE],
            annot)) {
      assert((annot == 1) && "Unexpected annotation on a surface symbol");
      return true;
    }
  }
  return false;
}

bool NVVMAnnotations::isSampler(const Value &val) const {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (findOneNVVMAnnotation(
            gv, PropertyAnnotationNames[PROPERTY_ISSAMPLER],
            annot)) {
      assert((annot == 1) && "Unexpected annotation on a sampler symbol");
      return true;
    }
  }
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (findAllNVVMAnnotation(
            func, PropertyAnnotationNames[PROPERTY_ISSAMPLER],
            annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool NVVMAnnotations::isImageReadOnly(const Value &val) const {
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (findAllNVVMAnnotation(func,
                              PropertyAnnotationNames[
                                  PROPERTY_ISREADONLY_IMAGE_PARAM],
                              annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool NVVMAnnotations::isImageWriteOnly(const Value &val) const {
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (findAllNVVMAnnotation(func,
                              PropertyAnnotationNames[
                                  PROPERTY_ISWRITEONLY_IMAGE_PARAM],
                              annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool NVVMAnnotations::isImageReadWrite(const llvm::Value &val) const {
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (findAllNVVMAnnotation(func,
                              PropertyAnnotationNames[
                                  PROPERTY_ISREADWRITE_IMAGE_PARAM],
                              annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool NVVMAnnotations::isImage(const llvm::Value &val) const {
  return isImageReadOnly(val) || isImageWriteOnly(val) ||
         isImageReadWrite(val);
}

bool NVVMAnnotations::isManaged(const llvm::Value &val) const {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (findOneNVVMAnnotation(gv,
                              PropertyAnnotationNames[PROPERTY_MANAGED],
                              annot)) {
      assert((annot == 1) && "Unexpected annotation on a managed symbol");
      return true;
    }
  }
  return false;
}

bool NVVMAnnotations::getMaxNTIDx(const Function &F, unsigned &x) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_MAXNTID_X], x));
}

bool NVVMAnnotations::getMaxNTIDy(const Function &F, unsigned &y) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_MAXNTID_Y], y));
}

bool NVVMAnnotations::getMaxNTIDz(const Function &F, unsigned &z) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_MAXNTID_Z], z));
}

bool NVVMAnnotations::getReqNTIDx(const Function &F, unsigned &x) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_REQNTID_X], x));
}

bool NVVMAnnotations::getReqNTIDy(const Function &F, unsigned &y) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_REQNTID_Y], y));
}

bool NVVMAnnotations::getReqNTIDz(const Function &F, unsigned &z) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_REQNTID_Z], z));
}

bool NVVMAnnotations::getMinCTASm(const Function &F, unsigned &x) const {
  return (findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_MINNCTAPERSM], x));
}

bool NVVMAnnotations::isKernelFunction(const Function &F) const {
  unsigned x = 0;
  bool retval = findOneNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_ISKERNEL_FUNCTION], x);
  if (retval == false) {
    // There is no NVVM metadata, check the calling convention
    if (F.getCallingConv() == CallingConv::PTX_Kernel)
      return true;
    else
      return false;
  }
  return (x == 1);
}

bool NVVMAnnotations::getAlign(const Function &F, unsigned index, unsigned &align) const {
  std::vector<unsigned> Vs;
  bool retval = findAllNVVMAnnotation(
      &F, PropertyAnnotationNames[PROPERTY_ALIGN], Vs);
  if (retval == false)
    return false;
  for (int i = 0, e = Vs.size(); i < e; i++) {
    unsigned v = Vs[i];
    if ((v >> 16) == index) {
      align = v & 0xFFFF;
      return true;
    }
  }
  return false;
}

bool getAlign(const CallInst &I, unsigned index, unsigned &align) {
  if (MDNode *alignNode = I.getMetadata("callalign")) {
    for (int i = 0, n = alignNode->getNumOperands(); i < n; i++) {
      if (const ConstantInt *CI =
              dyn_cast<ConstantInt>(alignNode->getOperand(i))) {
        unsigned v = CI->getZExtValue();
        if ((v >> 16) == index) {
          align = v & 0xFFFF;
          return true;
        }
        if ((v >> 16) > index) {
          return false;
        }
      }
    }
  }
  return false;
}

std::string getTextureName(const Value &val) {
    assert(val.hasName() && "Found texture variable with no name");
    return val.getName();
}

std::string getSurfaceName(const Value &val) {
    assert(val.hasName() && "Found surface variable with no name");
    return val.getName();
}

std::string getSamplerName(const Value &val) {
    assert(val.hasName() && "Found sampler variable with no name");
    return val.getName();
}

bool isBarrierIntrinsic(Intrinsic::ID id) {
  if ((id == Intrinsic::nvvm_barrier0) ||
      (id == Intrinsic::nvvm_barrier0_popc) ||
      (id == Intrinsic::nvvm_barrier0_and) ||
      (id == Intrinsic::nvvm_barrier0_or) ||
      (id == Intrinsic::cuda_syncthreads))
    return true;
  return false;
}

// Interface for checking all memory space transfer related intrinsics
bool isMemorySpaceTransferIntrinsic(Intrinsic::ID id) {
  if (id == Intrinsic::nvvm_ptr_local_to_gen ||
      id == Intrinsic::nvvm_ptr_shared_to_gen ||
      id == Intrinsic::nvvm_ptr_global_to_gen ||
      id == Intrinsic::nvvm_ptr_constant_to_gen ||
      id == Intrinsic::nvvm_ptr_gen_to_global ||
      id == Intrinsic::nvvm_ptr_gen_to_shared ||
      id == Intrinsic::nvvm_ptr_gen_to_local ||
      id == Intrinsic::nvvm_ptr_gen_to_constant ||
      id == Intrinsic::nvvm_ptr_gen_to_param) {
    return true;
  }

  return false;
}

// consider several special intrinsics in striping pointer casts, and
// provide an option to ignore GEP indicies for find out the base address only
// which could be used in simple alias disambigurate.
const Value *
skipPointerTransfer(const Value *V, bool ignore_GEP_indices) {
  V = V->stripPointerCasts();
  while (true) {
    if (const IntrinsicInst *IS = dyn_cast<IntrinsicInst>(V)) {
      if (isMemorySpaceTransferIntrinsic(IS->getIntrinsicID())) {
        V = IS->getArgOperand(0)->stripPointerCasts();
        continue;
      }
    } else if (ignore_GEP_indices)
      if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
        V = GEP->getPointerOperand()->stripPointerCasts();
        continue;
      }
    break;
  }
  return V;
}

// consider several special intrinsics in striping pointer casts, and
// - ignore GEP indicies for find out the base address only, and
// - tracking PHINode
// which could be used in simple alias disambigurate.
const Value *
skipPointerTransfer(const Value *V, std::set<const Value *> &processed) {
  if (processed.find(V) != processed.end())
    return NULL;
  processed.insert(V);

  const Value *V2 = V->stripPointerCasts();
  if (V2 != V && processed.find(V2) != processed.end())
    return NULL;
  processed.insert(V2);

  V = V2;

  while (true) {
    if (const IntrinsicInst *IS = dyn_cast<IntrinsicInst>(V)) {
      if (isMemorySpaceTransferIntrinsic(IS->getIntrinsicID())) {
        V = IS->getArgOperand(0)->stripPointerCasts();
        continue;
      }
    } else if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand()->stripPointerCasts();
      continue;
    } else if (const PHINode *PN = dyn_cast<PHINode>(V)) {
      if (V != V2 && processed.find(V) != processed.end())
        return NULL;
      processed.insert(PN);
      const Value *common = 0;
      for (unsigned i = 0; i != PN->getNumIncomingValues(); ++i) {
        const Value *pv = PN->getIncomingValue(i);
        const Value *base = skipPointerTransfer(pv, processed);
        if (base) {
          if (common == 0)
            common = base;
          else if (common != base)
            return PN;
        }
      }
      if (common == 0)
        return PN;
      V = common;
    }
    break;
  }
  return V;
}

// The following are some useful utilities for debuggung

BasicBlock *getParentBlock(Value *v) {
  if (BasicBlock *B = dyn_cast<BasicBlock>(v))
    return B;

  if (Instruction *I = dyn_cast<Instruction>(v))
    return I->getParent();

  return 0;
}

Function *getParentFunction(Value *v) {
  if (Function *F = dyn_cast<Function>(v))
    return F;

  if (Instruction *I = dyn_cast<Instruction>(v))
    return I->getParent()->getParent();

  if (BasicBlock *B = dyn_cast<BasicBlock>(v))
    return B->getParent();

  return 0;
}

// Dump a block by name
void dumpBlock(Value *v, char *blockName) {
  Function *F = getParentFunction(v);
  if (F == 0)
    return;

  for (Function::iterator it = F->begin(), ie = F->end(); it != ie; ++it) {
    BasicBlock *B = it;
    if (strcmp(B->getName().data(), blockName) == 0) {
      B->dump();
      return;
    }
  }
}

// Find an instruction by name
Instruction *getInst(Value *base, char *instName) {
  Function *F = getParentFunction(base);
  if (F == 0)
    return 0;

  for (inst_iterator it = inst_begin(F), ie = inst_end(F); it != ie; ++it) {
    Instruction *I = &*it;
    if (strcmp(I->getName().data(), instName) == 0) {
      return I;
    }
  }

  return 0;
}

// Dump an instruction by nane
void dumpInst(Value *base, char *instName) {
  Instruction *I = getInst(base, instName);
  if (I)
    I->dump();
}

// Dump an instruction and all dependent instructions
void dumpInstRec(Value *v, std::set<Instruction *> *visited) {
  if (Instruction *I = dyn_cast<Instruction>(v)) {

    if (visited->find(I) != visited->end())
      return;

    visited->insert(I);

    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      dumpInstRec(I->getOperand(i), visited);

    I->dump();
  }
}

// Dump an instruction and all dependent instructions
void dumpInstRec(Value *v) {
  std::set<Instruction *> visited;

  //BasicBlock *B = getParentBlock(v);

  dumpInstRec(v, &visited);
}

// Dump the parent for Instruction, block or function
void dumpParent(Value *v) {
  if (Instruction *I = dyn_cast<Instruction>(v)) {
    I->getParent()->dump();
    return;
  }

  if (BasicBlock *B = dyn_cast<BasicBlock>(v)) {
    B->getParent()->dump();
    return;
  }

  if (Function *F = dyn_cast<Function>(v)) {
    F->getParent()->dump();
    return;
  }
}

}  // llvm

