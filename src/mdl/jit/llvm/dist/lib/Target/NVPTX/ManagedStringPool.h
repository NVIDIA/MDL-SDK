//===-- ManagedStringPool.h - Managed String Pool ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The strings allocated from a managed string pool are owned by the string
// pool and will be deleted together with the managed string pool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MANAGED_STRING_H
#define LLVM_SUPPORT_MANAGED_STRING_H

#include "llvm/ADT/SmallVector.h"
#include <string>

namespace llvm {

/// ManagedStringPool - The strings allocated from a managed string pool are
/// owned by the string pool and will be deleted together with the managed
/// string pool.
class ManagedStringPool {
  SmallVector<MISTD::string *, 8> Pool;

public:
  ManagedStringPool() {}
  ~ManagedStringPool() {
    SmallVectorImpl<MISTD::string *>::iterator Current = Pool.begin();
    while (Current != Pool.end()) {
      delete *Current;
      Current++;
    }
  }

  MISTD::string *getManagedString(const char *S) {
    MISTD::string *Str = new MISTD::string(S);
    Pool.push_back(Str);
    return Str;
  }
};

}

#endif
