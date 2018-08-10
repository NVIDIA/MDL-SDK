//===- llvm/Support/Unix/Debug.inc ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX extra debug support.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

// no extra debug stream on Unix.
namespace llvm {
  /// extra_dbgs - Return errs().
  raw_ostream &extra_dbgs() {
    return errs();
  }
}
