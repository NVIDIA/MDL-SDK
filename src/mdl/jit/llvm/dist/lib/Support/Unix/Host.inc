 //===- llvm/Support/Unix/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX Host support.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/ADT/StringRef.h"
#include "Unix.h"
#include <sys/utsname.h>
#include <cctype>
#include <string>
#include <cstdlib> // ::getenv

using namespace llvm;

static MISTD::string getOSVersion() {
  struct utsname info;

  if (uname(&info))
    return "";

  return info.release;
}

MISTD::string sys::getDefaultTargetTriple() {
  StringRef TargetTripleString(LLVM_DEFAULT_TARGET_TRIPLE);
  MISTD::pair<StringRef, StringRef> ArchSplit = TargetTripleString.split('-');

  // Normalize the arch, since the target triple may not actually match the target.
  MISTD::string Arch = ArchSplit.first;

  MISTD::string Triple(Arch);
  Triple += '-';
  Triple += ArchSplit.second;

  // Force i<N>86 to i386.
  if (Triple[0] == 'i' && isdigit(Triple[1]) &&
      Triple[2] == '8' && Triple[3] == '6')
    Triple[1] = '3';

  // On darwin, we want to update the version to match that of the
  // target.
  MISTD::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx != MISTD::string::npos) {
    Triple.resize(DarwinDashIdx + strlen("-darwin"));
    Triple += getOSVersion();
  }

  return Triple;
}
