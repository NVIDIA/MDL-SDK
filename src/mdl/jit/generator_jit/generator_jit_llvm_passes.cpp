/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include "pch.h"

#include <string>
#include <set>

#include <cstdlib>
#include <cstdio>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/PassRegistry.h>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

#include "generator_jit_llvm_passes.h"

namespace {

using namespace llvm;

/// A very simple pass that removes unused function from libdevice.
/// All functions starting with "__nv_" and marked with AlwaysInline
/// will be removed.
class DeleteUnusedLibDevice : public llvm::ModulePass
{
public:
    /// Run the pass on the given module.
    bool runOnModule(llvm::Module &M) MDL_FINAL {
        bool changed = false;
        for (llvm::Module::iterator it(M.begin()), end(M.end()); it != end;) {
            llvm::Function *F = &*it;
            ++it;

            // intrinsics are not in the call graph, do not try to remove them neither remove
            // address taken or used functions, which might occur if optimizations are set to
            // a lower level.
            if (!F->isIntrinsic() && !F->hasAddressTaken() && F->use_empty() && isLibDeviceFunc(F))
            {
                M.getFunctionList().remove(F);
                delete F;
                changed = true;
            }
        }
        return changed;
    }

    /// Check if the given Function is from libDevice.
    bool isLibDeviceFunc(llvm::Function const *F) {
        llvm::AttributeList Attr = F->getAttributes();
        if (!Attr.hasAttribute(llvm::AttributeList::FunctionIndex, llvm::Attribute::AlwaysInline))
            return false;

        if (!F->getName().startswith("__nv_"))
            return false;

        return true;
    }

    /// Default Constructor.
    DeleteUnusedLibDevice() : ModulePass(ID) {}

public:
    static char ID; // Class identification, replacement for typeinfo
};

char DeleteUnusedLibDevice::ID = 0;
char &DeleteUnusedLibDeviceID = DeleteUnusedLibDevice::ID;

}  // anonymous

namespace llvm {

/// Creates our pass.
llvm::ModulePass *createDeleteUnusedLibDevicePass() {
    return new DeleteUnusedLibDevice();
}

}  // llvm

INITIALIZE_PASS(DeleteUnusedLibDevice, "delete-unused-libdevice",
              "Delete unused LibDevice functions from a module", false, false)
