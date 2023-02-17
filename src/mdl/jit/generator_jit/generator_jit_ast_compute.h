/******************************************************************************
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include <map>

#include <llvm/Pass.h>

namespace mi {
namespace mdl {
// forward
class Type_mapper;
}
}

namespace llvm {
namespace sl {

class Region;
class StructuredFunction;

/// This pass computes a structured control flow above the LLVM control flow.
/// Irreducible control flow is removed using "Controlled Node Splitting".
class StructuredControlFlowPass : public ModulePass
{
public:
    static char ID;

public:
    /// Constructor.
    ///
    /// \param type_mapper  the MDL type mapper.
    explicit StructuredControlFlowPass(
        mi::mdl::Type_mapper &type_mapper);

    /// Destructor.
    ~StructuredControlFlowPass() override;

    void getAnalysisUsage(
        AnalysisUsage &usage) const final;

    StringRef getPassName() const final {
        return "Structured Control Flow";
    }

    /// Process a whole module.
    bool runOnModule(Module &M) final;

    /// Get the structured function for the given LLVM function.
    /// Returns nullptr, if the LLVM function is unknown.
    StructuredFunction const *getStructuredFunction(
        llvm::Function *func) const
    {
        auto it = m_structured_function_map.find(func);
        if (it == m_structured_function_map.end()) {
            return nullptr;
        }
        return it->second;
    }

private:

private:
    /// The MDL type mapper.
    mi::mdl::Type_mapper &m_type_mapper;

    /// Map from LLVM functions to structured functions.
    std::map<llvm::Function *, StructuredFunction *> m_structured_function_map;
};

/// Creates a new AST compute pass.
///
/// \param type_mapper  the MDL type mapper.
Pass *createASTComputePass(mi::mdl::Type_mapper &type_mapper);


/// This pass ensures that loops only have one exit node as preparation
// for the StructuredControlFlowPass.
class LoopExitEnumerationPass : public FunctionPass
{
public:
    static char ID;

public:
    LoopExitEnumerationPass();

    void getAnalysisUsage(AnalysisUsage &usage) const final;

    bool runOnFunction(Function& function) final;

    StringRef getPassName() const final {
        return "Loop exit enumeration";
    }
};

Pass *createLoopExitEnumerationPass();

/// This pass converts switches to if cascades.
class UnswitchPass : public FunctionPass
{
public:
    static char ID;

public:
    UnswitchPass();

    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    bool runOnFunction(Function &function) final;

    StringRef getPassName() const final {
        return "Unswitch";
    }

private:
    /// Fixes the PHI nodes in the given block, when the predecessor old_pred is replaced
    /// by new_pred.
    static void fixPhis(
        BasicBlock *bb,
        BasicBlock *old_pred,
        BasicBlock *new_pred);

};

/// Creates the "unswitch" pass.
Pass *createUnswitchPass();

} // sl;
} // llvm;
