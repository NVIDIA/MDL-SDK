/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <llvm/ADT/EquivalenceClasses.h>

#if defined(__GNUC__) && (__GNUC__ >= 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <llvm/ADT/SetVector.h>
#if defined(__GNUC__) && (__GNUC__ >= 7)
#pragma GCC diagnostic pop
#endif

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>

#include <mdl/compiler/compilercore/compilercore_tools.h>

#include "generator_jit_llvm.h"
#include "generator_jit_sl_passes.h"

namespace llvm {
namespace sl {

/// This pass removes any pointer selects.
/// Basically, it transforms
///
///   *(cond ? ptr1 : ptr2)
///
/// into
///
///   cond ? *ptr1 : *ptr2
///
/// This operation is safe  general, because the single Load/Store
/// is replaced by an if.
class HandlePointerSelects : public llvm::FunctionPass
{
public:
    static char ID;

public:
    /// Constructor.
    HandlePointerSelects();

    /// Process a function.
    ///
    /// \param function  the function to process
    ///
    /// \return true if the IR was modified
    bool runOnFunction(llvm::Function &function) final;

    /// Return a nice clean name for a pass.
    llvm::StringRef getPassName() const final;

private:
    /// Transform an identified Select instruction.
    ///
    /// \param select      the Select instruction
    /// \param to_remove   list of instructions to remove AFTER the transformation
    ///
    /// \return false if the select instruction could not be removed
    bool handlePointerSelect(
        llvm::SelectInst                          &select,
        llvm::SmallVector<llvm::Instruction *, 4> &to_remove);

private:
    std::queue<llvm::SelectInst *> m_queue;
};

// Constructor.
HandlePointerSelects::HandlePointerSelects()
: FunctionPass( ID )
, m_queue()
{
}

// Transform an identified Select instruction.
bool HandlePointerSelects::handlePointerSelect(
    llvm::SelectInst                          &select,
    llvm::SmallVector<llvm::Instruction *, 4> &to_remove)
{
    // %ptr_x = select %cond, %ptr_A, %ptr_B
    // load %ptr_x  / store %val, %ptr_x
    //
    // ->
    //
    // br %cond, case_A, case_B
    //
    // case_A:
    //    load/store %ptr_A
    //    br ...
    //
    // case_B:
    //    load/store %ptr_B
    //    br ...

    llvm::Value *cond      = select.getCondition();
    llvm::Value *ptr_true  = select.getTrueValue();
    llvm::Value *ptr_false = select.getFalseValue();

    llvm::IRBuilder<> ir_builder(&select);

    bool all_users = true;
    for (auto user : select.users()) {
        if (llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(user)) {
            llvm::Instruction *then_term = nullptr;
            llvm::Instruction *else_term = nullptr;

            SplitBlockAndInsertIfThenElse(cond, load, &then_term, &else_term);

            ir_builder.SetInsertPoint(then_term);
            llvm::LoadInst *then_val = ir_builder.CreateLoad(ptr_true);

            ir_builder.SetInsertPoint(else_term);
            llvm::LoadInst *else_val = ir_builder.CreateLoad(ptr_false);

            ir_builder.SetInsertPoint(load);
            llvm::PHINode *phi = ir_builder.CreatePHI(load->getType(), 2);
            phi->addIncoming(then_val, then_val->getParent());
            phi->addIncoming(else_val, else_val->getParent());

            load->replaceAllUsesWith(phi);
            to_remove.push_back(load);
        } else if (llvm::StoreInst *store = llvm::dyn_cast<llvm::StoreInst>(user)) {
            if (store->getPointerOperand() == &select) {
                llvm::Instruction *then_term = nullptr;
                llvm::Instruction *else_term = nullptr;

                SplitBlockAndInsertIfThenElse(cond, store, &then_term, &else_term);

                ir_builder.SetInsertPoint(then_term);
                ir_builder.CreateStore(store->getValueOperand(), ptr_true);

                ir_builder.SetInsertPoint(else_term);
                ir_builder.CreateStore(store->getValueOperand(), ptr_false);

                to_remove.push_back(store);
            } else {
                // the address is stored here, we cannot handle it, give up
                MDL_ASSERT(!"Stores of pointers are unsupported");
                return false;
            }
        } else if (llvm::SelectInst *user_select = llvm::dyn_cast<llvm::SelectInst>(user)) {
            if (llvm::isa<llvm::PointerType>(user_select->getType())) {
                // a nested select, enqueue the first again, assume user_select will be in
                // the queue
                m_queue.push(&select);
            }
            all_users = false;
        } else {
            // the user is neither a Load nor a Store, we cannot handle it:
            // as this could lead to an endless loop IFF this is a nested select, give up
            // here
            return false;
        }
    }
    if (all_users) {
        // note: we can remove the select itself, IFF all users where removed. This is necessary,
        // so selects below will not stop because of "dead" select users
        to_remove.push_back(&select);
    }
    return true;
}

// Return a nice clean name for a pass.
llvm::StringRef HandlePointerSelects::getPassName() const {
    return "HandlePointerSelects";
}

// Process a function.
bool HandlePointerSelects::runOnFunction(llvm::Function &function)
{
    // always insert new blocks after "end" to avoid iterating over them
    for (llvm::Function::iterator BI = function.begin(); BI != function.end(); ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::SelectInst *select = llvm::dyn_cast<llvm::SelectInst>(II)) {
                if (llvm::isa<llvm::PointerType>(select->getType())) {
                    // enqueue it
                    m_queue.push(select);
                }
            }
        }
    }

    bool result = false;
    llvm::SmallVector<llvm::Instruction *, 4> to_remove;
    while (!m_queue.empty()) {
        llvm::SelectInst *select = m_queue.front();
        m_queue.pop();

        if (!handlePointerSelect(*select, to_remove)) {
            // give up
            break;
        }

        if (!to_remove.empty()) {
            // remove dead code, to move nested selects to top
            for (llvm::Instruction *inst : to_remove) {
                inst->eraseFromParent();
                result = true;
            }
            to_remove.clear();
        }
    }

    return result;
}

char HandlePointerSelects::ID = 0;

llvm::Pass *createHandlePointerSelectsPass()
{
    return new HandlePointerSelects();
}

// ------------------------------------------------------------------------------------------

/// This pass removes any memory PHIs where all input values are the same or undefined.
class RemovePointerPHIs : public llvm::FunctionPass
{
    /// compare_values() result.
    enum ValueCompareResult {
        VCR_DIFFERENT,       ///< Represent different values
        VCR_SAME,            ///< Represent the same value
        VCR_SAME_GEP_CHAIN   ///< Like VCR_SAME but needs to rebuild a
                             ///  new GEP from a chain of GEP instructions
    };

public:
    static char ID;

public:
    /// Constructor.
    RemovePointerPHIs();

    /// Process a function.
    ///
    /// \param function  the function to process
    ///
    /// \return true if the IR was modified
    bool runOnFunction(llvm::Function &function) final;

    /// Return a nice clean name for a pass.
    llvm::StringRef getPassName() const final;

private:
    /// Move a Load instruction that uses a Phi node as address over the phi by creating a Load
    /// in every predecessor and replace the original Load by a new Phi (of all those created
    /// loads).
    /// For undef pointers provide an Undef to the new Phi instead of a Load.
    llvm::PHINode *move_load_over_phi(
        llvm::LoadInst *load,
        llvm::PHINode  *phi);

    /// Move a GEP instruction that uses a Phi node as base address over the Phi by creating a GEP
    /// in every predecessor and replace the original GEP by a new Phi (of all those created
    /// GEPs).
    /// For undef pointers provide an Undef to the new Phi instead of a GEP.
    llvm::PHINode *move_gep_over_phi(
        llvm::GetElementPtrInst *gep,
        llvm::PHINode           *phi);

    /// Move a Cast instruction that uses a Phi node as argument over the Phi by creating a Cast
    /// in every predecessor and replace the original Cast by a new Phi (of all those created
    /// casts).
    /// For undef pointers provide an undef to the new Phi instead of a cast.
    llvm::PHINode *move_cast_over_phi(
        llvm::CastInst *castinst,
        llvm::PHINode  *phi);

    /// Move a Call argument that uses a Phi node over the Phi by creating a Load
    /// in every predecessor, store(new_temp, new_phi) an the begin of the Call's BB and replace the
    /// original argument by new_temp.
    /// For undef pointers provide an undef to the Phi instead of a load.
    llvm::PHINode *move_call_argument_over_phi(
        llvm::Function &function,
        llvm::PHINode *phi,
        llvm::CallInst *call);

    /// Skip all bitcasts on a value.
    static llvm::Value *skip_bitcasts(llvm::Value *op) {
        while (llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(op)) {
            op = cast->getOperand(0);
        }
        return op;
    }

    /// Check, whether the given values represent the same address, and whether
    /// a chain of GEP instructions is involved.
    static ValueCompareResult compare_values(
        llvm::Value *a_val,
        llvm::Value *b_val);

    /// Check, whether all members of the given PHI equivalence class represent the same address,
    /// and whether a chain of GEP instructions is involved.
    ///
    /// \return The one address or nullptr, if the PHIs point to different addresses
    static llvm::Value *get_unique_phi_value(
        llvm::EquivalenceClasses<llvm::PHINode *> const     &phi_classes,
        llvm::EquivalenceClasses<llvm::PHINode *>::iterator cur_phi_class,
        bool                                                *out_need_clone_gep_chain);

    /// Clone a chain of GEP instructions.
    ///
    /// \param ir_builder  the IR-builder to be used
    /// \param inst        the GEP instruction to clone
    ///
    /// \return the cloned GEP chain
    static llvm::Instruction *clone_gep_chain(
        llvm::IRBuilder<> &ir_builder,
        llvm::Instruction *inst);

    /// Create a clone of the given value in the block of the phi.
    static llvm::Value *create_phi_replacement(
        llvm::PHINode *phi,
        llvm::Value   *val,
        bool           need_clone_gep_chain);
};

RemovePointerPHIs::RemovePointerPHIs()
: FunctionPass( ID )
{
}

// Check, whether the given values represent the same address, and whether
// a chain of GEP instructions is involved.
RemovePointerPHIs::ValueCompareResult RemovePointerPHIs::compare_values(
    llvm::Value *a_val,
    llvm::Value *b_val)
{
    if (a_val == b_val) {
        return VCR_SAME;
    }

    llvm::Instruction *a = llvm::dyn_cast<llvm::Instruction>(a_val);
    llvm::Instruction *b = llvm::dyn_cast<llvm::Instruction>(b_val);
    if (a == nullptr || b == nullptr ||
            a->getOpcode() != b->getOpcode() ||
            a->getNumOperands() != b->getNumOperands())
        return VCR_DIFFERENT;

    // Two allocas are only the same, if they are identical
    if (llvm::isa<llvm::AllocaInst>(a))
        return VCR_DIFFERENT;

    bool all_ops_same = true;
    for (unsigned i = 0, n = a->getNumOperands(); i < n; ++i) {
        if (a->getOperand(i) != b->getOperand(i)) {
            all_ops_same = false;
            break;
        }
    }
    if (all_ops_same)
        return VCR_SAME;

    llvm::GetElementPtrInst *gep_1 = llvm::dyn_cast<llvm::GetElementPtrInst>(a);
    if (gep_1 == nullptr)
        return VCR_DIFFERENT;

    llvm::GetElementPtrInst *gep_2 = llvm::cast<llvm::GetElementPtrInst>(b);

    while (true) {
        for (unsigned i = 1, n = gep_1->getNumOperands(); i < n; ++i) {
            if (gep_1->getOperand(i) != gep_2->getOperand(i)) {
                return VCR_DIFFERENT;
            }
        }

        // found equal base of GEP chain? -> done, consider as same
        if (gep_1->getOperand(0) == gep_2->getOperand(0)) {
            return VCR_SAME_GEP_CHAIN;
        }

        gep_1 = llvm::dyn_cast<llvm::GetElementPtrInst>(gep_1->getOperand(0));
        gep_2 = llvm::dyn_cast<llvm::GetElementPtrInst>(gep_2->getOperand(0));
        if (gep_1 == nullptr || gep_2 == nullptr ||
                gep_1->getNumOperands() != gep_2->getNumOperands())
        {
            return VCR_DIFFERENT;
        }
    }
}

// Check, whether all members of the given PHI equivalence class represent the same address,
// and whether a chain of GEP instructions is involved.
llvm::Value *RemovePointerPHIs::get_unique_phi_value(
    llvm::EquivalenceClasses<llvm::PHINode *> const &phi_classes,
    llvm::EquivalenceClasses<llvm::PHINode *>::iterator cur_phi_class,
    bool *out_need_clone_gep_chain)
{
    llvm::Value *phi_value = nullptr;
    *out_need_clone_gep_chain = false;

    for (auto MI = phi_classes.member_begin(cur_phi_class), ME = phi_classes.member_end();
            MI != ME; ++MI)
    {
        llvm::PHINode *phi = *MI;
        for (llvm::Value *op : phi->incoming_values()) {
            op = skip_bitcasts(op);

            // phi nodes are already handled
            if (llvm::isa<llvm::PHINode>(op)) {
                continue;
            }

            // treat as not different
            if (llvm::isa<llvm::UndefValue>(op)) {
                continue;
            }

            if (phi_value == nullptr) {
                phi_value = op;
            } else {
                ValueCompareResult cmp = compare_values(phi_value, op);
                if (cmp == VCR_DIFFERENT) {
                    return nullptr;
                } else if (cmp == VCR_SAME_GEP_CHAIN) {
                    *out_need_clone_gep_chain = true;
                }
            }
        }
    }

    // if no value found, all incoming values must be undef -> take the first undef
    if (phi_value == nullptr) {
        phi_value = cur_phi_class->getData()->getIncomingValue(0);
    }
    return phi_value;
}

// Clone a chain of GEP instructions.
llvm::Instruction *RemovePointerPHIs::clone_gep_chain(
    llvm::IRBuilder<> &ir_builder,
    llvm::Instruction *inst)
{
    // build index list from end to start, adding first and last indices
    // of "adjacent" GEPs together
    MDL_ASSERT(llvm::isa<llvm::GetElementPtrInst>(inst));

    llvm::SmallVector<llvm::Value *, 8> idxs;
    llvm::Value *cur_base = inst;
    while (llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(cur_base)) {
        unsigned gep_num_ops = gep->getNumOperands();
        for (unsigned i = gep_num_ops - 1; i > 0; --i) {
            llvm::Value *cur_op = gep->getOperand(i);

            // normalize to 32-bit indices
            if (cur_op->getType()->getIntegerBitWidth() != 32) {
                cur_op = ir_builder.CreateTrunc(
                    cur_op, llvm::IntegerType::get(ir_builder.getContext(), 32));
            }

            if (i == gep_num_ops - 1 && !idxs.empty()) {
                // %Y = gep %X, a, b
                // %Z = gep %Y, c, d
                // -> %Z = gep %X, a, b + c, d
                idxs[0] = ir_builder.CreateAdd(cur_op, idxs[0]);
            } else {
                idxs.insert(idxs.begin(), cur_op);
            }
        }

        cur_base = gep->getOperand(0);
    }

    return llvm::GetElementPtrInst::Create(/*PointeeType=*/nullptr, cur_base, idxs);
}

// Create a clone of the given value in the block of the phi.
llvm::Value *RemovePointerPHIs::create_phi_replacement(
    llvm::PHINode *phi,
    llvm::Value   *val,
    bool           need_clone_gep_chain)
{
    if (llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(val)) {
        llvm::IRBuilder<> ir_builder(&*(phi->getParent()->getFirstInsertionPt()));
        llvm::Instruction *clone;

        // TODO: we do not guarantee, that all operands of the instructions actually
        //       dominate the block of the phi, where we place the clone

        if (need_clone_gep_chain) {
            clone = clone_gep_chain(ir_builder, inst);
        } else {
            clone = inst->clone();
        }
        if (inst->hasName())
            clone->setName(inst->getName() + ".rmemphi");
        phi->getParent()->getInstList().insert(ir_builder.GetInsertPoint(), clone);

        llvm::Value *clone_value = clone;
        if (phi->getType() != clone_value->getType())
            clone_value = ir_builder.CreatePointerCast(clone_value, phi->getType());

        return clone_value;
    } else {
        // not an instruction, may be a parameter
        llvm::Value *replacement = val;
        if (phi->getType() != replacement->getType()) {
            // insert cast at end of entry block
            llvm::IRBuilder<> ir_builder(
                phi->getParent()->getParent()->getEntryBlock().getTerminator());

            replacement = ir_builder.CreatePointerCast(replacement, phi->getType());
        }

        return replacement;
    }
}

// Return a nice clean name for a pass.
llvm::StringRef RemovePointerPHIs::getPassName() const {
    return "RemovePointerPHIs";
}

// Move a Load instruction that uses a Phi node as address over the phi by creating a Load
// in every predecessor and replace the original Load by a new Phi (of all those created
// loads).
// For undef pointers provide an Undef to the new Phi instead of a Load.
llvm::PHINode *RemovePointerPHIs::move_load_over_phi(
    llvm::LoadInst *load,
    llvm::PHINode  *phi)
{
    // create a load in each predecessor block, a PHI in the current block
    // and replace all users of the load by the new PHI.
    llvm::PHINode *new_phi = llvm::PHINode::Create(
        load->getType(), phi->getNumIncomingValues(), "rmemload_phi",
        &phi->getParent()->front());
    for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
        llvm::Value *cur_ptr = phi->getIncomingValue(i);
        llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
        llvm::Value *cur_val;
        if (llvm::isa<llvm::UndefValue>(cur_ptr)) {
            cur_val = llvm::UndefValue::get(load->getType());
        } else {
            cur_val = new llvm::LoadInst(
                cur_ptr->getType()->getPointerElementType(),
                cur_ptr,
                "rmemload",
                cur_block->getTerminator());
        }
        new_phi->addIncoming(cur_val, cur_block);
    }
    return new_phi;
}

// Move a GEP instruction that uses a Phi node as base address over the Phi by creating a GEP
// in every predecessor and replace the original GEP by a new phi (of all those created
// GEPs).
// For undef pointers provide an Undef to the new PHI instead of a GEP.
llvm::PHINode *RemovePointerPHIs::move_gep_over_phi(
    llvm::GetElementPtrInst *gep,
    llvm::PHINode           *phi)
{
    llvm::PHINode *new_phi = llvm::PHINode::Create(
        gep->getType(), phi->getNumIncomingValues(), "rmemgep_phi",
        &phi->getParent()->front());
    for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
        llvm::Value *cur_ptr = phi->getIncomingValue(i);
        llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
        llvm::Value *cur_val;
        if (llvm::isa<llvm::UndefValue>(cur_ptr)) {
            cur_val = llvm::UndefValue::get(gep->getType());
        } else {
            // clone the GEP and replace the ptrval operand by cur_ptr
            llvm::GetElementPtrInst *new_gep =
                llvm::cast<llvm::GetElementPtrInst>(gep->clone());
            new_gep->setOperand(0, cur_ptr);
            cur_block->getInstList().insert(
                cur_block->getTerminator()->getIterator(), new_gep);
            cur_val = new_gep;
        }
        new_phi->addIncoming(cur_val, cur_block);
    }
    return new_phi;
}

// Move a Cast instruction that uses a Phi node as base address over the Phi by creating a Cast
// in every predecessor and replace the original cast by a new phi (of all those created Casts).
// For undef pointers provide an undef to the new Phi instead of a Cast.
llvm::PHINode *RemovePointerPHIs::move_cast_over_phi(
    llvm::CastInst *castinst,
    llvm::PHINode  *phi)
{
    // create a cast in each predecessor block, and a PHI in the current block
    // and replace all users of the cast by the new PHI
    // for undef pointers provide an undef to the PHI instead of a cast
    llvm::PHINode *new_phi = llvm::PHINode::Create(
        castinst->getType(), phi->getNumIncomingValues(), "rmemcast_phi",
        &phi->getParent()->front());
    for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
        llvm::Value *cur_ptr = phi->getIncomingValue(i);
        llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
        llvm::Value *cur_val;
        if (llvm::isa<llvm::UndefValue>(cur_ptr)) {
            cur_val = llvm::UndefValue::get(castinst->getType());
        } else {
            cur_val = llvm::CastInst::Create(
                castinst->getOpcode(),
                cur_ptr,
                castinst->getType(),
                "rmemcast",
                cur_block->getTerminator());
        }
        new_phi->addIncoming(cur_val, cur_block);
    }                        return new_phi;
}

// Move a Call argument that uses a Phi node over the Phi by creating a Load
// in every predecessor, store(new_temp, new_phi) an the begin of the Call's BB and replace the
// original argument by new_temp.
// For undef pointers provide an undef to the Phi instead of a load.
llvm::PHINode *RemovePointerPHIs::move_call_argument_over_phi(
    llvm::Function &function,
    llvm::PHINode  *phi,
    llvm::CallInst *call)
{
    llvm::PHINode *new_phi = llvm::PHINode::Create(
        phi->getType()->getPointerElementType(),
        phi->getNumIncomingValues(), "rmemload_phi",
        &phi->getParent()->front());

    // handle all arguments of the call
    for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
        llvm::Value *cur_ptr = phi->getIncomingValue(i);
        llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
        llvm::Value *cur_val;
        if (llvm::isa<llvm::UndefValue>(cur_ptr)) {
            cur_val = llvm::UndefValue::get(new_phi->getType());
        } else {
            cur_val = new llvm::LoadInst(
                new_phi->getType(),
                cur_ptr,
                "rmemload",
                cur_block->getTerminator());
        }
        new_phi->addIncoming(cur_val, cur_block);
    }

    // create a new temporary
    llvm::AllocaInst *rmem_local = new llvm::AllocaInst(
        new_phi->getType(),
        function.getParent()->getDataLayout().getAllocaAddrSpace(),
        "rmemlocal",
        &*function.getEntryBlock().begin());

    // find the insert point for the new store after ALL Phis
    llvm::BasicBlock::iterator insert_point = call->getParent()->begin();
    while (llvm::isa<llvm::PHINode>(insert_point)) {
        ++insert_point;
    }
    // ... and insert the store here
    new llvm::StoreInst(new_phi, rmem_local, &*insert_point);

    // replace all occurrences of the original Phi by the new local in the call arguments
    for (unsigned i = 0, n_args = call->getNumArgOperands(); i < n_args; ++i) {
        if (call->getArgOperand(i) == phi) {
            call->setArgOperand(i, rmem_local);
        }
    }
    return new_phi;
}

// Process a function.
bool RemovePointerPHIs::runOnFunction(llvm::Function &function)
{
    bool changed = false;

    llvm::SmallSetVector<llvm::PHINode *, 8> ptr_phis;

    // collect all pointer PHIs
    for (llvm::Function::iterator BI = function.begin(); BI != function.end(); ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(II)) {
                if (llvm::isa<llvm::PointerType>(phi->getType())) {
                    ptr_phis.insert(phi);
                }
            }
        }
    }

    // loop until the set of pointer PHIs doesn't change anymore or is empty
    bool try_again = true;
    while (try_again && !ptr_phis.empty()) {
        try_again = false;

        // compute equivalence classes of phi nodes possibly pointing to the same address
        llvm::EquivalenceClasses<llvm::PHINode *> phi_classes;
        for (llvm::PHINode *phi : ptr_phis) {
            llvm::PHINode *leader = phi_classes.getOrInsertLeaderValue(phi);
            for (llvm::Value *op : phi->incoming_values()) {
                op = skip_bitcasts(op);

                if (llvm::PHINode *op_phi = llvm::dyn_cast<llvm::PHINode>(op)) {
                    llvm::PHINode *op_leader =
                        phi_classes.getOrInsertLeaderValue(op_phi);
                    phi_classes.unionSets(leader, op_leader);
                }
            }
        }

        llvm::SmallVector<llvm::Instruction *, 8> to_remove;
        for (auto I = phi_classes.begin(), E = phi_classes.end(); I != E; ++I) {
            if (!I->isLeader()) {
                continue;
            }

            // check whether all members of the current equivalence class point
            // to the same address and whether we need to clone a whole chain of GEPs
            bool need_clone_gep_chain = false;
            llvm::Value *phi_value = get_unique_phi_value(phi_classes, I, &need_clone_gep_chain);
            if (phi_value != nullptr) {
                // all pointing to same address, replace each PHI of the current equivalence class
                for (auto MI = phi_classes.member_begin(I), ME = phi_classes.member_end();
                        MI != ME; ++MI)
                {
                    llvm::PHINode *phi = *MI;
                    llvm::Value *repl = create_phi_replacement(
                        phi, phi_value, need_clone_gep_chain);
                    phi->replaceAllUsesWith(repl);
                    to_remove.push_back(phi);
                }
                continue;
            }

            // not all PHI values are the same.
            // Try to move GEP, bitcasts and loads over the PHIs.
            // For calls, load the value into a new local in the predecessor blocks and use the
            // new local as argument
            llvm::SetVector<llvm::PHINode *> worklist;
            worklist.insert(phi_classes.member_begin(I), phi_classes.member_end());
            while (!worklist.empty()) {
                llvm::PHINode *phi = worklist.pop_back_val();
                for (auto phi_user : phi->users()) {
                    llvm::PHINode *new_phi;
                    if (llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(phi_user)) {
                        new_phi = move_load_over_phi(load, phi);
                    } else if (llvm::GetElementPtrInst *gep =
                            llvm::dyn_cast<llvm::GetElementPtrInst>(phi_user))
                    {
                        new_phi = move_gep_over_phi(gep, phi);
                    } else if (llvm::CastInst *castinst =
                            llvm::dyn_cast<llvm::CastInst>(phi_user))
                    {
                        new_phi = move_cast_over_phi(castinst, phi);
                    } else if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(phi_user)) {
                        new_phi = move_call_argument_over_phi(function, phi, call);

                        // the original phi is not used anymore by the call, but not replaced!
                        continue;
                    } else {
                        continue;
                    }

                    // ... else replace the original Phi by the newly created one
                    phi_user->replaceAllUsesWith(new_phi);
                    to_remove.push_back(llvm::cast<llvm::Instruction>(phi_user));

                    // if we created a new pointer PHI, add it to the work lists
                    if (llvm::isa<llvm::PointerType>(new_phi->getType())) {
                        worklist.insert(new_phi);
                        ptr_phis.insert(new_phi);
                        try_again = true;
                    }
                }

                // remove PHI, if we were able to get rid of all users
                if (phi->user_empty()) {
                    to_remove.push_back(phi);
                }
            }
        }

        // remove instructions while no iterators are active
        if (!to_remove.empty()) {
            for (llvm::Instruction *inst : to_remove) {
                if (llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(inst)) {
                    ptr_phis.remove(phi);
                }
                inst->eraseFromParent();
                try_again = true;
            }
            changed = true;
        }
    }

    MDL_ASSERT(ptr_phis.empty() && "Not all pointer PHIs could be removed");

    return changed;
}

char RemovePointerPHIs::ID = 0;


llvm::Pass *createRemovePointerPHIsPass()
{
    return new RemovePointerPHIs();
}

// ------------------------------------------------------------------------------------------

/// This pass replaces some constructs that are not directly supported by GLSL, in particular
/// and and or on boolean vector.
class RemoveBoolVectorOPs : public llvm::FunctionPass
{
public:
    static char ID;

public:
    /// Constructor.
    RemoveBoolVectorOPs(
        mi::mdl::LLVM_code_generator &code_gen);

    /// Process a function.
    ///
    /// \param function  the function to process
    ///
    /// \return true if the IR was modified
    bool runOnFunction(llvm::Function &function) final;

    /// Return a nice clean name for a pass.
    llvm::StringRef getPassName() const final;

private:
    /// Replace an operation.
    bool replace_op(
        unsigned          opcode,
        llvm::Instruction &Inst);

private:

    /// The code generator.
    mi::mdl::LLVM_code_generator &m_code_gen;

    /// The type mapper.
    mi::mdl::Type_mapper &m_type_mapper;
};

// Constructor.
RemoveBoolVectorOPs::RemoveBoolVectorOPs(
    mi::mdl::LLVM_code_generator &code_gen)
: FunctionPass(ID)
, m_code_gen(code_gen)
, m_type_mapper(code_gen.get_type_mapper())
{
}

// Return a nice clean name for a pass.
llvm::StringRef RemoveBoolVectorOPs::getPassName() const {
    return "RemoveBoolVectorOPs";
}

// Replace an operation.
bool RemoveBoolVectorOPs::replace_op(
    unsigned          opcode,
    llvm::Instruction &Inst)
{
    llvm::Type     *type = Inst.getType();
    llvm::Function *func = m_code_gen.get_target_operator_function(opcode, type);

    if (func == nullptr) {
        return false;
    }

    llvm::BasicBlock   *curr_block = Inst.getParent();
    llvm::Value        *left       = Inst.getOperand(0);
    llvm::Value        *right      = Inst.getOperand(1);
    llvm::FunctionType *ft_type    =
        llvm::cast<llvm::FunctionType>(func->getType()->getElementType());

    llvm::Instruction *call = llvm::CallInst::Create(
        ft_type, func, { left, right }, Inst.getName(), curr_block->getTerminator());
    call->copyMetadata(Inst);

    Inst.replaceAllUsesWith(call);
    return true;
}

/// Helper class handling the module.
class Module_scope {
public:
    /// Constructor.
    Module_scope(mi::mdl::LLVM_code_generator &cg, llvm::Module *M)
    : m_cg(cg)
    , m_old_module(cg.get_llvm_module())
    {
        cg.set_llvm_module(M);
    }

    /// Destructor.
    ~Module_scope()
    {
        m_cg.set_llvm_module(m_old_module);
    }

private:
    mi::mdl::LLVM_code_generator &m_cg;
    llvm::Module                 *m_old_module;
};

// Process a function.
bool RemoveBoolVectorOPs::runOnFunction(llvm::Function &function)
{
    // Note: this pass might create new runtime functions, hence the current module must be set.
    // But we run it in later phase, so ensure it is
    Module_scope scope(m_code_gen, function.getParent());

    llvm::SmallVector<llvm::Instruction *, 8> to_remove;

    // iterate over all instructions
    for (llvm::Function::iterator BI = function.begin(); BI != function.end(); ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            llvm::Instruction &Inst = *II;

            unsigned opcode = Inst.getOpcode();
            switch (opcode) {
            case llvm::Instruction::And:
            case llvm::Instruction::Or:
                {
                    llvm::Type *type = Inst.getType();

                    if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
                        llvm::Type *et = vt->getElementType();

                        if (et == m_type_mapper.get_bool_type()) {
                            if (replace_op(opcode, Inst)) {
                                to_remove.push_back(&Inst);
                            }
                        }
                    }
                }
                break;
            default:
                break;
            }
        }
    }

    if (!to_remove.empty()) {
        for (llvm::Instruction *inst : to_remove) {
            inst->eraseFromParent();
        }
        return true;
    }

    return false;
}

char RemoveBoolVectorOPs::ID = 0;

llvm::Pass *createRemoveBoolVectorOPsPass(
    mi::mdl::LLVM_code_generator &code_gen)
{
    return new RemoveBoolVectorOPs(code_gen);
}


}  // sl
}  // llvm
