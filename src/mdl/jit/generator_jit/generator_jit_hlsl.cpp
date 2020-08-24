/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <llvm/ADT/SetVector.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/Support/SourceMgr.h>

#include "generator_jit_llvm.h"
#include "generator_jit_ast_compute.h"
#include "generator_jit_cns_pass.h"
#include "generator_jit_hlsl_writer.h"
#include "generator_jit_streams.h"

namespace mi {
namespace mdl {

/// This pass removes any pointer selects.
class HandlePointerSelects : public llvm::FunctionPass
{
public:
    static char ID;

public:
    HandlePointerSelects();

    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    bool runOnFunction(llvm::Function &function) final;

    void handlePointerSelect(
        llvm::SelectInst &select,
        llvm::SmallVector<llvm::Instruction *, 4> &to_remove);

    llvm::StringRef getPassName() const final {
        return "HandlePointerSelects";
    }
};

HandlePointerSelects::HandlePointerSelects()
: FunctionPass( ID )
{
}

void HandlePointerSelects::getAnalysisUsage(llvm::AnalysisUsage &usage) const
{
}

void HandlePointerSelects::handlePointerSelect(
    llvm::SelectInst &select,
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

    llvm::Value *cond = select.getCondition();
    llvm::Value *ptr_true = select.getTrueValue();
    llvm::Value *ptr_false = select.getFalseValue();

    llvm::IRBuilder<> ir_builder(&select);

    for (auto user : select.users()) {
        if (llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(user)) {
            llvm::TerminatorInst *then_term = nullptr;
            llvm::TerminatorInst *else_term = nullptr;

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
                llvm::TerminatorInst *then_term = nullptr;
                llvm::TerminatorInst *else_term = nullptr;

                SplitBlockAndInsertIfThenElse(cond, store, &then_term, &else_term);

                ir_builder.SetInsertPoint(then_term);
                ir_builder.CreateStore(store->getValueOperand(), ptr_true);

                ir_builder.SetInsertPoint(else_term);
                ir_builder.CreateStore(store->getValueOperand(), ptr_false);

                to_remove.push_back(store);
            }
        }
    }
}

bool HandlePointerSelects::runOnFunction(llvm::Function &function)
{
    bool changed = false;
    llvm::SmallVector<llvm::Instruction *, 4> to_remove;

    // always insert new blocks after "end" to avoid iterating over them
    for (llvm::Function::iterator BI = function.begin(); BI != function.end(); ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::SelectInst *select = llvm::dyn_cast<llvm::SelectInst>(II)) {
                if (llvm::isa<llvm::PointerType>(select->getType()))
                    handlePointerSelect(*select, to_remove);
            }
        }
    }

    if (!to_remove.empty()) {
        for (llvm::Instruction *inst : to_remove) {
            inst->eraseFromParent();
        }
        changed = true;
    }

    return changed;
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
    enum ValueCompareResult
    {
        VCR_DIFFERENT,
        VCR_SAME,
        VCR_SAME_GEP_CHAIN   // needs to rebuild a new GEP from a chain of GEP instructions
    };

public:
    static char ID;

public:
    RemovePointerPHIs();

    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    bool runOnFunction(llvm::Function &function) final;

    llvm::StringRef getPassName() const final {
        return "RemovePointerPHIs";
    }

private:
    /// Skip all bitcasts.
    static llvm::Value *skip_bitcasts(llvm::Value *op) {
        while (llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(op)) {
            op = cast->getOperand(0);
        }
        return op;
    }

    /// Check, whether the given values represent the same address, and whether
    /// a chain of GEP instructions is involved.
    static ValueCompareResult compare_values(llvm::Value *a_val, llvm::Value *b_val);

    /// Check, whether all members of the given PHI equivalence class represent the same address,
    /// and whether a chain of GEP instructions is involved.
    ///
    /// \return The one address or nullptr, if the PHIs point to different addresses
    static llvm::Value *get_unique_phi_value(
        llvm::EquivalenceClasses<llvm::PHINode *> const &phi_classes,
        llvm::EquivalenceClasses<llvm::PHINode *>::iterator cur_phi_class,
        bool *out_need_clone_gep_chain);

    /// Clone a chain of GEP instructions.
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

void RemovePointerPHIs::getAnalysisUsage(llvm::AnalysisUsage &usage) const
{
}

// Check, whether the given values represent the same address, and whether
// a chain of GEP instructions is involved.
RemovePointerPHIs::ValueCompareResult RemovePointerPHIs::compare_values(
    llvm::Value *a_val,
    llvm::Value *b_val)
{
    if (a_val == b_val)
        return VCR_SAME;

    llvm::Instruction *a = llvm::dyn_cast<llvm::Instruction>(a_val);
    llvm::Instruction *b = llvm::dyn_cast<llvm::Instruction>(b_val);
    if (a == nullptr || b == nullptr ||
            a->getOpcode() != b->getOpcode() ||
            a->getNumOperands() != b->getNumOperands())
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
            if (llvm::isa<llvm::PHINode>(op))
                continue;

            // treat as not different
            if (llvm::isa<llvm::UndefValue>(op))
                continue;

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

    llvm::SmallVector<llvm::Value *, 8> idxs;
    llvm::Value *cur_base = inst;
    while (llvm::GetElementPtrInst *gep =
            llvm::dyn_cast<llvm::GetElementPtrInst>(cur_base)) {
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
            } else
                idxs.insert(idxs.begin(), cur_op);
        }

        cur_base = gep->getOperand(0);
    }

    return llvm::GetElementPtrInst::Create(nullptr, cur_base, idxs);
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
            if (!I->isLeader())
                continue;

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
            // Try to move GEP, bitcasts and loads over the PHIs
            llvm::SetVector<llvm::PHINode *> worklist;
            worklist.insert(phi_classes.member_begin(I), phi_classes.member_end());
            while (!worklist.empty()) {
                llvm::PHINode *phi = worklist.pop_back_val();
                for (auto phi_user : phi->users()) {
                    llvm::PHINode *new_phi;
                    if (llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(phi_user)) {
                        // create a load in each predecessor block, a PHI in the current block
                        // and replace all users of the load by the new PHI.
                        // for undef pointers provide an undef to the PHI instead of a load
                        new_phi = llvm::PHINode::Create(
                            load->getType(), phi->getNumIncomingValues(), "rmemload_phi",
                            &phi->getParent()->front());
                        for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
                            llvm::Value *cur_ptr = phi->getIncomingValue(i);
                            llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
                            llvm::Value *cur_val;
                            if (llvm::isa<llvm::UndefValue>(cur_ptr))
                                cur_val = llvm::UndefValue::get(load->getType());
                            else
                                cur_val = new llvm::LoadInst(
                                    cur_ptr, "rmemload", cur_block->getTerminator());
                            new_phi->addIncoming(cur_val, cur_block);
                        }
                    } else if (llvm::GetElementPtrInst *gep =
                            llvm::dyn_cast<llvm::GetElementPtrInst>(phi_user))
                    {
                        // create a GEP in each predecessor block, and a PHI in the current block
                        // and replace all users of the GEP by the new PHI
                        // for undef pointers provide an undef to the PHI instead of a GEP
                        new_phi = llvm::PHINode::Create(
                            gep->getType(), phi->getNumIncomingValues(), "rmemgep_phi",
                            &phi->getParent()->front());
                        for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
                            llvm::Value *cur_ptr = phi->getIncomingValue(i);
                            llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
                            llvm::Value *cur_val;
                            if (llvm::isa<llvm::UndefValue>(cur_ptr))
                                cur_val = llvm::UndefValue::get(load->getType());
                            else {
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
                    } else if (llvm::CastInst *castinst =
                            llvm::dyn_cast<llvm::CastInst>(phi_user))
                    {
                        // create a cast in each predecessor block, and a PHI in the current block
                        // and replace all users of the cast by the new PHI
                        // for undef pointers provide an undef to the PHI instead of a cast
                        new_phi = llvm::PHINode::Create(
                            castinst->getType(), phi->getNumIncomingValues(), "rmemcast_phi",
                            &phi->getParent()->front());
                        for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
                            llvm::Value *cur_ptr = phi->getIncomingValue(i);
                            llvm::BasicBlock *cur_block = phi->getIncomingBlock(i);
                            llvm::Value *cur_val;
                            if (llvm::isa<llvm::UndefValue>(cur_ptr))
                                cur_val = llvm::UndefValue::get(load->getType());
                            else {
                                cur_val = llvm::CastInst::Create(
                                    castinst->getOpcode(),
                                    cur_ptr,
                                    castinst->getType(),
                                    "rmemcast",
                                    cur_block->getTerminator());
                            }
                            new_phi->addIncoming(cur_val, cur_block);
                        }
                    } else
                        continue;

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
                if (phi->user_empty())
                    to_remove.push_back(phi);
            }
        }

        // remove instructions while no iterators are active
        if (!to_remove.empty()) {
            for (llvm::Instruction *inst : to_remove) {
                if (llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(inst)) {
                    ptr_phis.remove(phi);
                    try_again = true;
                }
                inst->eraseFromParent();
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

// Compile the given module into HLSL code.
void LLVM_code_generator::hlsl_compile(llvm::Module *module, string &code)
{
    std::unique_ptr<llvm::Module> loaded_module;

    if (char const *load_module_name = getenv("MI_MDL_HLSL_LOAD_MODULE")) {
        llvm::SMDiagnostic err;
        loaded_module = std::move(llvm::parseIRFile(load_module_name, err, m_llvm_context));

        module = loaded_module.get();

        llvm::dbgs() << "\nhlsl_compile: Loaded module from \"" << load_module_name << "\"!\n";
    }

    if (char const *save_module_name = getenv("MI_MDL_HLSL_SAVE_MODULE")) {
        std::error_code ec;
        llvm::raw_fd_ostream file(save_module_name, ec);

        llvm::legacy::PassManager mpm;
        mpm.add(llvm::createPrintModulePass(file));
        mpm.run(*module);

        llvm::dbgs() << "\nhlsl_compile: Saved input module to \"" << save_module_name << "\".\n";
    }

    {
        String_stream_writer out(code);

        llvm::legacy::PassManager mpm;
        mpm.add(llvm::createCFGSimplificationPass(     // must be executed before CNS
            1, false, false, true, false, /*AvoidPointerPHIs=*/ true));
        mpm.add(llvm::hlsl::createControlledNodeSplittingPass());  // resolve irreducible CF
        mpm.add(llvm::createCFGSimplificationPass(     // eliminate dead blocks created by CNS
            1, false, false, true, false, /*AvoidPointerPHIs=*/ true));
        mpm.add(llvm::hlsl::createUnswitchPass());     // get rid of all switch instructions
        mpm.add(llvm::createLoopSimplifyCFGPass());    // ensure all exit blocks are dominated by
                                                       // the loop header
        mpm.add(llvm::hlsl::createLoopExitEnumerationPass());  // ensure all loops have <= 1 exits
        mpm.add(llvm::hlsl::createUnswitchPass());     // get rid of all switch instructions
                                                       // introduced by the loop exit enumeration
        mpm.add(createRemovePointerPHIsPass());
        mpm.add(createHandlePointerSelectsPass());
        mpm.add(llvm::hlsl::createASTComputePass(m_type_mapper));
        mpm.add(hlsl::createHLSLWriterPass(
            get_allocator(),
            m_type_mapper,
            out,
            m_num_texture_spaces,
            m_num_texture_results,
            m_enable_full_debug,
            m_link_libbsdf_df_handle_slot_mode,
            m_exported_func_list));
        mpm.run(*module);
    }
}

// Get the HLSL function suffix for the texture type in the first parameter of the given
// function definition.
char const *LLVM_code_generator::get_hlsl_tex_type_func_suffix(IDefinition const *tex_func_def)
{
    if (tex_func_def->get_semantics() == IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID) {
        // we have only one tex::isvalid() function in the HLSL runtime
        return "";
    }
    IType_function const *func_type = cast<IType_function>(tex_func_def->get_type());

    IType const *tex_param_type;
    ISymbol const *tex_sym;
    func_type->get_parameter(0, tex_param_type, tex_sym);

    switch (cast<IType_texture>(tex_param_type->skip_type_alias())->get_shape()) {
    case IType_texture::TS_2D:        return "_2d";
    case IType_texture::TS_3D:        return "_3d";
    case IType_texture::TS_CUBE:      return "_cube";
    case IType_texture::TS_PTEX:      return "_ptex";
    case IType_texture::TS_BSDF_DATA: return "_3d";  // map to 3D
    }

    MDL_ASSERT(!"Unexpected texture shape");
    return "";
}

// Get the intrinsic LLVM function for a MDL function for HLSL code.
llvm::Function *LLVM_code_generator::get_hlsl_intrinsic_function(
    IDefinition const *def,
    bool               return_derivs)
{
    char const *module_name = NULL;
    enum Module_enum {
        ME_OTHER,
        ME_DF,
        ME_SCENE,
        ME_TEX,
    } module = ME_OTHER;

    bool can_return_derivs = false;
    IDefinition::Semantics sema = def->get_semantics();
    if (is_tex_semantics(sema)) {
        module_name = "tex";
        module = ME_TEX;
    } else {
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            module_name = "df";
            module = ME_DF;
            break;

        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
            module_name = "scene";
            module = ME_SCENE;
            break;

        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
        case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
            can_return_derivs = true;
            module_name = "scene";
            module = ME_SCENE;
            break;

        default:
            break;
        }
    }

    // no special handling required?
    if (module_name == NULL)
        return NULL;

    Function_instance inst(get_allocator(), def, return_derivs);
    llvm::Function *func = get_function(inst);
    if (func != NULL)
        return func;

    IModule const *owner = m_compiler->find_builtin_module(string(module_name, get_allocator()));

    if (return_derivs && !can_return_derivs) {
        // create a wrapper which calls the non-derivative runtime function
        // and expands the result to a dual
        LLVM_context_data *ctx_data = get_or_create_context_data(
            owner, inst, module_name, /*is_prototype=*/ false);
        func = ctx_data->get_function();

        Function_context ctx(
            get_allocator(), *this, inst, func, ctx_data->get_function_flags());

        llvm::Function *runtime_func =
            get_hlsl_intrinsic_function(def, /*return_derivs=*/ false);
        llvm::SmallVector<llvm::Value *, 8> args;
        for (llvm::Function::arg_iterator ai = ctx.get_first_parameter(), ae = func->arg_end();
                ai != ae; ++ai) {
            args.push_back(ai);
        }

        llvm::Value *res = ctx->CreateCall(runtime_func, args);
        res = ctx.get_dual(res);
        ctx.create_return(res);
        return func;
    } else {
        // return the external texture runtime function of the renderer
        LLVM_context_data *ctx_data = get_or_create_context_data(
            owner, inst, module_name, /*is_prototype=*/ true);
        func = ctx_data->get_function();

        char const *func_name = def->get_symbol()->get_name();

        if (module == ME_TEX) {
            if (is_texruntime_with_derivs()) {
                bool supports_derivatives = false;

                switch (sema) {
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
                case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
                    {
                        IType const *tex_param_type = nullptr;
                        ISymbol const *tex_param_sym = nullptr;
                        as<IType_function>(def->get_type())->get_parameter(
                            0, tex_param_type, tex_param_sym);
                        if (IType_texture const *tex_type = as<IType_texture>(tex_param_type)) {
                            switch (tex_type->get_shape()) {
                            case IType_texture::TS_2D:
                                supports_derivatives = true;
                                break;
                            case IType_texture::TS_3D:
                            case IType_texture::TS_CUBE:
                            case IType_texture::TS_PTEX:
                            case IType_texture::TS_BSDF_DATA:
                                // not supported
                                break;
                            }
                        }
                        break;
                    }
                default:
                    break;
                }

                if (supports_derivatives) {
                    switch (sema) {
                    case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
                        func_name = "lookup_deriv_float";
                        break;
                    case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
                        func_name = "lookup_deriv_float2";
                        break;
                    case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
                        func_name = "lookup_deriv_float3";
                        break;
                    case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
                        func_name = "lookup_deriv_float4";
                        break;
                    case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
                        func_name = "lookup_deriv_color";
                        break;
                    default:
                        break;
                    }
                }
            }
            func->setName("tex_" + llvm::Twine(func_name) + get_hlsl_tex_type_func_suffix(def));
        } else if (module == ME_SCENE &&
                sema != mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID) {
            // the renderer runtime has an additional uniform_lookup parameter instead of
            // two distinct functions. So call them instead and create them if necessary

            bool uniform = false;
            switch (sema) {
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
                    uniform = true;
                    break;

                default:
                    break;
            }

            llvm::Function **runtime_func;
            char const *runtime_func_name;
            switch (sema) {
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
                    runtime_func = &m_hlsl_func_scene_data_lookup_int;
                    runtime_func_name = "scene_data_lookup_int";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
                    runtime_func = &m_hlsl_func_scene_data_lookup_int2;
                    runtime_func_name = "scene_data_lookup_int2";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
                    runtime_func = &m_hlsl_func_scene_data_lookup_int3;
                    runtime_func_name = "scene_data_lookup_int3";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
                    runtime_func = &m_hlsl_func_scene_data_lookup_int4;
                    runtime_func_name = "scene_data_lookup_int4";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
                    if (return_derivs) {
                        runtime_func = &m_hlsl_func_scene_data_lookup_deriv_float;
                        runtime_func_name = "scene_data_lookup_deriv_float";
                    } else {
                        runtime_func = &m_hlsl_func_scene_data_lookup_float;
                        runtime_func_name = "scene_data_lookup_float";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
                    if (return_derivs) {
                        runtime_func = &m_hlsl_func_scene_data_lookup_deriv_float2;
                        runtime_func_name = "scene_data_lookup_deriv_float2";
                    } else {
                        runtime_func = &m_hlsl_func_scene_data_lookup_float2;
                        runtime_func_name = "scene_data_lookup_float2";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
                    if (return_derivs) {
                        runtime_func = &m_hlsl_func_scene_data_lookup_deriv_float3;
                        runtime_func_name = "scene_data_lookup_deriv_float3";
                    } else {
                        runtime_func = &m_hlsl_func_scene_data_lookup_float3;
                        runtime_func_name = "scene_data_lookup_float3";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
                    if (return_derivs) {
                        runtime_func = &m_hlsl_func_scene_data_lookup_deriv_float4;
                        runtime_func_name = "scene_data_lookup_deriv_float4";
                    } else {
                        runtime_func = &m_hlsl_func_scene_data_lookup_float4;
                        runtime_func_name = "scene_data_lookup_float4";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
                    if (return_derivs) {
                        runtime_func = &m_hlsl_func_scene_data_lookup_deriv_color;
                        runtime_func_name = "scene_data_lookup_deriv_color";
                    } else {
                        runtime_func = &m_hlsl_func_scene_data_lookup_color;
                        runtime_func_name = "scene_data_lookup_color";
                    }
                    break;

                default:
                    MDL_ASSERT(!"unexpected semantics");
                    return nullptr;
            }

            if (*runtime_func == nullptr) {
                // we need to create the runtime function prototype:
                // just take the type of the current function and add a bool to the end
                llvm::FunctionType *mdl_func_type = func->getFunctionType();
                llvm::SmallVector<llvm::Type *, 4> arg_types(
                    mdl_func_type->param_begin(), mdl_func_type->param_end());
                arg_types.push_back(m_type_mapper.get_bool_type());
                *runtime_func = llvm::Function::Create(
                    llvm::FunctionType::get(mdl_func_type->getReturnType(), arg_types, false),
                    llvm::GlobalValue::ExternalLinkage,
                    runtime_func_name,
                    m_module);
            }

            // call runtime function with additional uniform parameter depending on semantics
            Function_context ctx(
                get_allocator(), *this, inst, func, ctx_data->get_function_flags());

            llvm::SmallVector<llvm::Value *, 4> args;
            for (llvm::Function::arg_iterator ai = func->arg_begin(), ae = func->arg_end();
                    ai != ae; ++ai) {
                args.push_back(ai);
            }
            args.push_back(ctx.get_constant(uniform));

            llvm::Value *res = ctx->CreateCall(*runtime_func, args);
            ctx.create_return(res);
            return func;
        } else {
            func->setName(llvm::Twine(module_name) + "_" + func_name);
        }
        return func;
    }
}

}  // mdl
}  // mi
