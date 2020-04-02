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

/// This pass removes any memory PHIs where all input values are the same or undefined.
class HandlePointerValues : public llvm::FunctionPass
{
public:
    static char ID;

public:
    HandlePointerValues();

    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    bool runOnFunction(llvm::Function &function) final;

    void handlePointerPHI(
        llvm::PHINode &phi,
        llvm::SmallVector<llvm::Instruction *, 4> &to_remove);

    void handlePointerSelect(
        llvm::SelectInst &select,
        llvm::SmallVector<llvm::Instruction *, 4> &to_remove);

    llvm::StringRef getPassName() const final {
        return "RemoveMemPHIs";
    }
};

HandlePointerValues::HandlePointerValues()
: FunctionPass( ID )
{
}

void HandlePointerValues::getAnalysisUsage(llvm::AnalysisUsage &usage) const
{
}

void HandlePointerValues::handlePointerPHI(
    llvm::PHINode &phi,
    llvm::SmallVector<llvm::Instruction *, 4> &to_remove)
{
    bool all_values_same = true;
    bool clone_gep_chain = false;
    llvm::Value *first_value = nullptr;
    llvm::Instruction *base_inst = nullptr;
    unsigned opcode = 0;
    unsigned num_operands = 0;
    for (llvm::Value *cur_val : phi.incoming_values()) {
        if (llvm::isa<llvm::UndefValue>(cur_val))
            continue;

        // for the comparison, skip any bitcast
        if (llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(cur_val)) {
            cur_val = cast->getOperand(0);
        }

        if (first_value == nullptr) {
            first_value = cur_val;

            if ((base_inst = llvm::dyn_cast<llvm::Instruction>(cur_val)) != nullptr) {
                opcode = base_inst->getOpcode();
                num_operands = base_inst->getNumOperands();
            }
        }
        else if (first_value != cur_val) {
            // not an undef and does not point to the exact same value object?
            // compare instruction kind and operands (only one level)

            if (llvm::Instruction *cur_inst = llvm::dyn_cast<llvm::Instruction>(cur_val)) {
                if (cur_inst->getOpcode() != opcode ||
                    cur_inst->getNumOperands() != num_operands)
                {
                    all_values_same = false;
                    break;
                }

                for (unsigned i = 0; i < num_operands; ++i) {
                    if (cur_inst->getOperand(i) != base_inst->getOperand(i)) {
                        all_values_same = false;
                        break;
                    }
                }

                if (!all_values_same && llvm::isa<llvm::GetElementPtrInst>(cur_inst)) {
                    all_values_same = true;
                    llvm::GetElementPtrInst *gep_1 =
                        llvm::cast<llvm::GetElementPtrInst>(cur_inst);
                    llvm::GetElementPtrInst *gep_2 =
                        llvm::cast<llvm::GetElementPtrInst>(base_inst);
                    while (true) {
                        for (unsigned i = 1, n = gep_1->getNumOperands(); i < n; ++i) {
                            if (gep_1->getOperand(i) != gep_2->getOperand(i)) {
                                all_values_same = false;
                                break;
                            }
                        }
                        if (!all_values_same)
                            break;

                        // found equal base of GEP chain? -> done, consider as same
                        if (gep_1->getOperand(0) == gep_2->getOperand(0)) {
                            clone_gep_chain = true;
                            break;
                        }

                        gep_1 = llvm::dyn_cast<llvm::GetElementPtrInst>(
                            gep_1->getOperand(0));
                        gep_2 = llvm::dyn_cast<llvm::GetElementPtrInst>(
                            gep_2->getOperand(0));
                        if (gep_1 == nullptr || gep_2 == nullptr ||
                                gep_1->getNumOperands() != gep_2->getNumOperands()) {
                            all_values_same = false;
                            break;
                        }
                    }
                }
            } else {
                all_values_same = false;
                break;
            }
        }
        if (!all_values_same)
            break;
    }

    if (all_values_same) {
        // if no value found, all incoming values must be undef -> take the first undef
        if (first_value == nullptr) {
            first_value = phi.getIncomingValue(0);
        }

        // materialize the instruction after all PHIs in this block.
        // if any of the operands is not post-dominated by this block,
        // we also need to materials those operands
        if (llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(first_value)) {
            // TODO: handle operands not being post-dominated by this block!

            llvm::IRBuilder<> ir_builder(&*phi.getParent()->getFirstInsertionPt());
            llvm::Instruction *clone;
            if (clone_gep_chain) {
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
                                cur_op, llvm::IntegerType::get(
                                    phi.getParent()->getParent()->getContext(), 32));
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

                clone = llvm::GetElementPtrInst::Create(nullptr, cur_base, idxs);
            } else {
                clone = inst->clone();
            }
            if (inst->hasName())
                clone->setName(inst->getName() + ".rmemphi");
            phi.getParent()->getInstList().insert(ir_builder.GetInsertPoint(), clone);

            llvm::Value *clone_value = clone;
            if (phi.getType() != clone_value->getType())
                clone_value = ir_builder.CreatePointerCast(clone_value, phi.getType());

            phi.replaceAllUsesWith(clone_value);
        } else {
            // not an instruction, may be a parameter
            llvm::Value *replacement = first_value;
            if (phi.getType() != replacement->getType()) {
                // insert cast at end of entry block
                llvm::IRBuilder<> ir_builder(
                    phi.getParent()->getParent()->getEntryBlock().getTerminator());

                replacement = ir_builder.CreatePointerCast(replacement, phi.getType());
            }

            phi.replaceAllUsesWith(replacement);
        }
        to_remove.push_back(&phi);
    } else {
        MDL_ASSERT(!"Pointer PHI could not be eliminated");
    }
}

void HandlePointerValues::handlePointerSelect(
    llvm::SelectInst &select,
    llvm::SmallVector<llvm::Instruction *, 4> &to_remove)
{
    // %ptr_x = select %cond, %ptr_A, %ptr_B
    // load %ptr_x  / store %val, %ptr_x
    //
    // ->
    //
    // br %cond, case_A, case_B

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

bool HandlePointerValues::runOnFunction(llvm::Function &function)
{
    bool changed = false;
    llvm::SmallVector<llvm::Instruction *, 4> to_remove;

    // always insert new blocks after "end" to avoid iterating over them
    for (llvm::Function::iterator BI = function.begin(); BI != function.end(); ++BI) {
        for (llvm::BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
            if (llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(II)) {
                if (llvm::isa<llvm::PointerType>(phi->getType()))
                    handlePointerPHI(*phi, to_remove);
            } else if (llvm::SelectInst *select = llvm::dyn_cast<llvm::SelectInst>(II)) {
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

char HandlePointerValues::ID = 0;

llvm::Pass *createHandlePointerValuesPass()
{
    return new HandlePointerValues();
}


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
        mpm.add(createHandlePointerValuesPass());
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
    bool is_tex = false;

    IDefinition::Semantics sema = def->get_semantics();
    if (is_tex_semantics(sema)) {
        module_name = "tex";
        is_tex = true;
    } else {
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            module_name = "df";
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

    if (return_derivs) {
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

        if (is_tex) {
            if (is_texruntime_with_derivs()) {
                bool supports_derivatives = false;

                switch (def->get_semantics()) {
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
                    switch (def->get_semantics()) {
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
        } else {
            func->setName("df_" + llvm::Twine(func_name));
        }
        return func;
    }
}

}  // mdl
}  // mi
