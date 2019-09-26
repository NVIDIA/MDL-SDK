/******************************************************************************
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
class RemoveMemPHIs : public llvm::FunctionPass
{
public:
    static char ID;

public:
    RemoveMemPHIs();

    void getAnalysisUsage(llvm::AnalysisUsage &usage) const final;

    bool runOnFunction(llvm::Function &function) final;

    llvm::StringRef getPassName() const final {
        return "RemoveMemPHIs";
    }
};

RemoveMemPHIs::RemoveMemPHIs()
: FunctionPass( ID )
{
}

void RemoveMemPHIs::getAnalysisUsage(llvm::AnalysisUsage &usage) const
{
}

bool RemoveMemPHIs::runOnFunction(llvm::Function &function)
{
    bool changed = false;
    llvm::SmallVector<llvm::PHINode *, 4> to_remove;

    // always insert new blocks after "end" to avoid iterating over them
    for (llvm::Function::iterator it = function.begin(), end = function.end(); it != end; ++it) {
        for (llvm::PHINode &phi : it->phis()) {
            if (!llvm::isa<llvm::PointerType>(phi.getType()))
                continue;

            bool all_values_same = true;
            llvm::Value *first_value = nullptr;
            llvm::Instruction *base_inst = nullptr;
            unsigned opcode = 0;
            unsigned num_operands = 0;
            for (llvm::Value *cur_val : phi.incoming_values()) {
                if (llvm::isa<llvm::UndefValue>(cur_val))
                    continue;

                if (first_value == nullptr) {
                    first_value = cur_val;

                    // for the comparison, skip any bitcast
                    if (llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(cur_val)) {
                        cur_val = cast->getOperand(0);
                    }

                    if ((base_inst = llvm::dyn_cast<llvm::Instruction>(cur_val)) != nullptr) {
                        opcode = base_inst->getOpcode();
                        num_operands = base_inst->getNumOperands();
                    }
                }
                else if (first_value != cur_val) {
                    // not an undef and does not point to the exact same value object?
                    // compare instruction kind and operands (only one level)

                    // for the comparison, skip any bitcast
                    if (llvm::BitCastInst *cast = llvm::dyn_cast<llvm::BitCastInst>(cur_val)) {
                        cur_val = cast->getOperand(0);
                    }

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

                    llvm::Instruction *clone = inst->clone();
                    if (inst->hasName())
                        clone->setName(inst->getName() + ".rmemphi");
                    it->getInstList().insert(it->getFirstInsertionPt(), clone);
                    phi.replaceAllUsesWith(clone);
                } else {
                    phi.replaceAllUsesWith(first_value);
                }
                to_remove.push_back(&phi);
            }
        }
    }

    if (!to_remove.empty()) {
        for (llvm::PHINode *phi : to_remove) {
            phi->eraseFromParent();
        }
        changed = true;
    }

    return changed;
}

char RemoveMemPHIs::ID = 0;

llvm::Pass *createRemoveMemPHIsPass()
{
    return new RemoveMemPHIs();
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
        mpm.add(llvm::createLoopSimplifyCFGPass());    // ensure all exit blocks are dominated by
                                                       // the loop header
        mpm.add(llvm::hlsl::createLoopExitEnumerationPass());  // ensure all loops have <= 1 exits
        mpm.add(llvm::hlsl::createUnswitchPass());     // get rid of all switch instructions
        mpm.add(createRemoveMemPHIsPass());
        mpm.add(llvm::hlsl::createASTComputePass(m_type_mapper));
        mpm.add(hlsl::createHLSLWriterPass(
            get_allocator(),
            m_type_mapper,
            out,
            m_num_texture_spaces,
            m_num_texture_results,
            m_enable_full_debug,
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
    case IType_texture::TS_2D:   return "_2d";
    case IType_texture::TS_3D:   return "_3d";
    case IType_texture::TS_CUBE: return "_cube";
    case IType_texture::TS_PTEX: return "_ptex";
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
            func->setName("tex_" + llvm::Twine(func_name) + get_hlsl_tex_type_func_suffix(def));
        } else {
            func->setName("df_" + llvm::Twine(func_name));
        }
        return func;
    }
}

}  // mdl
}  // mi
