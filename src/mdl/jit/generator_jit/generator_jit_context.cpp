/******************************************************************************
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdio>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/DebugInfo.h>
#include <llvm/DIBuilder.h>

#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_positions.h>

#include "generator_jit_context.h"
#include "generator_jit_llvm.h"
#include "generator_jit_type_map.h"
#include "generator_jit_res_manager.h"

namespace mi {
namespace mdl {

// Constructor, creates a context for a function including its first scope.
Function_context::Function_context(
    mi::mdl::IAllocator        *alloc,
    LLVM_code_generator        &code_gen,
    Function_instance const    &func_inst,
    llvm::Function             *func,
    unsigned                   flags)
: m_arena(alloc)
, m_arena_builder(m_arena)
, m_var_context_data(0, Var_context_data_map::hasher(), Var_context_data_map::key_equal(), alloc)
, m_array_size_map(0, Array_size_map::hasher(), Array_size_map::key_equal(), alloc)
, m_type_mapper(code_gen.get_type_mapper())
, m_llvm_context(code_gen.get_llvm_context())
, m_ir_builder(create_bb(m_llvm_context, "start", func))
, m_md_builder(code_gen.get_llvm_context())
, m_di_builder(code_gen.get_debug_info_builder())
, m_di_file(code_gen.get_debug_info_file_entry())
, m_function(func)
, m_start_bb(m_ir_builder.GetInsertBlock())
, m_end_bb(create_bb(m_llvm_context, "end", func))
, m_unreachable_bb(NULL)
, m_body_start_point(NULL)
, m_retval_adr(NULL)
, m_lambda_results_override(NULL)
, m_curr_pos(NULL)
, m_res_manager(code_gen.get_resource_manager())
, m_code_gen(code_gen)
, m_flags(flags)
, m_optimize_on_finalize(true)
, m_break_stack(BB_stack::container_type(alloc))
, m_continue_stack(BB_stack::container_type(alloc))
, m_di_function()
, m_dilb_stack(DILB_stack::container_type(alloc))
, m_accesible_parameters(alloc)
, m_tex_loookup_call_mode(code_gen.get_tex_lookup_call_mode())
{
    // fill the array type map
    Function_instance::Array_instances const &ais(func_inst.get_array_instances());
    for (size_t i = 0, n = ais.size(); i < n; ++i) {
        Array_instance const &ai(ais[i]);

        m_array_size_map[ai.get_deferred_size()] = ai.get_immediate_size();
    }

    // set fast-math flags
    llvm::FastMathFlags FMF;
    if (code_gen.is_fast_math_enabled()) {
        FMF.setUnsafeAlgebra();
    } else if (code_gen.is_finite_math_enabled()) {
        FMF.setNoNaNs();
        FMF.setNoInfs();
    }

    if (code_gen.is_reciprocal_math_enabled()) {
        FMF.setAllowReciprocal();
    }

    m_ir_builder.SetFastMathFlags(FMF);

    // create the body block and jump from start to it
    llvm::BasicBlock *body_bb = create_bb("body");
    m_body_start_point = m_ir_builder.CreateBr(body_bb);

    // set the cursor to the body block
    m_ir_builder.SetInsertPoint(body_bb);

    llvm::Type *ret_tp = m_function->getReturnType();

    if (m_flags & LLVM_context_data::FL_SRET) {
        // first argument is used as return value store
        m_retval_adr = func->arg_begin();
    } else if (!ret_tp->isVoidTy()) {
        // create a store that will hold the return value ...
        m_retval_adr = create_local(ret_tp, "return_value");
    }

    mi::mdl::IDefinition const *func_def = func_inst.get_def();
    if (func_def == NULL) {
        // debug info is not possible without a definition
        m_di_builder = NULL;
    } else {
        // generate debug info
        if (m_di_builder != NULL) {
            llvm::DIFile &di_file = code_gen.get_debug_info_file_entry();

            unsigned start_line = 0;
            if (mi::mdl::Position const *pos = func_def->get_position()) {
                start_line = pos->get_start_line();

                string filename(get_dbg_curr_filename(), alloc);
                string directory(alloc);

                size_t p = filename.rfind('/');
                if (p != string::npos) {
                    directory = filename.substr(0, p);
                    filename  = filename.substr(p + 1);
                } else {
                    size_t p = filename.rfind('\\');
                    if (p != string::npos) {
                        directory = filename.substr(0, p);
                        filename  = filename.substr(p + 1);
                    }
                }
                m_di_file = m_di_builder->createFile(filename.c_str(), directory.c_str());

                di_file = m_di_file;
            }

            llvm::DICompositeType di_func_type = m_type_mapper.get_debug_info_type(
                m_di_builder, m_di_file, cast<mi::mdl::IType_function>(func_def->get_type()));

            m_di_function = m_di_builder->createFunction(
                /*Scope=*/di_file,
                /*Name=*/func_def->get_symbol()->get_name(),
                /*LinkeageName=*/func->getName(),
                /*File=*/di_file,
                start_line,
                di_func_type,
                func_def != NULL ?
                    !func_def->get_property(mi::mdl::IDefinition::DP_IS_EXPORTED) :
                    /*assume local*/true,
                /*isDefinition=*/true,
                start_line,
                llvm::DIDescriptor::FlagPrototyped,
                code_gen.is_optimized(),
                func);
        }
    }
    // create the function scope
    push_block_scope(func_def != NULL ? func_def->get_position() : NULL);
}

// Constructor, creates a context for modification of an already existing function.
// Array instances and the create_return functions are not available in this mode.
Function_context::Function_context(
    mi::mdl::IAllocator        *alloc,
    LLVM_code_generator        &code_gen,
    llvm::Function             *func,
    unsigned                   flags,
    bool                       optimize_on_finalize)
 : m_arena(alloc)
 , m_arena_builder(m_arena)
 , m_var_context_data(0, Var_context_data_map::hasher(), Var_context_data_map::key_equal(), alloc)
 , m_array_size_map(0, Array_size_map::hasher(), Array_size_map::key_equal(), alloc)
 , m_type_mapper(code_gen.get_type_mapper())
 , m_llvm_context(code_gen.get_llvm_context())
 , m_ir_builder(&func->getEntryBlock().front())
 , m_md_builder(code_gen.get_llvm_context())
 , m_di_builder(code_gen.get_debug_info_builder())
 , m_di_file(code_gen.get_debug_info_file_entry())
 , m_function(func)
 , m_start_bb(m_ir_builder.GetInsertBlock())
 , m_end_bb(NULL)
 , m_unreachable_bb(NULL)
 , m_body_start_point(NULL)
 , m_retval_adr(NULL)
 , m_lambda_results_override(NULL)
 , m_curr_pos(NULL)
 , m_res_manager(code_gen.get_resource_manager())
 , m_code_gen(code_gen)
 , m_flags(flags)
 , m_optimize_on_finalize(optimize_on_finalize)
 , m_break_stack(BB_stack::container_type(alloc))
 , m_continue_stack(BB_stack::container_type(alloc))
 , m_di_function()
 , m_dilb_stack(DILB_stack::container_type(alloc))
 , m_accesible_parameters(alloc)
 , m_tex_loookup_call_mode(code_gen.get_tex_lookup_call_mode())
{
    // set fast-math flags
    llvm::FastMathFlags FMF;
    if (code_gen.is_fast_math_enabled()) {
        FMF.setUnsafeAlgebra();
    } else if (code_gen.is_finite_math_enabled()) {
        FMF.setNoNaNs();
        FMF.setNoInfs();
    }

    if (code_gen.is_reciprocal_math_enabled()) {
        FMF.setAllowReciprocal();
    }

    m_ir_builder.SetFastMathFlags(FMF);

    // set the cursor to the first instruction after all Alloca instructions
    llvm::BasicBlock::iterator param_init_insert_point = func->front().begin();
    while (llvm::isa<llvm::AllocaInst>(param_init_insert_point))
        ++param_init_insert_point;
    if (param_init_insert_point == func->front().begin()) {
        // no Alloca instructions exist, so we need to add a separator
        // to avoid code being inserted in front of Alloca instructions
        llvm::Value *null_val = llvm::ConstantInt::getNullValue(
            llvm::Type::getInt1Ty(m_llvm_context));
        m_ir_builder.CreateAdd(null_val, null_val, "nop");
        m_body_start_point = m_ir_builder.GetInsertPoint();
    } else {
        m_body_start_point = param_init_insert_point;
        m_ir_builder.SetInsertPoint(m_body_start_point);
    }

    // debug info is not possible without a definition
    m_di_builder = NULL;
}

 // Destructor, closes the last scope and fills the end block, if the context was not
 // created for an already existing function.
Function_context::~Function_context()
{
    // End block is only known when the context was not used to modify an existing function.
    if (m_end_bb != NULL) {
        // creates final jump to the end block
        // Note: this is not really necessary, because we don't have void functions so far,
        // so all live BB's are terminated by a ret. It works, because the current block is
        // the "unreachable block" here ...
        m_ir_builder.CreateBr(m_end_bb);

        // fill the end block
        m_ir_builder.SetInsertPoint(m_end_bb);

        // finish the function scope
        pop_block_scope();
        MDL_ASSERT(m_dilb_stack.empty());

        if (m_flags & LLVM_context_data::FL_SRET) {
            // first argument is the return value reference
            m_retval_adr = m_function->arg_begin();

            // no return value
            m_ir_builder.CreateRetVoid();
        } else {
            llvm::Type *ret_tp = m_function->getReturnType();
            if (!ret_tp->isVoidTy()) {
                // load the return value at the end and return it
                m_ir_builder.CreateRet(m_ir_builder.CreateLoad(m_retval_adr));
            } else {
                // no return value
                m_ir_builder.CreateRetVoid();
            }
        }

        if (m_unreachable_bb != NULL) {
            // we have only one unreachable block that is filled with garbage,
            // first kill all instructions here that might be there due to naive construction.
            // This step is necessary, because it might contain several terminators.
            llvm::BasicBlock &BB = *m_unreachable_bb;

            for (llvm::BasicBlock::iterator it = BB.begin(); it != BB.end();) {
                llvm::Instruction *Inst = it++;
                Inst->dropAllReferences();
            }
            for (llvm::BasicBlock::iterator it = BB.begin(); it != BB.end();) {
                llvm::Instruction *Inst = it++;
                Inst->eraseFromParent();
            }

            // finally, place an unreachable instruction here
            m_ir_builder.SetInsertPoint(m_unreachable_bb);
            m_ir_builder.CreateUnreachable();
        }
    }

    // optimize function to improve inlining, if requested
    if (m_optimize_on_finalize)
        m_code_gen.m_func_pass_manager->run(*m_function);
}

// Get the first (real) parameter of the current function.
llvm::Function::arg_iterator Function_context::get_first_parameter()
{
    llvm::Function::arg_iterator arg_it = m_function->arg_begin();
    if (m_flags & LLVM_context_data::FL_SRET) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_STATE) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_RES) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_EXC) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_LMBD_RES) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_OBJ_ID) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_TRANSFORMS) {
        ++arg_it;
        ++arg_it;
    }
   return arg_it;
}

// Get the state parameter of the current function.
llvm::Value *Function_context::get_state_parameter()
{
    MDL_ASSERT(m_flags & LLVM_context_data::FL_HAS_STATE);

    llvm::Function::arg_iterator arg_it = m_function->arg_begin();
    if (m_flags & LLVM_context_data::FL_SRET) {
        ++arg_it;
    }
    return arg_it;
}

// Get the resource_data parameter of the current function.
llvm::Value *Function_context::get_resource_data_parameter()
{
    MDL_ASSERT(m_flags & LLVM_context_data::FL_HAS_RES);

    llvm::Function::arg_iterator arg_it = m_function->arg_begin();
    if (m_flags & LLVM_context_data::FL_SRET) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_STATE) {
        ++arg_it;
    }
    return arg_it;
}

// Get the exc_state parameter of the current function.
llvm::Value *Function_context::get_exc_state_parameter()
{
    MDL_ASSERT(m_flags & LLVM_context_data::FL_HAS_EXC);

    llvm::Function::arg_iterator arg_it = m_function->arg_begin();
    if (m_flags & LLVM_context_data::FL_SRET) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_STATE) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_RES) {
        ++arg_it;
    }
    return arg_it;
}

// Get the cap_args parameter of the current function.
llvm::Value *Function_context::get_cap_args_parameter()
{
    MDL_ASSERT(m_flags & LLVM_context_data::FL_HAS_CAP_ARGS);

    llvm::Function::arg_iterator arg_it = m_function->arg_begin();
    if (m_flags & LLVM_context_data::FL_SRET) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_STATE) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_RES) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_EXC) {
        ++arg_it;
    }
    return arg_it;
}

// Get the lambda_results parameter of the current function.
llvm::Value *Function_context::get_lambda_results_parameter()
{
    if (m_lambda_results_override)
        return m_lambda_results_override;

    MDL_ASSERT(m_flags & LLVM_context_data::FL_HAS_LMBD_RES);

    llvm::Function::arg_iterator arg_it = m_function->arg_begin();
    if (m_flags & LLVM_context_data::FL_SRET) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_STATE) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_RES) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_EXC) {
        ++arg_it;
    }
    if (m_flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
        ++arg_it;
    }
    return arg_it;
}

// Get the wavelength_min() value of the current function.
llvm::Value *Function_context::get_wavelength_min_value()
{
    // should be replaced during instantiation of a material, if not replace by constant
    return get_constant(380.0f);
}

// Get the wavelength_max() value of the current function.
llvm::Value *Function_context::get_wavelength_max_value()
{
    // should be replaced during instantiation of a material, if not replace by constant
    return get_constant(780.0f);
}

// Get the object_id value of the current function.
llvm::Value *Function_context::get_object_id_value()
{
    if (m_flags & LLVM_context_data::FL_HAS_OBJ_ID) {
        // object_id is passed as a parameter

        llvm::Function::arg_iterator arg_it = m_function->arg_begin();
        if (m_flags & LLVM_context_data::FL_SRET) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_STATE) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_RES) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_EXC) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_LMBD_RES) {
            ++arg_it;
        }
        return arg_it;
    } else {
        // object_id is a constant taken from the code generator
        return m_code_gen.get_current_object_id(*this);
    }
}

/// Get the world-to-object transform (matrix) value of the current function.
llvm::Value *Function_context::get_w2o_transform_value()
{
    if (m_flags & LLVM_context_data::FL_HAS_TRANSFORMS) {
        // matrix is passed as a parameter

        llvm::Function::arg_iterator arg_it = m_function->arg_begin();
        if (m_flags & LLVM_context_data::FL_SRET) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_STATE) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_RES) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_EXC) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_LMBD_RES) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_OBJ_ID) {
            ++arg_it;
        }
        return arg_it;
    } else {
        // world-to-object transform is a constant taken from the code generator
        return m_code_gen.get_w2o_transform_value(*this);
    }
}

/// Get the object-to-world transform (matrix) value of the current function.
llvm::Value *Function_context::get_o2w_transform_value()
{
    if (m_flags & LLVM_context_data::FL_HAS_TRANSFORMS) {
        // matrix is passed as a parameter

        llvm::Function::arg_iterator arg_it = m_function->arg_begin();
        if (m_flags & LLVM_context_data::FL_SRET) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_STATE) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_RES) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_EXC) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_CAP_ARGS) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_LMBD_RES) {
            ++arg_it;
        }
        if (m_flags & LLVM_context_data::FL_HAS_OBJ_ID) {
            ++arg_it;
        }
        ++arg_it;
        return arg_it;
    } else {
        // object-to-world transform is a constant taken from the code generator
        return m_code_gen.get_o2w_transform_value(*this);
    }
}

// Get the current break destination.
llvm::BasicBlock *Function_context::tos_break()
{
    MDL_ASSERT(!m_break_stack.empty());

    return m_break_stack.top();
}

// Get the current continue destination.
llvm::BasicBlock *Function_context::tos_continue()
{
    MDL_ASSERT(!m_continue_stack.empty());

    return m_continue_stack.top();
}

// Retrieve the LLVM context data for a MDL variable definition.
LLVM_context_data *Function_context::get_context_data(
    mi::mdl::IDefinition const *def)
{
    MDL_ASSERT(
        def->get_kind() == mi::mdl::IDefinition::DK_VARIABLE ||
        def->get_kind() == mi::mdl::IDefinition::DK_PARAMETER);

    Var_context_data_map::const_iterator it(m_var_context_data.find(def));
    if (it != m_var_context_data.end()) {
        return it->second;
    }

    // not yet allocated, allocate a new one

    // but instantiate the type
    IType const *var_type = def->get_type();

    llvm::Type *var_tp = m_code_gen.lookup_type(var_type, instantiate_type_size(var_type));
    LLVM_context_data *ctx = m_arena_builder.create<LLVM_context_data>(
        this, var_tp, def->get_symbol()->get_name());

    m_var_context_data[def] = ctx;

    // add debug info
    add_debug_info(def);

    return ctx;
}

// Retrieve the LLVM context data for a lambda parameter.
LLVM_context_data *Function_context::get_context_data(
    size_t idx)
{
    void const *key = reinterpret_cast<void const *>(idx);
    Var_context_data_map::const_iterator it(m_var_context_data.find(key));
    if (it != m_var_context_data.end()) {
        // no debug info yet

        return it->second;
    }
    MDL_ASSERT(!"unknown lambda parameter index");
    return NULL;
}

// Create the LLVM context data for a MDL variable/parameter definition.
LLVM_context_data *Function_context::create_context_data(
    mi::mdl::IDefinition const *def,
    llvm::Value                *value,
    bool                       by_reference)
{
    MDL_ASSERT(
        def->get_kind() == mi::mdl::IDefinition::DK_VARIABLE ||
        def->get_kind() == mi::mdl::IDefinition::DK_PARAMETER);

    MDL_ASSERT(
        m_var_context_data.find(def) == m_var_context_data.end() && "context already created");

    // not yet allocated, allocate a new one
    LLVM_context_data *ctx = m_arena_builder.create<LLVM_context_data>(
        this, value, def->get_symbol()->get_name(), by_reference);

    m_var_context_data[def] = ctx;

    // add debug info
    add_debug_info(def);

    return ctx;
}

// Create the LLVM context data for a lambda function parameter.
LLVM_context_data *Function_context::create_context_data(
    size_t      idx,
    llvm::Value *value,
    bool        by_reference)
{
    void const *key = reinterpret_cast<void const *>(idx);
    MDL_ASSERT(
        m_var_context_data.find(key) == m_var_context_data.end() && "context already created");

    // not yet allocated, allocate a new nameless one
    LLVM_context_data *ctx = m_arena_builder.create<LLVM_context_data>(
        this, value, "", by_reference);

    m_var_context_data[key] = ctx;

    // FIXME: no debug info yet for lambdas

    return ctx;
}

// Creates a void return.
void Function_context::create_void_return()
{
    MDL_ASSERT(m_end_bb != NULL && "create_void_return may not be called in modification mode");

    create_jmp(m_end_bb);
}

// Creates a return.
void Function_context::create_return(llvm::Value *expr)
{
    MDL_ASSERT(m_end_bb != NULL && "create_return may not be called in modification mode");

    llvm::StoreInst *st = m_ir_builder.CreateStore(expr, m_retval_adr);
    if (m_flags & LLVM_context_data::FL_UNALIGNED_RET) {
        llvm::Type *res_type = expr->getType();
        if (llvm::ArrayType * a_tp = llvm::dyn_cast<llvm::ArrayType>(res_type)) {
            res_type = a_tp->getElementType();
        }
        if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(res_type)) {
            res_type = v_tp->getElementType();

            // reduce the alignment of the store to this size
            st->setAlignment(res_type->getPrimitiveSizeInBits() / 8);
        }
    }
    create_jmp(m_end_bb);
}

// Creates a Br instruction and set the current block to unreachable.
void Function_context::create_jmp(llvm::BasicBlock *dest)
{
    m_ir_builder.CreateBr(dest);
    m_ir_builder.SetInsertPoint(get_unreachable_bb());
}

// Creates a &ptr[index] operation, index is assured to be in bounds.
llvm::Value *Function_context::create_simple_gep_in_bounds(llvm::Value *ptr, llvm::Value *index)
{
    llvm::Value *access[] = {
        get_constant(int(0)),
        index
    };

    llvm::Value *gep = m_ir_builder.CreateInBoundsGEP(ptr, access);
    return gep;
}

// Creates a &ptr[index] operation, index is assured to be in bounds.
llvm::Value *Function_context::create_simple_gep_in_bounds(llvm::Value *ptr, unsigned index)
{
    return create_simple_gep_in_bounds(ptr, get_constant(int(index)));
}

// Convert a boolean value.
llvm::ConstantInt *Function_context::get_constant(mi::mdl::IValue_bool const *b)
{
    return llvm::ConstantInt::get(m_type_mapper.get_bool_type(), b->get_value() ? 1 : 0);
}

// Convert a boolean value.
llvm::ConstantInt *Function_context::get_constant(bool b)
{
    return llvm::ConstantInt::get(m_type_mapper.get_bool_type(), b ? 1 : 0);
}

// Convert an integer value.
llvm::ConstantInt *Function_context::get_constant(mi::mdl::IValue_int_valued const *i)
{
    return llvm::ConstantInt::get(m_type_mapper.get_int_type(), i->get_value());
}

// Convert an integer value.
llvm::ConstantInt *Function_context::get_constant(int i)
{
    return llvm::ConstantInt::get(m_type_mapper.get_int_type(), i);
}

// Convert a size_t value.
llvm::ConstantInt *Function_context::get_constant(size_t u)
{
    return llvm::ConstantInt::get(m_type_mapper.get_size_t_type(), u);
}

// Convert a float value.
llvm::ConstantFP *Function_context::get_constant(mi::mdl::IValue_float const *f)
{
    return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(f->get_value()));
}

// Convert a float value.
llvm::ConstantFP *Function_context::get_constant(float f)
{
    return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(f));
}

// Convert a double value.
llvm::ConstantFP *Function_context::get_constant(mi::mdl::IValue_double const *d)
{
    return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(d->get_value()));
}

// Convert a double value.
llvm::ConstantFP *Function_context::get_constant(double d)
{
    return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(d));
}

// Convert a string value.
llvm::Value *Function_context::get_constant(mi::mdl::IValue_string const *s)
{
    LLVM_code_generator::Global_const_map::iterator it = m_code_gen.m_global_const_map.find(s);
    if (it == m_code_gen.m_global_const_map.end()) {
        llvm::Value *v = NULL;
        if (m_type_mapper.strings_mapped_to_ids()) {
            // retrieve the ID: it is potentially an error if no resource manager is available
            Type_mapper::Tag ID = m_res_manager != NULL ? m_res_manager->get_string_index(s) : 0u;
            // and add it to the string table
            m_code_gen.add_string_constant(s->get_value(), ID);
            v = get_constant(ID);
        } else {
            // generate the constant directly
            v = m_ir_builder.CreateGlobalStringPtr(s->get_value());
        }
        m_code_gen.m_global_const_map[s] =
            LLVM_code_generator::Value_offset_pair(v, /*is_offset=*/false);
        return v;
    } else {
        LLVM_code_generator::Value_offset_pair const &pair = it->second;

        MDL_ASSERT(!pair.is_offset);
        return pair.value;
    }
}

// Convert a string value.
llvm::Value *Function_context::get_constant(char const *s)
{
    return m_ir_builder.CreateGlobalStringPtr(s);
}

// Get a int/FP/vector splat constant.
llvm::Constant *Function_context::get_constant(llvm::Type *type, int v)
{
    if (type->isIntegerTy()) {
        return llvm::ConstantInt::get(type, v);
    } else if (type->isFloatTy()) {
        return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(float(v)));
    } else if (type->isDoubleTy()) {
        return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(double(v)));
    } else if (type->isVectorTy()) {
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(type);

        llvm::Constant *c = get_constant(v_tp->getElementType(), v);
        return llvm::ConstantVector::getSplat(v_tp->getNumElements(), c);
    }
    MDL_ASSERT(!"Cannot create constant of unexpected type");
    return NULL;
}

// Convert a resource value.
llvm::ConstantInt *Function_context::get_constant(mi::mdl::IValue_resource const *r)
{
    return llvm::ConstantInt::get(m_type_mapper.get_int_type(), r->get_tag_value());
}

// Get a FP/vector splat constant.
llvm::Constant *Function_context::get_constant(llvm::Type *type, double v)
{
    if (type->isFloatTy()) {
        return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(float(v)));
    } else if (type->isDoubleTy()) {
        return llvm::ConstantFP::get(m_llvm_context, llvm::APFloat(double(v)));
    } else if (type->isVectorTy()) {
        llvm::VectorType *v_tp = llvm::cast<llvm::VectorType>(type);

        llvm::Constant *c = get_constant(v_tp->getElementType(), v);
        return llvm::ConstantVector::getSplat(v_tp->getNumElements(), c);
    }
    MDL_ASSERT(!"Cannot create constant of unexpected type");
    return NULL;
}

// Get a constant from an MDL IValue.
llvm::Value *Function_context::get_constant(mi::mdl::IValue const *v)
{
    mi::mdl::IValue_string::Kind kind = v->get_kind();

    switch (kind) {
    case mi::mdl::IValue::VK_BAD:
        MDL_ASSERT(!"BAD value detected");
        return NULL;
    case mi::mdl::IValue::VK_BOOL:
        return get_constant(cast<mi::mdl::IValue_bool>(v));
    case mi::mdl::IValue::VK_INT:
    case mi::mdl::IValue::VK_ENUM:
        return get_constant(cast<mi::mdl::IValue_int_valued>(v));
    case mi::mdl::IValue::VK_FLOAT:
        return get_constant(cast<mi::mdl::IValue_float>(v));
    case mi::mdl::IValue::VK_DOUBLE:
        return get_constant(cast<mi::mdl::IValue_double>(v));
    case mi::mdl::IValue::VK_STRING:
        return get_constant(cast<mi::mdl::IValue_string>(v));

    case mi::mdl::IValue::VK_VECTOR:
    case mi::mdl::IValue::VK_RGB_COLOR:
    case mi::mdl::IValue::VK_MATRIX:
        // these three are all encoded similar
        {
            llvm::Constant *elems[16];

            mi::mdl::IValue_compound const *arr_v = cast<mi::mdl::IValue_compound>(v);
            size_t n_elemns = arr_v->get_component_count();

            if (kind == mi::mdl::IValue::VK_MATRIX) {
                // flat matrix values
                mi::mdl::IType_matrix const *m_type = cast<mi::mdl::IType_matrix>(v->get_type());
                mi::mdl::IType_vector const *v_type = m_type->get_element_type();
                size_t                      v_size  = v_type->get_size();

                for (size_t i = 0; i < n_elemns; ++i) {
                    mi::mdl::IValue_compound const *vec_v =
                        cast<mi::mdl::IValue_compound>(arr_v->get_value(i));

                    for (size_t j = 0; j < v_size; ++j) {
                        llvm::Value *v = get_constant(vec_v->get_value(j));

                        elems[i * v_size + j] = llvm::cast<llvm::Constant>(v);
                    }
                }
                n_elemns *= v_size;
            } else {
                for (size_t i = 0; i < n_elemns; ++i) {
                    llvm::Value *v = get_constant(arr_v->get_value(i));

                    elems[i] = llvm::cast<llvm::Constant>(v);
                }
            }

            mi::mdl::IType const *v_type = v->get_type();
            llvm::Type           *tp     = m_type_mapper.lookup_type(m_llvm_context, v_type);

            if (tp->isVectorTy()) {
                // encode into a vector
                return llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant *>(elems, n_elemns));
            } else {
                llvm::ArrayType *a_tp = llvm::cast<llvm::ArrayType>(tp);
                llvm::Type      *e_tp = a_tp->getArrayElementType();

                if (e_tp->isVectorTy()) {
                    // encode a matrix into an array of vectors (small vector mode)
                    mi::mdl::IType_matrix const *m_type = cast<mi::mdl::IType_matrix>(v_type);
                    mi::mdl::IType_vector const *v_type = m_type->get_element_type();

                    int n_cols = m_type->get_columns();
                    int n_rows = v_type->get_size();
                    llvm::Constant *vectors[4];

                    for (int col = 0; col < n_cols; ++col) {
                        vectors[col] = llvm::ConstantVector::get(
                            llvm::ArrayRef<llvm::Constant *>(&elems[col * n_rows], n_rows));
                    }
                    return llvm::ConstantArray::get(
                        a_tp, llvm::ArrayRef<llvm::Constant *>(vectors, n_cols));

                } else {
                    // encode a matrix/vector into an array of scalars (scalar mode)
                    return llvm::ConstantArray::get(
                        a_tp, llvm::ArrayRef<llvm::Constant *>(elems, n_elemns));
                }
            }
        }
        break;

    case mi::mdl::IValue::VK_ARRAY:
        {
            // Note: we represent string literals not by constants (they are variable length),
            // but by values ...
            mi::mdl::IValue_array const *arr_v = cast<mi::mdl::IValue_array>(v);

            size_t n_elemns = arr_v->get_component_count();

            llvm::SmallVector<llvm::Value *, 8> values(n_elemns);
            bool all_are_const = true;
            for (size_t i = 0; i < n_elemns; ++i) {
                llvm::Value *v = get_constant(arr_v->get_value(i));
                values[i] = v;

                all_are_const &= llvm::isa<llvm::Constant>(v);
            }

            llvm::Type *type = m_type_mapper.lookup_type(m_llvm_context, arr_v->get_type());

            if (all_are_const) {
                // create a compound constant
                llvm::SmallVector<llvm::Constant *, 8> elems(n_elemns);

                for (size_t i = 0; i < n_elemns; ++i) {
                    elems[i] = llvm::cast<llvm::Constant>(values[i]);
                }

                return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(type), elems);
            } else {
                // create an compound value
                MDL_ASSERT(!"NYI");
                break;
            }
        }
        break;

    case mi::mdl::IValue::VK_STRUCT:
        {
            mi::mdl::IValue_struct const *arr_v = cast<mi::mdl::IValue_struct>(v);

            size_t n_elemns = arr_v->get_component_count();

            llvm::SmallVector<llvm::Value *, 8> values(n_elemns);
            bool all_are_const = true;
            for (size_t i = 0; i < n_elemns; ++i) {
                llvm::Value *v = get_constant(arr_v->get_value(i));
                values[i] = v;

                all_are_const &= llvm::isa<llvm::Constant>(v);
            }

            if (all_are_const) {
                // create a compound constant
                llvm::SmallVector<llvm::Constant *, 8> elems(n_elemns);

                for (size_t i = 0; i < n_elemns; ++i) {
                    elems[i] = llvm::cast<llvm::Constant>(values[i]);
                }

                llvm::Type *type = m_type_mapper.lookup_type(m_llvm_context, arr_v->get_type());
                return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(type), elems);
            } else {
                // create an compound value
                MDL_ASSERT(!"NYI");
                break;
            }
        }
        break;

    case mi::mdl::IValue::VK_TEXTURE:
    case mi::mdl::IValue::VK_LIGHT_PROFILE:
    case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
        // textures, light_profile and bsdf_measurement tags are mapped to Tag type
        return get_constant(cast<mi::mdl::IValue_resource>(v));

    case mi::mdl::IValue::VK_INVALID_REF:
        // invalid refs are just represented as int32 0
        return get_constant(int(0));
    }
    MDL_ASSERT(!"unsupported value kind");

    llvm::Type *type = m_type_mapper.lookup_type(m_llvm_context, v->get_type());
    return llvm::UndefValue::get(type);
}

// Get a vector3 constant.
llvm::Constant *Function_context::get_constant(
    llvm::VectorType *vtype, float x, float y, float z)
{
    size_t l = vtype->getNumElements();
    llvm::SmallVector<llvm::Constant *, 4> cnst;

    cnst.push_back(get_constant(x));
    cnst.push_back(get_constant(y));
    cnst.push_back(get_constant(z));

    for (size_t i = 3; i < l; ++i) {
        cnst.push_back(get_constant(0.0f));
    }

    return llvm::ConstantVector::get(cnst);
}

// Get a vector4 constant.
llvm::Constant *Function_context::get_constant(
    llvm::VectorType *vtype, float x, float y, float z, float w)
{
    size_t l = vtype->getNumElements();
    llvm::SmallVector<llvm::Constant *, 4> cnst;

    cnst.push_back(get_constant(x));
    cnst.push_back(get_constant(y));
    cnst.push_back(get_constant(z));
    cnst.push_back(get_constant(w));

    for (size_t i = 4; i < l; ++i) {
        cnst.push_back(get_constant(0.0f));
    }

    return llvm::ConstantVector::get(cnst);
}

// Get the file name of a module.
llvm::Value *Function_context::get_module_name(mi::mdl::IModule const *mod)
{
    LLVM_code_generator::Global_const_map::iterator it = m_code_gen.m_global_const_map.find(mod);
    if (it == m_code_gen.m_global_const_map.end()) {
        char const *name = mod->get_filename();
        if (name == NULL || name[0] == '\0') {
            // try module name
            name = mod->get_name();
        }

        llvm::Value *v = m_ir_builder.CreateGlobalStringPtr(name);
        m_code_gen.m_global_const_map[mod] = LLVM_code_generator::Value_offset_pair(v, false);
        return v;
    } else {
        LLVM_code_generator::Value_offset_pair const &pair = it->second;

        MDL_ASSERT(!pair.is_offset);
        return pair.value;
    }
}

// Get a shuffle mask.
llvm::Constant *Function_context::get_shuffle(llvm::ArrayRef<int> values)
{
    llvm::SmallVector<llvm::Constant *, 4> cnst;

    for (size_t i = 0, n = values.size(); i < n; ++i) {
        cnst.push_back(get_constant(values[i]));
    }
    return llvm::ConstantVector::get(cnst);
}


// Creates a single LLVM multiplication instruction (integer OR FP).
llvm::Value *Function_context::create_mul(
    llvm::Type  *res_type,
    llvm::Value *lhs,
    llvm::Value *rhs)
{
    if (llvm::ArrayType *a_tp = llvm::dyn_cast<llvm::ArrayType>(res_type)) {
        llvm::Value *res = llvm::ConstantAggregateZero::get(a_tp);
        llvm::Type *e_tp = a_tp->getElementType();

        bool l_is_array = lhs->getType() == a_tp;
        bool r_is_array = rhs->getType() == a_tp;

        unsigned idxes[1];
        for (unsigned i = 0, n = unsigned(a_tp->getNumElements()); i < n; ++i) {
            idxes[0] = i;

            llvm::Value *l = lhs;
            if (l_is_array) {
                l = m_ir_builder.CreateExtractValue(lhs, idxes);
            }

            llvm::Value *r = rhs;
            if (r_is_array) {
                r = m_ir_builder.CreateExtractValue(rhs, idxes);
            }

            llvm::Value *elem = create_mul(e_tp, l, r);

            res = m_ir_builder.CreateInsertValue(res, elem, idxes);
        }
        return res;
    } else {
        if (llvm::VectorType *v_tp = llvm::dyn_cast<llvm::VectorType>(res_type)) {
            if (lhs->getType() != v_tp) {
                lhs = create_vector_splat(v_tp, lhs);
            }
            if (rhs->getType() != v_tp) {
                rhs = create_vector_splat(v_tp, rhs);
            }
        }
        llvm::Value *res = res_type->isIntOrIntVectorTy() ?
            m_ir_builder.CreateMul(lhs, rhs) : m_ir_builder.CreateFMul(lhs, rhs);
        return res;
    }
}

// Creates a splat vector from a scalar value.
llvm::Value *Function_context::create_vector_splat(llvm::VectorType *res_type, llvm::Value *v)
{
    llvm::Value *res = llvm::UndefValue::get(res_type);

    for (unsigned i = 0, n = res_type->getNumElements(); i < n; ++i) {
        llvm::Value *idx = get_constant(int(i));
        res = m_ir_builder.CreateInsertElement(res, v, idx);
    }
    return res;
}

// Create a weighted conditional branch.
llvm::BranchInst *Function_context::CreateWeightedCondBr(
    llvm::Value      *cond,
    llvm::BasicBlock *true_bb,
    llvm::BasicBlock *false_bb,
    uint32_t         true_weight,
    uint32_t         false_weight)
{
    llvm::MDNode *branch_weights = m_md_builder.createBranchWeights(true_weight, false_weight);
    return m_ir_builder.CreateCondBr(cond, true_bb, false_bb, branch_weights);
}

// Creates an out-of-bounds exception call
void Function_context::create_oob_exception_call(
    llvm::Value        *index,
    llvm::Value        *bound,
    Exc_location const &loc)
{
    mi::mdl::IModule const *mod = loc.get_module();

    llvm::Value *f;
    if (mod == NULL)
        f = llvm::Constant::getNullValue(m_type_mapper.get_cstring_type());
    else
        f = get_module_name(mod);

    llvm::Value *l = get_constant(int(loc.get_line()));

    // call the out-of-bounds function
    llvm::Function *oob_func = m_code_gen.get_out_of_bounds();
    m_ir_builder.CreateCall5(oob_func, get_exc_state_parameter(), index, bound, f, l);
}

// Creates a bounds check using an exception.
void Function_context::create_bounds_check_with_exception(
    llvm::Value        *index,
    llvm::Value        *bound,
    Exc_location const &loc)
{
    llvm::Value *uindex   = index;
    llvm::Type  *bound_tp = bound->getType();
    if (index->getType() != bound_tp)
        uindex = m_ir_builder.CreateSExt(index, bound_tp);

    if (llvm::isa<llvm::ConstantInt>(uindex) && llvm::isa<llvm::ConstantInt>(bound)) {
        // both are constant, decide here
        llvm::ConstantInt *c_index = llvm::cast<llvm::ConstantInt>(uindex);
        llvm::ConstantInt *c_bound = llvm::cast<llvm::ConstantInt>(bound);

        llvm::APInt const &v_index = c_index->getValue();
        llvm::APInt const &v_bound = c_bound->getValue();

        if (v_index.ult(v_bound)) {
            // index is valid, do nothing
        } else {
            // index is out of bounds
            create_oob_exception_call(index, bound, loc);
        }
    } else {
        // need a run time check
        llvm::BasicBlock *ok_bb   = create_bb("bound_ok");
        llvm::BasicBlock *fail_bb = create_bb("bound_fail");

        create_bounds_check_cmp(uindex, bound, ok_bb, fail_bb);

        m_ir_builder.SetInsertPoint(fail_bb);
        {
            create_oob_exception_call(index, bound, loc);

            // for now, just a jump to the ok_bb
            m_ir_builder.CreateBr(ok_bb);
        }
        m_ir_builder.SetInsertPoint(ok_bb);
    }
}

// Creates a div-by-zero check using an exception.
void Function_context::create_div_exception_call(
    Exc_location const &loc)
{
    IModule const *mod = loc.get_module();

    llvm::Value *f;
    if (mod == NULL)
        f = llvm::Constant::getNullValue(m_type_mapper.get_cstring_type());
    else
        f = get_module_name(mod);

    llvm::Value *l = get_constant(int(loc.get_line()));

    // call the div-by-zero function
    llvm::Function *dbz_func = m_code_gen.get_div_by_zero();
    m_ir_builder.CreateCall3(dbz_func, get_exc_state_parameter(), f, l);
}

// Creates a div-by-zero check using an exception.
void Function_context::create_div_check_with_exception(
    llvm::Value        *v,
    Exc_location const &loc)
{
    if (llvm::ConstantInt *c_v = llvm::dyn_cast<llvm::ConstantInt>(v)) {
        llvm::APInt const &v_v = c_v->getValue();

        if (!v_v) {
            // value is zero, division will fail
            create_div_exception_call(loc);
        } else {
            // value is non-zero, do nothing
        }
    } else {
        // need a run time check
        llvm::BasicBlock *ok_bb   = create_bb("idiv_ok");
        llvm::BasicBlock *fail_bb = create_bb("idiv_fail");

        create_non_zero_check_cmp(v, ok_bb, fail_bb);

        m_ir_builder.SetInsertPoint(fail_bb);
        {
            create_div_exception_call(loc);

            // for now, just a jump to the ok_bb
            m_ir_builder.CreateBr(ok_bb);
        }
        m_ir_builder.SetInsertPoint(ok_bb);
    }
}

// Creates a bounds check.
void Function_context::create_bounds_check_cmp(
    llvm::Value        *index,
    llvm::Value        *bound,
    llvm::BasicBlock   *ok_bb,
    llvm::BasicBlock   *fail_bb)
{
    // convert int index to size_t
    llvm::Value *uindex   = index;
    llvm::Type  *bound_tp = bound->getType();
    if (index->getType() != bound_tp)
        uindex = m_ir_builder.CreateSExt(index, bound_tp);

    llvm::Value *cmp = m_ir_builder.CreateICmpULT(uindex, bound);

    // set branch weights, we expect all bounds checks to be ok and never fail
    CreateWeightedCondBr(cmp, ok_bb, fail_bb, 1, 0);
}

// Creates a non-zero check compare.
void Function_context::create_non_zero_check_cmp(
    llvm::Value        *v,
    llvm::BasicBlock   *non_zero_bb,
    llvm::BasicBlock   *zero_bb)
{
    llvm::Type *v_type = v->getType();
    llvm::Value *zero  = llvm::Constant::getNullValue(v_type);

    llvm::Value *cmp = m_ir_builder.CreateICmpEQ(v, zero);

    // set branch weights, we expect all div checks to be ok and never fail
    CreateWeightedCondBr(cmp, zero_bb, non_zero_bb, 0, 1);
}

// Pushes a break destination on the break stack.
void Function_context::push_break(llvm::BasicBlock *dest)
{
    m_break_stack.push(dest);
}

// Pop a break destination from the break stack.
void Function_context::pop_break()
{
    m_break_stack.pop();
}

// Pushes a continue destination on the continue stack.
void Function_context::push_continue(llvm::BasicBlock *dest)
{
    m_continue_stack.push(dest);
}

// Pop a continue destination from the continue stack.
void Function_context::pop_continue()
{
    m_continue_stack.pop();
}

// Creates a new variable scope and push it.
void Function_context::push_block_scope(mi::mdl::Position const *pos)
{
    if (m_di_builder != NULL) {
        llvm::DIScope parentScope;

        if (!m_dilb_stack.empty())
            parentScope = m_dilb_stack.top();
        else
            parentScope = m_di_function;

        unsigned start_line   = 0;
        unsigned start_column = 0;

        if (pos != NULL) {
            m_curr_pos   = pos;

            start_line   = pos->get_start_line();
            start_column = pos->get_start_column();
        }

        llvm::DILexicalBlock lexicalBlock = m_di_builder->createLexicalBlock(
            parentScope,
            m_di_file,
            start_line,
            start_column);
        m_dilb_stack.push(lexicalBlock);
    }
}

// Pop a variable scope, running destructors if necessary.
void Function_context::pop_block_scope()
{
    if (m_di_builder != NULL) {
        m_dilb_stack.pop();
    }
}

// Get the current LLVM debug info scope.
llvm::DIScope Function_context::get_debug_info_scope()
{
    return m_dilb_stack.top();
}

// Set the current source position.
void Function_context::set_curr_pos(mi::mdl::Position const &pos)
{
    m_curr_pos = &pos;
    if (m_di_builder != NULL) {
        m_ir_builder.SetCurrentDebugLocation(
            llvm::DebugLoc::get(
                m_curr_pos->get_start_line(),
                m_curr_pos->get_start_column(),
                get_debug_info_scope()
            )
        );
    }
}

// Add debug info for a variable declaration.
void Function_context::add_debug_info(mi::mdl::IDefinition const *var_def) {
    if (m_di_builder == NULL)
        return;

    LLVM_context_data *ctx_data = get_context_data(var_def);
    llvm::Value       *var_adr  = ctx_data->get_var_address();

    if (var_adr == NULL)
        return;

    mi::mdl::Position const *pos = var_def->get_position();
    if (pos == NULL)
        return;

    set_curr_pos(*pos);

    bool is_parameter = var_def->get_kind() == mi::mdl::IDefinition::DK_PARAMETER;

    llvm::DIScope scope = is_parameter ? m_di_function : get_debug_info_scope();
    llvm::DIType diType =
        m_type_mapper.get_debug_info_type(m_di_builder, scope, var_def->get_type());

    unsigned start_line = 0;
    if (pos != NULL) {
        start_line = pos->get_start_line();
    }

    llvm::DIVariable var = m_di_builder->createLocalVariable(
        is_parameter ? llvm::dwarf::DW_TAG_arg_variable : llvm::dwarf::DW_TAG_auto_variable,
        scope,
        var_def->get_symbol()->get_name(),
        m_di_file,
        start_line,
        diType,
        true, // preserve even in optimized builds
        /*Flags=*/0,
        is_parameter ? var_def->get_parameter_index() : 0);
    MDL_ASSERT(var.Verify());

    m_di_builder->insertDeclare(var_adr, var, m_ir_builder.GetInsertBlock());
}

// Make the given function/material parameter accessible.
void Function_context::make_accessible(mi::mdl::IParameter const *param)
{
    mi::mdl::IDefinition const *def = param->get_name()->get_definition();
    MDL_ASSERT(def != NULL && def->get_kind() == mi::mdl::IDefinition::DK_PARAMETER);

    m_accesible_parameters.push_back(def);
}

// Find a parameter for a given array size.
mi::mdl::IDefinition const *Function_context::find_parameter_for_size(
    mi::mdl::ISymbol const *sym) const
{
    for (size_t i = 0, n = m_accesible_parameters.size(); i < n; ++i) {
        mi::mdl::IDefinition const *p_def  = m_accesible_parameters[i];
        mi::mdl::IType_array const *a_type = as<mi::mdl::IType_array>(p_def->get_type());

        if (a_type == NULL)
            continue;
        if (a_type->is_immediate_sized())
            continue;
        mi::mdl::IType_array_size const *size = a_type->get_deferred_size();

        // Beware: we extracted the type from an definition that might originate from
        // another module, hence we cannot compare the symbols directly
        mi::mdl::ISymbol const *size_sym = size->get_size_symbol();

        if (strcmp(size_sym->get_name(), sym->get_name()) == 0)
            return p_def;
    }
    return NULL;
}

// Get the "unreachable" block of the current function, create one if necessary.
llvm::BasicBlock *Function_context::get_unreachable_bb()
{
    if (m_unreachable_bb == NULL) {
        m_unreachable_bb = create_bb("unreachable");
    }
    return m_unreachable_bb;
}

// Register a resource value and return its index.
size_t Function_context::get_resource_index(mi::mdl::IValue_resource const *resource)
{
    if (m_res_manager != NULL)
        return m_res_manager->get_resource_index(resource);

    // no resource manager, leave it "as is"
    return resource->get_tag_value();
}

// Get the array base address of an deferred-sized array.
llvm::Value *Function_context::get_deferred_base(llvm::Value *arr_desc)
{
    unsigned idxes[1] = { Type_mapper::ARRAY_DESC_BASE };
    return m_ir_builder.CreateExtractValue(arr_desc, idxes);
}

// Get the array size of an deferred-sized array.
llvm::Value *Function_context::get_deferred_size(llvm::Value *arr_desc)
{
    unsigned idxes[1] = { Type_mapper::ARRAY_DESC_SIZE };
    return m_ir_builder.CreateExtractValue(arr_desc, idxes);
}

// Get the array base address of an deferred-sized array.
llvm::Value *Function_context::get_deferred_base_from_ptr(llvm::Value *arr_desc_ptr)
{
    llvm::Value *idx      = get_constant(Type_mapper::ARRAY_DESC_BASE);
    llvm::Value *base_ptr = create_simple_gep_in_bounds(arr_desc_ptr, idx);
    return m_ir_builder.CreateLoad(base_ptr);
}

// Get the array size of an deferred-sized array.
llvm::Value *Function_context::get_deferred_size_from_ptr(llvm::Value *arr_desc_ptr)
{
    llvm::Value *idx      = get_constant(Type_mapper::ARRAY_DESC_SIZE);
    llvm::Value *size_adr = create_simple_gep_in_bounds(arr_desc_ptr, idx);
    return m_ir_builder.CreateLoad(size_adr);
}

// Set the array base address of an deferred-sized array.
void Function_context::set_deferred_base(llvm::Value *arr_desc_ptr, llvm::Value *arr)
{
    llvm::Value *idx      = get_constant(Type_mapper::ARRAY_DESC_BASE);
    llvm::Value *base_ptr = create_simple_gep_in_bounds(arr_desc_ptr, idx);

    // Note: because in LLVM world the pointer to an array is NOT equal to a pointer
    // to the first element, we need some casting here
    llvm::Type *tp = base_ptr->getType()->getPointerElementType();
    arr = m_ir_builder.CreateBitCast(arr, tp);
    m_ir_builder.CreateStore(arr, base_ptr);
}

// Set the array size of an deferred-sized array (type size_t).
void Function_context::set_deferred_size(llvm::Value *arr_desc_ptr, llvm::Value *size)
{
    llvm::Value *idx      = get_constant(Type_mapper::ARRAY_DESC_SIZE);
    llvm::Value *size_adr = create_simple_gep_in_bounds(arr_desc_ptr, idx);
    m_ir_builder.CreateStore(size, size_adr);
}

// Get a pointer type from a base type.
llvm::PointerType *Function_context::get_ptr(llvm::Type *type)
{
    return m_type_mapper.get_ptr(type);
}

// Get the number of elements of an aggregate type (struct or array).
static unsigned get_aggregate_num_elements(llvm::Type const *type)
{
    if (llvm::isa<llvm::ArrayType>(type))
        return unsigned(type->getArrayNumElements());
    MDL_ASSERT(llvm::isa<llvm::StructType>(type));
    return type->getStructNumElements();
}

#ifdef ENABLE_ASSERT   // the function is currently only used within an assertion

// Get the type of the first element of an aggregate type (struct or array).
static llvm::Type *get_aggregate_first_type(llvm::Type const *type)
{
    if (llvm::isa<llvm::ArrayType>(type))
        return type->getArrayElementType();
    MDL_ASSERT(llvm::isa<llvm::StructType>(type));
    return type->getStructElementType(0);
}

#endif  // ENABLE_ASSERT

// Load a value and convert the representation.
llvm::Value *Function_context::load_and_convert(llvm::Type *dst_type, llvm::Value *ptr)
{
    llvm::PointerType *src_ptr_type = llvm::cast<llvm::PointerType>(ptr->getType());
    llvm::Type        *src_type     = src_ptr_type->getElementType();

    if (src_type != dst_type) {
        // Type mismatch: Assume that the sizes matches for now or that the destination
        // type is a smaller one. Should be ok for our types ...

        if (src_type->isAggregateType() && llvm::isa<llvm::VectorType>(dst_type)) {
            // vector types could have a higher natural alignment, so load them by elements
            MDL_ASSERT(
                get_aggregate_num_elements(src_type) >= uint64_t(dst_type->getVectorNumElements())
            );

            llvm::Value *a   = m_ir_builder.CreateLoad(ptr);
            llvm::Value *res = llvm::UndefValue::get(dst_type);

            for (unsigned i = 0, n = dst_type->getVectorNumElements(); i < n; ++i) {
                unsigned idxs[] = { i };
                llvm::Value *v   = m_ir_builder.CreateExtractValue(a, idxs);
                res = m_ir_builder.CreateInsertElement(res, v, get_constant(int(i)));
            }
            return res;
        } else if (src_type->isAggregateType() && dst_type->isAggregateType()) {
            // as the types are different, one is a struct and one is an array.
            // The struct type could have any custom alignment, so load them by elements.

            // we only check the type of the first element of the struct...
            MDL_ASSERT(
                get_aggregate_num_elements(src_type) == get_aggregate_num_elements(dst_type) &&
                get_aggregate_first_type(src_type) == get_aggregate_first_type(dst_type));

            llvm::Value *a   = m_ir_builder.CreateLoad(ptr);
            llvm::Value *res = llvm::UndefValue::get(dst_type);

            for (unsigned i = 0, n = get_aggregate_num_elements(dst_type); i < n; ++i) {
                unsigned idxs[] = { i };
                llvm::Value *v   = m_ir_builder.CreateExtractValue(a, idxs);
                res = m_ir_builder.CreateInsertValue(res, v, idxs);
            }
            return res;
        } else if (llvm::isa<llvm::VectorType>(src_type) && dst_type->isAggregateType()) {
            // we only check the type of the first element of the struct...
            MDL_ASSERT(
                src_type->getVectorNumElements() == get_aggregate_num_elements(dst_type) &&
                src_type->getVectorElementType() == get_aggregate_first_type(dst_type));

            llvm::Value *a   = m_ir_builder.CreateLoad(ptr);
            llvm::Value *res = llvm::UndefValue::get(dst_type);

            for (unsigned i = 0, n = get_aggregate_num_elements(dst_type); i < n; ++i) {
                unsigned idxs[] = { i };
                llvm::Value *v   = m_ir_builder.CreateExtractElement(a, get_constant(int(i)));
                res = m_ir_builder.CreateInsertValue(res, v, idxs);
            }
            return res;
        } else if (llvm::isa<llvm::IntegerType>(src_type) &&
                dst_type == llvm::IntegerType::get(m_llvm_context, 1)) {
            // convert from integer to 1-bit bool
            llvm::Value *val = m_ir_builder.CreateLoad(ptr);
            return m_ir_builder.CreateICmpNE(val, llvm::ConstantInt::getNullValue(src_type));
        } else {
            // try casting the pointer
            MDL_ASSERT(
                // array to array, assume same alignment
                (llvm::isa<llvm::ArrayType>(src_type) && llvm::isa<llvm::ArrayType>(dst_type)) ||
                // vector to vector, assume same or higher alignment
                (llvm::isa<llvm::VectorType>(src_type) && llvm::isa<llvm::VectorType>(dst_type)) ||
                // first element from array
                (llvm::isa<llvm::ArrayType>(src_type) &&
                 src_type->getArrayElementType() == dst_type)
            );
            unsigned addr_space = src_ptr_type->getAddressSpace();
            llvm::PointerType *dst_ptr_type = dst_type->getPointerTo(addr_space);

            ptr = m_ir_builder.CreateBitCast(ptr, dst_ptr_type);
            return m_ir_builder.CreateLoad(ptr);
        }
    } else {
        // just Load
        return m_ir_builder.CreateLoad(ptr);
    }
}

// Convert a value and store it into a pointer location.
llvm::StoreInst *Function_context::convert_and_store(llvm::Value *value, llvm::Value *ptr)
{
    llvm::PointerType *dst_ptr_type = llvm::cast<llvm::PointerType>(ptr->getType());
    llvm::Type        *dst_type     = dst_ptr_type->getElementType();
    llvm::Type        *src_type     = value->getType();

    if (src_type != dst_type) {
        // Type mismatch: Assume that the sizes matches for now or that the destination
        // type is a smaller one. Should be ok for our types ...

        if (llvm::isa<llvm::VectorType>(src_type) && dst_type->isAggregateType()) {
            MDL_ASSERT(
                uint64_t(src_type->getVectorNumElements()) >= get_aggregate_num_elements(dst_type)
            );

            llvm::Value *a   = value;
            llvm::Value *res = llvm::UndefValue::get(dst_type);

            for (unsigned i = 0, n = get_aggregate_num_elements(dst_type); i < n; ++i) {
                unsigned idxs[] = { i };
                llvm::Value *v   = m_ir_builder.CreateExtractElement(a, get_constant(int(i)));
                res = m_ir_builder.CreateInsertValue(res, v, idxs);
            }
            value = res;
        } else if (src_type->isAggregateType() && dst_type->isAggregateType()) {
            // as the types are different, one is a struct and one is an array.
            // The struct type could have any custom alignment, so load them by elements.

            // we only check the type of the first element of the struct...
            MDL_ASSERT(
                get_aggregate_num_elements(src_type) == get_aggregate_num_elements(dst_type) &&
                get_aggregate_first_type(src_type) == get_aggregate_first_type(dst_type)
            );

            llvm::Value *a   = value;
            llvm::Value *res = llvm::UndefValue::get(dst_type);

            for (unsigned i = 0, n = get_aggregate_num_elements(dst_type); i < n; ++i) {
                unsigned idxs[] = { i };
                llvm::Value *v   = m_ir_builder.CreateExtractValue(a, idxs);
                res = m_ir_builder.CreateInsertValue(res, v, idxs);
            }
            value = res;
        } else if (src_type == llvm::IntegerType::get(m_llvm_context, 8) &&
                dst_type == llvm::IntegerType::get(m_llvm_context, 1) ) {
            // Convert from 8-bit bool to 1-bit bool
            value = m_ir_builder.CreateICmpNE(
                value, llvm::ConstantInt::getNullValue(value->getType()));
        } else {
            MDL_ASSERT(
                src_type->isAggregateType() && llvm::isa<llvm::VectorType>(dst_type) &&
                get_aggregate_num_elements(src_type) >= uint64_t(dst_type->getVectorNumElements())
            );

            llvm::Value *a   = value;
            llvm::Value *res = llvm::UndefValue::get(dst_type);

            for (unsigned i = 0, n = dst_type->getVectorNumElements(); i < n; ++i) {
                unsigned idxs[] = { i };
                llvm::Value *v   = m_ir_builder.CreateExtractValue(a, idxs);
                res = m_ir_builder.CreateInsertElement(res, v, get_constant(int(i)));
            }
            value = res;
        }
    }
    // now store the value
    return m_ir_builder.CreateStore(value, ptr);
}

// Store a int2(0,0) into a pointer location.
llvm::StoreInst *Function_context::store_int2_zero(llvm::Value *ptr)
{
    llvm::PointerType *dst_ptr_type = llvm::cast<llvm::PointerType>(ptr->getType());
    llvm::Type        *dst_type = dst_ptr_type->getElementType();

    llvm::Value *zero = NULL;
    if (dst_type->isAggregateType()) {
        zero = llvm::ConstantAggregateZero::get(dst_type);
    } else if (dst_type->isVectorTy()) {
        zero = llvm::ConstantVector::getSplat(
            2, llvm::ConstantInt::get(m_type_mapper.get_int_type(), 0));
    } else {
        MDL_ASSERT(!"unsupported int2 representation");

        zero = llvm::UndefValue::get(dst_type);
    }
    // now store the value
    return m_ir_builder.CreateStore(zero, ptr);
}

// Get the real return type of the function (i.e. does NOT return void for sret functions).
llvm::Type *Function_context::get_return_type() const
{
    if (m_retval_adr != NULL) {
        llvm::PointerType *ptr_type = llvm::cast<llvm::PointerType>(m_retval_adr->getType());
        return ptr_type->getElementType();
    }
    return m_function->getReturnType();
}

// Map type sizes due to function instancing.
int Function_context::instantiate_type_size(
    mi::mdl::IType const *type) const
{
    IType_array const *a_type = as<IType_array>(type);
    if (a_type != NULL && !a_type->is_immediate_sized()) {
        Array_size_map::const_iterator it(m_array_size_map.find(a_type->get_deferred_size()));
        if (it != m_array_size_map.end())
            return it->second;
    }
    return -1;
}

/// Create a mangled name for an Optix entity.
static mi::mdl::string optix_mangled_name(
    IAllocator *alloc,
    char const *nspc,
    char const *name)
{
    mi::mdl::string res("_ZN", alloc);

    char buffer[32];

    snprintf(buffer, sizeof(buffer), "%u", unsigned(strlen(nspc)));
    res.append(buffer);
    res.append(nspc);

    snprintf(buffer, sizeof(buffer), "%u", unsigned(strlen(name)));
    res.append(buffer);
    res.append(name);

    res.append('E');
    return res;
}

// Get the tex_lookup function for a given vtable index.
llvm::Value *Function_context::get_tex_lookup_func(
    llvm::Value                                    *self,
    mi::mdl::Type_mapper::Tex_handler_vtable_index index)
{

#define ARGS4(a,b,c,d)           "(" #a ", " #b ", " #c ", " #d ")"
#define ARGS5(a,b,c,d,e)         "(" #a ", " #b ", " #c ", " #d ", " #e ")"
#define ARGS8(a,b,c,d,e,f,g,h)   "(" #a ", " #b ", " #c ", " #d ", " #e ", " #f ", " #g ", " #h ")"
#define ARGSX(a,b,c,d,e,f,g,h,i,j) \
    "(" #a ", " #b ", " #c ", " #d ", " #e ", " #f ", " #g ", " #h ", " #i ", " #j ")"

#define OCP(args) "rtCallableProgramId<void " args ">"

#define ARGS_lookup_float4_2d \
    ARGS8( \
        float result[4], \
        Core_tex_handler const *self, \
        unsigned texture_idx, \
        float const coord[2], \
        tex_wrap_mode const wrap_u, \
        tex_wrap_mode const wrap_v, \
        float const crop_u[2], \
        float const crop_v[2])

#define ARGS_lookup_float3_2d \
    ARGS8( \
        float result[3], \
        Core_tex_handler const *self, \
        unsigned texture_idx, \
        float const coord[2], \
        tex_wrap_mode const wrap_u, \
        tex_wrap_mode const wrap_v, \
        float const crop_u[2], \
        float const crop_v[2])

#define ARGS_texel_2d \
    ARGS5( \
        float result[4], \
        Core_tex_handler const *self, \
        unsigned texture_idx, \
        int const coord[2], \
        int const uv_tile[2])

#define ARGS_lookup_float4_3d \
    ARGSX( \
        float result[4], \
        Core_tex_handler const *self, \
        unsigned texture_idx, \
        float const coord[3], \
        tex_wrap_mode const wrap_u, \
        tex_wrap_mode const wrap_v, \
        tex_wrap_mode const wrap_w, \
        float const crop_u[2], \
        float const crop_v[2], \
        float const crop_w[2])

#define ARGS_lookup_float3_3d \
    ARGSX( \
        float result[3], \
        Core_tex_handler const *self, \
        unsigned texture_idx, \
        float const coord[3], \
        tex_wrap_mode const wrap_u, \
        tex_wrap_mode const wrap_v, \
        tex_wrap_mode const wrap_w, \
        float const crop_u[2], \
        float const crop_v[2], \
        float const crop_w[2])

#define ARGS_texel_3d \
    ARGS4(float result[4], Core_tex_handler const *self, unsigned texture_idx, int const coord[3])

#define ARGS_lookup_float4_cube \
    ARGS4(float result[4], Core_tex_handler const *self, unsigned texture_idx, int const coord[3])

#define ARGS_lookup_float3_cube \
    ARGS4(float result[3], Core_tex_handler const *self, unsigned texture_idx, int const coord[3])

#define ARGS_resolution_2d \
    ARGS4(int result[2], Core_tex_handler const *self, unsigned texture_idx, int const uv_tile[2])

    static struct {
        const char *name;
        const char *optix_typename;
    } names[] = {
        { "tex_lookup_float4_2d",   OCP(ARGS_lookup_float4_2d) },
        { "tex_lookup_float3_2d",   OCP(ARGS_lookup_float3_2d) },
        { "tex_texel_float4_2d",    OCP(ARGS_texel_2d) },
        { "tex_lookup_float4_3d",   OCP(ARGS_lookup_float4_3d) },
        { "tex_lookup_float3_3d",   OCP(ARGS_lookup_float3_3d) },
        { "tex_texel_float4_3d",    OCP(ARGS_texel_3d) },
        { "tex_lookup_float4_cube", OCP(ARGS_lookup_float4_cube) },
        { "tex_lookup_float3_cube", OCP(ARGS_lookup_float3_cube) },
        { "tex_resolution_2d",      OCP(ARGS_resolution_2d) }
    };

#undef ARGS_resolution_2d
#undef ARGS_lookup_float3_cube
#undef ARGS_lookup_float4_cube
#undef ARGS_texel_3d
#undef ARGS_lookup_float3_3d
#undef ARGS_lookup_float4_3d
#undef ARGS_texel_2d
#undef ARGS_lookup_float3_2d
#undef ARGS_lookup_float4_2d
#undef OCP
#undef ARGSX
#undef ARGS8
#undef ARGS4

    switch (m_tex_loookup_call_mode) {
    case TLCM_VTABLE:
        {
            // get the vtable from this
            llvm::Value *vt_adr = create_simple_gep_in_bounds(
                self, get_constant(Type_mapper::TH_VTABLE));
            llvm::Value *vtable = m_ir_builder.CreateLoad(vt_adr);

            // get the function at index from the vtable
            llvm::Value *f_adr  = create_simple_gep_in_bounds(
                vtable, get_constant(index));
            return m_ir_builder.CreateLoad(f_adr);
        }
        break;
    case TLCM_DIRECT:
        // call it directly
        if (m_code_gen.m_tex_lookup_functions[index] == NULL) {
            // prototype was not yet registered, do it
            llvm::StructType *self_type        = m_type_mapper.get_core_tex_handler_type();
            llvm::PointerType *vtable_ptr_type =
                llvm::cast<llvm::PointerType>(self_type->getElementType(0));
            llvm::StructType *vtable_type      =
                llvm::cast<llvm::StructType>(vtable_ptr_type->getElementType());

            llvm::PointerType *fp_type =
                llvm::cast<llvm::PointerType>(vtable_type->getElementType(index));
            llvm::FunctionType *f_type = llvm::cast<llvm::FunctionType>(fp_type->getElementType());

            // declare it with external linkage
            m_code_gen.m_tex_lookup_functions[index] = llvm::Function::Create(
                f_type,
                llvm::GlobalValue::ExternalLinkage,
                names[index].name,
                m_code_gen.m_module);
        }
        return m_code_gen.m_tex_lookup_functions[index];
    case TLCM_OPTIX_CP:
        if (m_code_gen.m_optix_cps[index] == NULL) {
            // generate entry for this bindless callable program

            // namespace rti_internal_typeinfo {
            //    __device__ ::rti_internal_typeinfo::rti_typeinfo name = {
            //        ::rti_internal_typeinfo::_OPTIX_VARIABLE, sizeof(type)};
            // }
            {
                llvm::StructType *tp   = m_type_mapper.get_optix_typeinfo_type();
                llvm::Constant   *init = llvm::ConstantStruct::get(
                    tp,
                    get_constant(/*_OPTIX_VARIABLE=*/0x796152),
                    get_constant(/*sizeof(rtCallableProgramId<T>)=*/4),
                    NULL
                    );

                new llvm::GlobalVariable(
                    *m_code_gen.m_module,
                    tp,
                    /*isConstant=*/false,
                    llvm::GlobalValue::ExternalLinkage,
                    init,
                    optix_mangled_name(
                        get_allocator(), "rti_internal_typeinfo", names[index].name).c_str());
            }

            // namespace rti_internal_typename {
            //    __device__ char name[] = #type;
            // }
            {
                llvm::IntegerType *char_tp = m_type_mapper.get_char_type();
                char const *data = names[index].optix_typename;

                size_t l = strlen(data);

                mi::mdl::vector<llvm::Constant *>::Type initializer(get_allocator());

                for (size_t i = 0; i <= l; ++i) {
                    initializer.push_back(llvm::ConstantInt::get(char_tp, data[i]));
                }

                llvm::ArrayType *tp = llvm::ArrayType::get(char_tp, l + 1);
                llvm::Constant  *init = llvm::ConstantArray::get(tp, initializer);

                new llvm::GlobalVariable(
                    *m_code_gen.m_module,
                    tp,
                    /*isConstant=*/false,
                    llvm::GlobalValue::ExternalLinkage,
                    init,
                    optix_mangled_name(
                        get_allocator(), "rti_internal_typename", names[index].name).c_str());
            }

            // namespace rti_internal_typeenum {
            //    __device__ int name = ::rti_internal_typeinfo::rti_typeenum<type>::m_typeenum;
            // }
            {
                llvm::IntegerType *tp = m_type_mapper.get_int_type();
                llvm::ConstantInt *init = llvm::ConstantInt::get(
                    tp, /*_OPTIX_TYPE_ENUM_PROGRAM_ID=*/4920);

                new llvm::GlobalVariable(
                    *m_code_gen.m_module,
                    tp,
                    /*isConstant=*/false,
                    llvm::GlobalValue::ExternalLinkage,
                    init,
                    optix_mangled_name(
                        get_allocator(), "rti_internal_typeenum", names[index].name).c_str());
            }

            // namespace rti_internal_semantic {
            // __device__ char name[] = #semantic;
            // }
            {
                llvm::IntegerType *char_tp = m_type_mapper.get_char_type();
                llvm::Constant    *initializer = llvm::ConstantInt::get(char_tp, 0); // ""

                llvm::ArrayType *tp   = llvm::ArrayType::get(char_tp, 1);
                llvm::Constant  *init = llvm::ConstantArray::get(tp, initializer);

                new llvm::GlobalVariable(
                    *m_code_gen.m_module,
                    tp,
                    /*isConstant=*/false,
                    llvm::GlobalValue::ExternalLinkage,
                    init,
                    optix_mangled_name(
                    get_allocator(), "rti_internal_semantic", names[index].name).c_str());
            }

            // namespace rti_internal_annotation {
            // __device__ char name[] = #annotation;
            // }
            {
                llvm::IntegerType *char_tp = m_type_mapper.get_char_type();
                llvm::Constant    *initializer = llvm::ConstantInt::get(char_tp, 0); // ""

                llvm::ArrayType *tp   = llvm::ArrayType::get(char_tp, 1);
                llvm::Constant  *init = llvm::ConstantArray::get(tp, initializer);

                new llvm::GlobalVariable(
                    *m_code_gen.m_module,
                    tp,
                    /*isConstant=*/false,
                    llvm::GlobalValue::ExternalLinkage,
                    init,
                    optix_mangled_name(
                    get_allocator(), "rti_internal_annotation", names[index].name).c_str());
            }

            // __device__ rtCallableProgramId<T> name
            {
                llvm::IntegerType *int_tp = m_type_mapper.get_int_type();

                llvm::Type *members[] = {
                    int_tp,        // m_id member
                };

                llvm::StructType *tp = llvm::StructType::create(
                    m_type_mapper.get_llvm_context(),
                    members,
                    names[index].optix_typename,
                    /*is_packed=*/false);

                llvm::Constant *init = llvm::ConstantStruct::get(
                    tp, llvm::ConstantInt::get(int_tp, 0), NULL);

                m_code_gen.m_optix_cps[index] = new llvm::GlobalVariable(
                    *m_code_gen.m_module,
                    tp,
                    /*isConstant=*/false,
                    llvm::GlobalValue::ExternalLinkage,
                    init,
                    names[index].name);
            }
        }
        {
            // call rt_callable_program_from_id(name.m_id)
            llvm::Value *id_ptr = create_simple_gep_in_bounds(
                m_code_gen.m_optix_cps[index], get_constant(0));
            llvm::Value *id     = m_ir_builder.CreateLoad(id_ptr);

#if 0
            llvm::Function *rt_callable_program_from_id_64 = m_code_gen.get_optix_cp_from_id();
            llvm::Value *f_ptr  = m_ir_builder.CreateCall(rt_callable_program_from_id_64, id);
#else
            llvm::PointerType *void_ptr_tp = m_type_mapper.get_void_ptr_type();
            llvm::IntegerType *int_tp      = m_type_mapper.get_int_type();

            llvm::FunctionType *f_tp = llvm::FunctionType::get(
                void_ptr_tp, int_tp, /*is_VarArg=*/false);

            llvm::InlineAsm *ia = llvm::InlineAsm::get(
                f_tp,
                "call ($0), _rt_callable_program_from_id_64, ($1);",
                "=l,r",
                /*hasSideEffects=*/false);
            llvm::Value *f_ptr = m_ir_builder.CreateCall(ia, id);
#endif

            // and cast it to the right type for LLVM
            llvm::StructType *self_type        = m_type_mapper.get_core_tex_handler_type();
            llvm::PointerType *vtable_ptr_type =
                llvm::cast<llvm::PointerType>(self_type->getElementType(0));
            llvm::StructType *vtable_type      =
                llvm::cast<llvm::StructType>(vtable_ptr_type->getElementType());

            llvm::PointerType *fp_type =
                llvm::cast<llvm::PointerType>(vtable_type->getElementType(index));

            return m_ir_builder.CreatePointerCast(f_ptr, fp_type);
        }
        break;
    }

    MDL_ASSERT(!"unsupported call mode");
    llvm::StructType *self_type        = m_type_mapper.get_core_tex_handler_type();
    llvm::PointerType *vtable_ptr_type =
        llvm::cast<llvm::PointerType>(self_type->getElementType(0));
    llvm::StructType *vtable_type      =
        llvm::cast<llvm::StructType>(vtable_ptr_type->getElementType());

    llvm::PointerType *fp_type =
        llvm::cast<llvm::PointerType>(vtable_type->getElementType(index));
    llvm::FunctionType *f_type = llvm::cast<llvm::FunctionType>(fp_type->getElementType());

    return llvm::UndefValue::get(f_type);
}

// Get the current line (computed from current position debug info)
int Function_context::get_dbg_curr_line()
{
    int curr_line = 0;
    if (m_curr_pos != NULL)
        curr_line = m_curr_pos->get_start_line();
    return curr_line;
}

// Get the current file name (computed from current position debug info)
char const *Function_context::get_dbg_curr_filename()
{
    mi::mdl::IModule const *mod = m_code_gen.tos_module();
    char const *fname = mod->get_filename();
    if (fname != NULL && fname[0] == '\0')
        fname = NULL;

    if (fname == NULL /* && is_compilerowned */) {
        // built-in modules and string modules have no file name, which is bad
        // especially for base, try to fix that.
        char const *mod_name = mod->get_name();

        if (mod_name[0] == ':' && mod_name[1] == ':')
            mod_name += 2;

        switch (mod_name[0]) {
        case 'a':
            if (strcmp("anno", mod_name) == 0)
                fname = "anno.mdl";
            break;
        case 'b':
            if (strcmp("base", mod_name) == 0)
                fname = "base.mdl";
            break;
        case 'd':
            if (strcmp("df", mod_name) == 0)
                fname = "df.mdl";
            else if (strcmp("debug", mod_name) == 0)
                fname = "debug.mdl";
            break;
        case 'l':
            if (strcmp("limits", mod_name) == 0)
                fname = "limits.mdl";
            break;
        case 'm':
            if (strcmp("math", mod_name) == 0)
                fname = "math.mdl";
            break;
        case 'n':
            if (strcmp("noise", mod_name) == 0)
                fname = "noise.mdl";
            else if (strcmp("nvidia::df", mod_name) == 0)
                fname = "nvidia/df.mdl";
            break;
        case 's':
            if (strcmp("state", mod_name) == 0)
                fname = "state.mdl";
            else if (strcmp("std", mod_name) == 0)
                fname = "std.mdl";
            break;
        case 't':
            if (strcmp("tex", mod_name) == 0)
                fname = "tex.mdl";
            break;
        }
    }

    return fname == NULL ? "<unknown>" : fname;
}

// Get the current function name.
char const *Function_context::get_dbg_curr_function()
{
    return m_function->getName().str().c_str();
}

/// Returns true, if the value \p val is a constant with a value equal to \p int_val.
bool Function_context::is_constant_value(llvm::Value *val, int int_val)
{
    llvm::ConstantInt *const_int = llvm::dyn_cast<llvm::ConstantInt>(val);
    if (const_int == NULL) return false;

    int const_int_val = int(const_int->getValue().getZExtValue());
    return const_int_val == int_val;
}

}  // mdl
}  // mi
