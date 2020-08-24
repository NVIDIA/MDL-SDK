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

#include <algorithm>
#include <string>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/Support/raw_ostream.h>

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_compiler.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_tools.h>

#include "generator_jit_ast_compute.h"
#include "generator_jit_hlsl_function.h"
#include "generator_jit_hlsl_writer.h"
#include "generator_jit_streams.h"

namespace mi {
namespace mdl {
namespace hlsl {

/// The zero location.
static Location const zero_loc(0, 0);

// Constructor.
HLSLWriterPass::HLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    mi::mdl::IOutput_stream                              &out,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    bool                                                 enable_debug,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode)
: llvm::ModulePass(ID)
, m_alloc(alloc)
, m_type_mapper(type_mapper)
, m_out(out)
, m_exp_func_list(exp_func_list)
, m_hlsl_compiler(initialize(m_alloc))
, m_unit(impl_cast<Compilation_unit>(m_hlsl_compiler->create_unit("generated")))
, m_dg(alloc)
, m_curr_node(nullptr)
, m_curr_func(nullptr)
, m_decl_factory(m_unit->get_declaration_factory())
, m_expr_factory(m_unit->get_expression_factory())
, m_stmt_factory(m_unit->get_statement_factory())
, m_type_factory(m_unit->get_type_factory())
, m_value_factory(m_unit->get_value_factory())
, m_symbol_table(m_unit->get_symbol_table())
, m_def_tab(m_unit->get_definition_table())
, m_type_cache(0, Type2type_map::hasher(), Type2type_map::key_equal(), alloc)
, m_cur_start_block(nullptr)
, m_local_var_map(0, Variable_map::hasher(), Variable_map::key_equal(), alloc)
, m_global_var_map(0, Variable_map::hasher(), Variable_map::key_equal(), alloc)
, m_phi_var_in_out_map(0, Phi_map::hasher(), Phi_map::key_equal(), alloc)
, m_struct_constructor_map(0, Struct_map::hasher(), Struct_map::key_equal(), alloc)
, m_llvm_function_map(0, Function_map::hasher(), Function_map::key_equal(), alloc)
, m_out_def(nullptr)
, m_num_texture_spaces(num_texture_spaces)
, m_num_texture_results(num_texture_results)
, m_df_handle_slot_mode(df_handle_slot_mode)
, m_use_dbg(enable_debug)
, m_cur_data_layout(nullptr)
, m_ref_fnames(0, Ref_fname_id_map::hasher(), Ref_fname_id_map::key_equal(), alloc)
, m_struct_dbg_info(0, Struct_info_map::hasher(), Struct_info_map::key_equal(), alloc)
, m_next_unique_name_id(0)
{
}

void HLSLWriterPass::getAnalysisUsage(llvm::AnalysisUsage &usage) const
{
    usage.addRequired<llvm::hlsl::ASTComputePass>();
    usage.setPreservesAll();
}

bool HLSLWriterPass::runOnModule(llvm::Module &M)
{
    Store<llvm::DataLayout const *> layout_store(m_cur_data_layout, &M.getDataLayout());

    llvm::hlsl::ASTComputePass &ast_compute_pass = getAnalysis<llvm::hlsl::ASTComputePass>();

    m_def_tab.transition_to_scope(m_def_tab.get_predef_scope());

    // create all HLSL predefined entities first
    fillPredefinedEntities();

    m_def_tab.transition_to_scope(m_def_tab.get_global_scope());

    // collect type debug info
    process_type_debug_info(M);

    // create definitions for all user defined functions, so we can create a call graph
    // on the fly
    for (llvm::Function &func : M.functions()) {
        if (func.isDeclaration())
            continue;

        create_definition(&func);
    }

    for (llvm::Function &func : M.functions()) {
        if (func.isDeclaration())
            continue;

        llvm::hlsl::ASTFunction const *ast_func = ast_compute_pass.getASTFunction(&func);
        translate_function(ast_func);
    }

    MDL_ASSERT(m_def_tab.get_curr_scope() == m_def_tab.get_global_scope() && "Scope error");

    class Enter_func_decl : public DG_visitor {
    public:
        /// Pre-visit a node.
        void pre_visit(DG_node const *node) final {};

        /// Post-visit a node.
        void post_visit(DG_node const *node) final
        {
            hlsl::Definition  const *def  = node->get_definition();
            hlsl::Declaration       *decl = def->get_declaration();

            if (decl != nullptr)
                m_unit->add_decl(decl);
        }

        /// Constructor.
        Enter_func_decl(hlsl::Compilation_unit *unit) : m_unit(unit) {}

    private:
        hlsl::Compilation_unit *m_unit;
    };

    Enter_func_decl visitor(m_unit.get());
    m_dg.walk(visitor);

    // analyze and optimize it
    m_unit->analyze(*m_hlsl_compiler.get());

    mi::base::Handle<IPrinter> printer(m_hlsl_compiler->create_printer(&m_out));

    printer->enable_locations(m_use_dbg);
    printer->print(m_unit.get());

    string prototype(m_alloc);
    String_stream_writer writer(prototype);
    mi::base::Handle<IPrinter> prototype_printer(m_hlsl_compiler->create_printer(&writer));

    for (mi::mdl::LLVM_code_generator::Exported_function &exp_func : m_exp_func_list) {
        hlsl::Def_function *def = m_llvm_function_map[exp_func.func];

        // Update function name, which may have been changed due to duplicates or invalid characters
        exp_func.name = def->get_symbol()->get_name();

        prototype.clear();
        prototype_printer->print(def);
        exp_func.set_function_prototype(IGenerated_code_executable::PL_HLSL, prototype.c_str());
    }

    return false;
}

// Generate HLSL predefined entities into the definition table.
void HLSLWriterPass::fillPredefinedEntities()
{
    // This is a work-around function, so far it adds only the float3 and float4 constructors.

    MDL_ASSERT(m_def_tab.get_curr_scope() == m_def_tab.get_predef_scope());

    hlsl::Type_scalar *float_type  = m_type_factory.get_float();

    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    {
        hlsl::Type_function::Parameter p(float_type, hlsl::Type_function::Parameter::PM_IN);
        hlsl::Type_function::Parameter params[] = { p, p, p };
        hlsl::Type_function *func_type = m_type_factory.get_function(float3_type, params);

        m_def_tab.enter_function_definition(
            m_symbol_table.get_symbol("float3"),
            func_type,
            hlsl::Def_function::DS_ELEM_CONSTRUCTOR,
            NULL);
    }

    {
        hlsl::Type_vector *float4_type = m_type_factory.get_vector(float_type, 4);

        hlsl::Type_function::Parameter p(float_type, hlsl::Type_function::Parameter::PM_IN);
        hlsl::Type_function::Parameter params[] = { p, p, p, p };
        hlsl::Type_function *func_type = m_type_factory.get_function(float4_type, params);

        m_def_tab.enter_function_definition(
            m_symbol_table.get_symbol("float4"),
            func_type,
            hlsl::Def_function::DS_ELEM_CONSTRUCTOR,
            NULL);
    }
}

// Create the HLSL definition for a user defined LLVM function.
hlsl::Def_function *HLSLWriterPass::create_definition(llvm::Function *func)
{
    MDL_ASSERT(m_llvm_function_map.find(func) == m_llvm_function_map.end());

    llvm::FunctionType *llvm_func_type = func->getFunctionType();
    hlsl::Type         *ret_type = convert_type(llvm_func_type->getReturnType());
    hlsl::Type         *out_type = nullptr;

    llvm::SmallVector<hlsl::Type_function::Parameter, 8> params;

    if (hlsl::is<hlsl::Type_array>(ret_type)) {
        // HLSL does not support returning arrays, turn into an out parameter
        out_type = ret_type;
        ret_type = m_type_factory.get_void();

        params.push_back(hlsl::Type_function::Parameter(
            out_type,
            hlsl::Type_function::Parameter::PM_OUT));
    }

    // collect parameters for the function definition
    for (llvm::Argument &arg_it : func->args()) {
        llvm::Type *arg_llvm_type = arg_it.getType();
        hlsl::Type *param_type    = convert_type(arg_llvm_type);

        // skip void typed parameters
        if (hlsl::is<hlsl::Type_void>(param_type))
            continue;

        hlsl::Type_function::Parameter::Modifier param_mod = hlsl::Type_function::Parameter::PM_IN;

        if (llvm::isa<llvm::PointerType>(arg_llvm_type)) {
            if (arg_it.hasStructRetAttr()) {
                // the sret attribute marks "return" values, so OUT is enough
                param_mod = hlsl::Type_function::Parameter::PM_OUT;
            } else if (arg_it.onlyReadsMemory()) {
                // can be safely passed as an IN attribute IF noalias
                param_mod = hlsl::Type_function::Parameter::PM_IN;
            } else {
                // can be safely passed as INOUT IF noalias
                param_mod = hlsl::Type_function::Parameter::PM_INOUT;
            }
        }

        params.push_back(hlsl::Type_function::Parameter(param_type, param_mod));
    }

    // create the function definition
    hlsl::Symbol        *func_sym = get_unique_hlsl_sym(func->getName(), "func");
    hlsl::Type_function *func_type = m_type_factory.get_function(
        ret_type, Array_ref<hlsl::Type_function::Parameter>(params.data(), params.size()));
    hlsl::Def_function  *func_def = m_def_tab.enter_function_definition(
        func_sym, func_type, hlsl::Def_function::DS_UNKNOWN, &zero_loc);

    m_llvm_function_map[func] = func_def;

    return func_def;
}

// Get the definition for a LLVM function, if one exists.
hlsl::Def_function *HLSLWriterPass::get_definition(llvm::Function *func)
{
    Function_map::iterator it = m_llvm_function_map.find(func);
    if (it != m_llvm_function_map.end())
        return it->second;
    return nullptr;
}

// Generate HLSL AST for the given function.
void HLSLWriterPass::translate_function(llvm::hlsl::ASTFunction const *ast_func)
{
    llvm::Function *func = &ast_func->getFunction();

    Store<llvm::Function *> func_store(m_curr_func, func);

    hlsl::Def_function  *func_def  = get_definition(func);
    hlsl::Type_function *func_type = func_def->get_type();
    hlsl::Type          *ret_type  = func_type->get_return_type();
    hlsl::Type          *out_type  = NULL;

    // reset the name IDs
    m_next_unique_name_id = 0;

    if (hlsl::is<hlsl::Type_void>(ret_type) &&
        !func->getFunctionType()->getReturnType()->isVoidTy())
    {
        // return type was converted into out parameter
        hlsl::Type_function::Parameter *param = func_type->get_parameter(0);
        if (param->get_modifier() == hlsl::Type_function::Parameter::PM_OUT)
            out_type = param->get_type();
    }

    // create a new node for this function and make it current
    Store<DG_node *> curr_node(m_curr_node, m_dg.get_node(func_def));

    // create the declaration for the function
    hlsl::Type_name *ret_type_name = get_type_name(ret_type);

    if (func->getLinkage() == llvm::GlobalValue::InternalLinkage)
        ret_type_name->get_qualifier().set_storage_qualifier(hlsl::SQ_STATIC);

    hlsl::Name           *func_name = get_name(zero_loc, func_def->get_symbol());
    Declaration_function *decl_func = m_decl_factory.create_function(
        ret_type_name, func_name);

    func_def->set_declaration(decl_func);
    func_name->set_definition(func_def);

    // create the function body
    {
        hlsl::Definition_table::Scope_enter enter(m_def_tab, func_def);

        // now create the declarations
        unsigned first_param_ofs = 0;
        if (out_type != nullptr) {
            hlsl::Type_name         *param_type_name = get_type_name(out_type);
            hlsl::Declaration_param *decl_param = m_decl_factory.create_param(param_type_name);
            add_array_specifiers(decl_param, out_type);
            param_type_name->get_qualifier().set_parameter_qualifier(PQ_OUT);

            hlsl::Symbol *param_sym  = get_unique_hlsl_sym("p_result", "p_result");
            hlsl::Name   *param_name = get_name(zero_loc, param_sym);
            decl_param->set_name(param_name);

            m_out_def = m_def_tab.enter_parameter_definition(
                param_sym, out_type, &param_name->get_location());
            m_out_def->set_declaration(decl_param);
            param_name->set_definition(m_out_def);

            decl_func->add_param(decl_param);

            ++first_param_ofs;
        }

        for (llvm::Argument &arg_it : func->args()) {
            llvm::Type *arg_llvm_type = arg_it.getType();
            hlsl::Type *param_type    = convert_type(arg_llvm_type);

            if (hlsl::is<hlsl::Type_void>(param_type)) {
                // skip void typed parameters
                continue;
            }

            unsigned                i                = arg_it.getArgNo();
            hlsl::Type_name         *param_type_name = get_type_name(param_type);
            hlsl::Declaration_param *decl_param = m_decl_factory.create_param(param_type_name);
            add_array_specifiers(decl_param, param_type);

            hlsl::Type_function::Parameter *param = func_type->get_parameter(i + first_param_ofs);
            hlsl::Parameter_qualifier param_qualifier = hlsl::PQ_NONE;
            switch (param->get_modifier()) {
            case hlsl::Type_function::Parameter::PM_IN:
                param_qualifier = hlsl::PQ_IN;
                break;
            case hlsl::Type_function::Parameter::PM_OUT:
                param_qualifier = hlsl::PQ_OUT;
                break;
            case hlsl::Type_function::Parameter::PM_INOUT:
                param_qualifier = hlsl::PQ_INOUT;
                break;
            }
            param_type_name->get_qualifier().set_parameter_qualifier(param_qualifier);

            char templ[16];
            snprintf(templ, sizeof(templ), "p_%u", i);

            hlsl::Symbol *param_sym = get_unique_hlsl_sym(arg_it.getName(), templ);
            hlsl::Name   *param_name = get_name(zero_loc, param_sym);
            decl_param->set_name(param_name);

            hlsl::Def_param *param_def = m_def_tab.enter_parameter_definition(
                param_sym, param_type, &param_name->get_location());
            param_def->set_declaration(decl_param);
            param_name->set_definition(param_def);

            m_local_var_map[&arg_it] = param_def;

            decl_func->add_param(decl_param);
        }

        // local variables possibly used in outer scopes will be declared in the
        // first block of the function
        m_cur_start_block = m_stmt_factory.create_compound(zero_loc);

        hlsl::Stmt *stmt = translate_region(ast_func->getBody());
        hlsl::Stmt *body = join_statements(m_cur_start_block, stmt);
        decl_func->set_body(body);
    }

    // cleanup
    m_cur_start_block = nullptr;
    m_out_def         = nullptr;
    m_local_var_map.clear();
}

// Translate a region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_region(llvm::hlsl::Region const *region)
{
    switch (region->get_kind()) {
    case llvm::hlsl::Region::SK_BLOCK:
        return translate_block(region);
    case llvm::hlsl::Region::SK_SEQUENCE:
        {
            llvm::hlsl::RegionSequence const *seq =
                llvm::hlsl::cast<llvm::hlsl::RegionSequence>(region);
            hlsl::Stmt *stmt = translate_region(seq->getHead());
            for (llvm::hlsl::Region const *next_region: seq->getTail()) {
                hlsl::Stmt *next_stmt = translate_region(next_region);
                stmt = join_statements(stmt, next_stmt);
            }
            return stmt;
        }
    case llvm::hlsl::Region::SK_NATURAL_LOOP:
        return translate_natural(llvm::hlsl::cast<llvm::hlsl::RegionNaturalLoop>(region));
    case llvm::hlsl::Region::SK_IF_THEN:
        return translate_if_then(llvm::hlsl::cast<llvm::hlsl::RegionIfThen>(region));
    case llvm::hlsl::Region::SK_IF_THEN_ELSE:
        return translate_if_then_else(llvm::hlsl::cast<llvm::hlsl::RegionIfThenElse>(region));
    case llvm::hlsl::Region::SK_BREAK:
        return translate_break(llvm::hlsl::cast<llvm::hlsl::RegionBreak>(region));
    case llvm::hlsl::Region::SK_CONTINUE:
        return translate_continue(llvm::hlsl::cast<llvm::hlsl::RegionContinue>(region));
    case llvm::hlsl::Region::SK_RETURN:
        return translate_return(llvm::hlsl::cast<llvm::hlsl::RegionReturn>(region));
    default:
        MDL_ASSERT(!"Region kind not supported, yet");
        return m_stmt_factory.create_invalid(zero_loc);
    }
}

/// Check if the given instruction is a InsertValue instruction and all its users are
/// InsertValues too.
static bool part_of_InsertValue_chain(llvm::Instruction &inst)
{
    llvm::InsertValueInst *iv = llvm::dyn_cast<llvm::InsertValueInst>(&inst);
    if (iv == nullptr)
        return false;

    for (llvm::Value *user : iv->users()) {
        if (!llvm::isa<llvm::InsertValueInst>(user))
            return false;
    }
    return true;
}

// Translate a block-region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_block(llvm::hlsl::Region const *region)
{
    llvm::BasicBlock *bb = region->get_bb();

    // Assumption: MDL has no pointers and uses copy-by-value, thus there are no aliases.
    //
    // Generate temporaries for
    //  - all loads which have not been used, yet, when a store to that pointer should be emitted
    //  - values which are used multiple times
    //  - all live-outs of the block (for phi inputs, use one input variable per phi)
    //
    // Generate statements for
    //  - assigning the temporaries
    //  - store instructions
    //  - call instructions to non-readonly functions

    // Scheduled instructions to generate.
    // Unless they are store instructions, the result needs to be stored in a temporary
    llvm::SmallVector<llvm::Value *, 8> gen_insts;
    llvm::SmallPtrSet<llvm::Value *, 8> gen_insts_set;

    // Base pointers where functions like modf and sincos might have written to.
    // Any loads to these pointers must be scheduled for generation.
    llvm::SmallPtrSet<llvm::Value *, 8> dirty_base_pointers;
    for (llvm::Instruction &value : *bb) {
        if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(&value)) {
            if (llvm::Function *called_func = call->getCalledFunction()) {
                // FIXME: check by name is slow
                llvm::StringRef name = called_func->getName();
                if (name == "hlsl.modf" || name == "hlsl.sincos") {
                    // second parameter is the out parameter
                    llvm::Value *out_pointer = call->getArgOperand(1);
                    llvm::Value *base_pointer = get_base_pointer(out_pointer);

                    // mark to enforce materializing loads
                    dirty_base_pointers.insert(base_pointer);
                } else {
                    // for other functions, check whether any pointers are provided to
                    // non-readonly parameters
                    for (unsigned i = 0; i < call->getNumArgOperands(); ++i) {
                        llvm::Value *arg = call->getArgOperand(i);
                        if (arg->getType()->isPointerTy() &&
                                !called_func->hasParamAttribute(i, llvm::Attribute::ReadOnly)) {
                            llvm::Value *base_pointer = get_base_pointer(arg);

                            // mark to enforce materializing loads
                            dirty_base_pointers.insert(base_pointer);
                        }
                    }
                }
            }
        }
    }

    for (llvm::Instruction &value : *bb) {
        // don't generate statements for vector element access of already available values
        if (llvm::ExtractElementInst *extract = llvm::dyn_cast<llvm::ExtractElementInst>(&value)) {
            llvm::Value *v = extract->getVectorOperand();
            if (has_local_var(v))
                continue;
            if (gen_insts_set.count(v) != 0)
                continue;
        }

        if (llvm::isa<llvm::AllocaInst>(value)) {
            create_local_var(&value);
            continue;
        }

        // don't generate statements for getelementptr, as there are no pointers in HLSL
        if (llvm::isa<llvm::GetElementPtrInst>(value)) {
            continue;
        }

        // PHI nodes are handled specially
        if (llvm::isa<llvm::PHINode>(value)) {
            continue;
        }

        bool gen_statement = false;

        if (llvm::isa<llvm::StoreInst>(value)) {
            // generate a statement for every store
            gen_statement = true;
        } else if (llvm::isa<llvm::ArrayType>(value.getType()) && !part_of_InsertValue_chain(value)) {
            // we need temporaries for every array typed return (except inner nodes of a
            // insertValue chain), because we do not have array typed literals in HLSL, only
            // compound expressions which can only be used in variable initializers
            gen_statement = true;
        } else if (llvm::isa<llvm::SelectInst>(value) &&
                llvm::isa<llvm::StructType>(value.getType())) {
            // conditional operator only supports results with numeric scalar, vector
            // or matrix types, so we need to generate an if-statement for it
            gen_statement = true;
        } else {
            if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(&value)) {
                if (llvm::Function *called_func = call->getCalledFunction()) {
                    // we need to handle memset and memcpy intrinsics
                    switch (called_func->getIntrinsicID()) {
                    case llvm::Intrinsic::memset:
                    case llvm::Intrinsic::memcpy:
                        gen_statement = true;
                        break;
                    case llvm::Intrinsic::dbg_addr:       // llvm.dbg.addr
                    case llvm::Intrinsic::dbg_declare:    // llvm.dbg.declare
                    case llvm::Intrinsic::dbg_label:      // llvm.dbg.label
                    case llvm::Intrinsic::dbg_value:      // llvm.dbg.value
                    case llvm::Intrinsic::lifetime_end:   // llvm.lifetime.end
                    case llvm::Intrinsic::lifetime_start: // llvm.lifetime.start
                        // no code for debug intrinsics
                        gen_statement = false;
                        break;

                    default:
                        {
                            llvm::Type *ret_type = called_func->getReturnType();
                            if (ret_type->isVoidTy()) {
                                // must be handled as a statement, or will be lost
                                gen_statement = true;
                            } else {
                                // FIXME: check by name is slow
                                llvm::StringRef name = called_func->getName();
                                if (name == "hlsl.modf" || name == "hlsl.sincos") {
                                    gen_statement = true;
                                }
                            }
                        }
                        break;
                    }
                } else {
                    // indirect call?
                    gen_statement = true;
                }
            }

            if (llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(&value)) {
                llvm::Value *base_pointer = get_base_pointer(load->getPointerOperand());
                if (dirty_base_pointers.count(base_pointer) != 0)
                    gen_statement = true;
            }

            if (!gen_statement) {
                // check users
                int num_users = 0;
                for (auto user : value.users()) {
                    // does inst have multiple users?
                    ++num_users;
                    if (num_users >= 2) {
                        gen_statement = true;
                        break;
                    }

                    llvm::Instruction *user_inst = llvm::cast<llvm::Instruction>(user);
                    // is inst a live-out?
                    if (llvm::cast<llvm::Instruction>(user)->getParent() != bb) {
                        gen_statement = true;
                        break;
                    }

                    switch (user_inst->getOpcode()) {
                    case llvm::Instruction::Call:
                        {
                            llvm::CallInst *call = llvm::cast<llvm::CallInst>(user_inst);
                            llvm::Function *called_func = call->getCalledFunction();

                            // don't let memset, memcpy or llvm.lifetime_end enforce materialization
                            // of instructions
                            llvm::Intrinsic::ID intrinsic_id =
                                called_func ? called_func->getIntrinsicID()
                                            : llvm::Intrinsic::not_intrinsic;
                            if (intrinsic_id != llvm::Intrinsic::memset &&
                                intrinsic_id != llvm::Intrinsic::memcpy &&
                                intrinsic_id != llvm::Intrinsic::lifetime_end &&
                                !call->onlyReadsMemory())
                            {
                                gen_statement = true;
                            }
                            break;
                        }
                    case llvm::Instruction::ShuffleVector:
                        gen_statement = true;
                        break;
                    }
                }
            }
        }

        if (gen_statement) {
            // skip any pointer casts
            llvm::Value *cur_val = &value;
            while (llvm::Instruction *cur_inst = llvm::dyn_cast<llvm::Instruction>(cur_val)) {
                if (!(cur_inst->isCast() && cur_inst->getType()->isPointerTy()))
                    break;
                cur_val = cur_inst->getOperand(0);
            }

            // only add to generate set, if it's really an instruction
            if (llvm::Instruction *cur_inst = llvm::dyn_cast<llvm::Instruction>(cur_val)) {
                // still don't generate statements for getelementptr, alloca and phis
                if (llvm::isa<llvm::GetElementPtrInst>(cur_inst) ||
                        llvm::isa<llvm::AllocaInst>(cur_inst) ||
                        llvm::isa<llvm::PHINode>(value)) {
                    continue;
                }

                if (gen_insts_set.count(cur_val) == 0) {
                    gen_insts.push_back(cur_inst);
                    gen_insts_set.insert(cur_inst);
                }
            }
        }
    }

    llvm::SmallVector<hlsl::Stmt *, 8> stmts;

    // check for phis and assign to their in-variables from the predecessor blocks to
    // the out-variables at the beginning of this block
    for (llvm::PHINode &phi : bb->phis()) {
        auto phi_in_out_vars = get_phi_vars(&phi);
        hlsl::Expr *phi_in_expr = create_reference(phi_in_out_vars.first);
        stmts.push_back(create_assign_stmt(phi_in_out_vars.second, phi_in_expr));
    }

    for (llvm::Value *value : gen_insts) {
        // handle intrinsic functions
        if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(value)) {
            if (translate_intrinsic_call(stmts, call))
                continue;
        }

        if (llvm::StoreInst *store = llvm::dyn_cast<llvm::StoreInst>(value)) {
            translate_store(stmts, store);
            continue;
        }

        if (llvm::SelectInst *sel = llvm::dyn_cast<llvm::SelectInst>(value)) {
            if (llvm::isa<llvm::StructType>(sel->getType())) {
                hlsl::Def_variable *var_def = create_local_var(value, /*do_not_register=*/true);
                hlsl::Stmt *stmt = translate_struct_select(sel, var_def);
                stmts.push_back(stmt);
                if (var_def != nullptr) {
                    m_local_var_map[value] = var_def;
                }
                continue;
            }
        }

        hlsl::Def_variable *var_def = create_local_var(
            value, /*do_not_register=*/ true, /*add_decl_statement=*/ false);

        hlsl::Expr *res = translate_expr(value);
        if (var_def != nullptr) {
            // don't initialize with itself
            if (!is_ref_to_def(res, var_def)) {
                // set variable initializer
                hlsl::Declaration_variable *decl_var = var_def->get_declaration();
                for (hlsl::Init_declarator &init_decl : *decl_var) {
                    if (init_decl.get_name()->get_symbol() == var_def->get_symbol()) {
                        init_decl.set_initializer(res);
                    }
                }

                // insert variable declaration here
                stmts.push_back(m_stmt_factory.create_declaration(decl_var));
            }

            // register now
            m_local_var_map[value] = var_def;
        } else {
            // for void calls, var_def is nullptr
            MDL_ASSERT(hlsl::is<hlsl::Expr_call>(res));

            // ignore reference expressions
            if (!hlsl::is<hlsl::Expr_ref>(res)) {
                stmts.push_back(m_stmt_factory.create_expression(res->get_location(), res));
            }
        }
    }

    // check for phis in successor blocks and assign to their in-variables at the end of this block
    for (llvm::BasicBlock *succ_bb : bb->getTerminator()->successors()) {
        for (llvm::PHINode &phi : succ_bb->phis()) {
            for (unsigned i = 0, n = phi.getNumIncomingValues(); i < n; ++i) {
                if (phi.getIncomingBlock(i) == bb) {
                    hlsl::Def_variable *phi_in_var = get_phi_in_var(&phi);
                    hlsl::Expr         *res        = translate_expr(phi.getIncomingValue(i));
                    stmts.push_back(create_assign_stmt(phi_in_var, res));
                }
            }
        }
    }

    if (stmts.size() > 1) {
        return m_stmt_factory.create_compound(
            zero_loc, Array_ref<hlsl::Stmt *>(stmts.data(), stmts.size()));
    } else if (stmts.size() == 1) {
        return stmts[0];
    } else {
        // no code was generated
        return m_stmt_factory.create_expression(zero_loc, nullptr);
    }
}

// Translate a natural loop into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_natural(llvm::hlsl::RegionNaturalLoop const *region)
{
    llvm::hlsl::Region *body      = region->getHead();
    hlsl::Stmt         *body_stmt = translate_region(body);
    hlsl::Expr         *cond      = nullptr;

    // TODO: currently disabled because of variable declaration at assignment
    //       (see deriv_tests::test_math_emission_color_2 with enabled derivatives)
#if 0
    // check if we can transform it into a do-while loop
    if (hlsl::Stmt_compound *c_body = hlsl::as<hlsl::Stmt_compound>(body_stmt)) {
        if (hlsl::Stmt *last = c_body->back()) {
            if (hlsl::Stmt_if *if_stmt = hlsl::as<hlsl::Stmt_if>(last)) {
                if (if_stmt->get_else_statement() == nullptr &&
                    hlsl::is<Stmt_break>(if_stmt->get_then_statement()))
                {
                    // the last statement of our loop is a if(cond) break, turn the natural loop
                    // into a do { ... } while(! cond)

                    cond = if_stmt->get_condition();
                    hlsl::Type *cond_type = cond->get_type();
                    cond = m_expr_factory.create_unary(
                        zero_loc,
                        hlsl::Expr_unary::OK_LOGICAL_NOT,
                        cond);
                    cond->set_type(cond_type);

                    // remove the if from the body
                    c_body->pop_back();
                }
            }
        }
    }
#endif

    if (cond == nullptr) {
        // create endless loop
        cond = m_expr_factory.create_literal(zero_loc, m_value_factory.get_bool(true));
    }
    return m_stmt_factory.create_do_while(zero_loc, cond, body_stmt);
}

// Translate a if-then-region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_if_then(llvm::hlsl::RegionIfThen const *region)
{
    llvm::hlsl::Region *head = region->getHead();
    hlsl::Stmt *head_stmt = translate_region(head);

    llvm::BranchInst *branch = llvm::cast<llvm::BranchInst>(region->get_terminator_inst());
    hlsl::Expr *cond_expr = translate_expr(branch->getCondition());

    if (region->isNegated()) {
        cond_expr = m_expr_factory.create_unary(
            cond_expr->get_location(), hlsl::Expr_unary::OK_LOGICAL_NOT, cond_expr);
    }

    hlsl::Stmt *then_stmt = translate_region(region->getThen());

    hlsl::Stmt *if_stmt = m_stmt_factory.create_if(
        cond_expr->get_location(), cond_expr, then_stmt, /*else_stmt=*/nullptr);

    return join_statements(head_stmt, if_stmt);
}

// Translate a if-then-else-region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_if_then_else(llvm::hlsl::RegionIfThenElse const *region)
{
    llvm::hlsl::Region *head = region->getHead();
    hlsl::Stmt *head_stmt = translate_region(head);

    llvm::BranchInst *branch = llvm::cast<llvm::BranchInst>(region->get_terminator_inst());
    hlsl::Expr *cond_expr = translate_expr(branch->getCondition());

    if (region->isNegated()) {
        MDL_ASSERT(!"if-then-else regions should not use negated");
        cond_expr = m_expr_factory.create_unary(
            cond_expr->get_location(), hlsl::Expr_unary::OK_LOGICAL_NOT, cond_expr);
    }

    hlsl::Stmt *then_stmt = translate_region(region->getThen());
    hlsl::Stmt *else_stmt = translate_region(region->getElse());

    hlsl::Stmt *if_stmt = m_stmt_factory.create_if(
        cond_expr->get_location(), cond_expr, then_stmt, else_stmt);

    return join_statements(head_stmt, if_stmt);
}

// Translate a break-region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_break(llvm::hlsl::RegionBreak const *region)
{
    return m_stmt_factory.create_break(zero_loc);
}

// Translate a continue-region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_continue(llvm::hlsl::RegionContinue const *region)
{
    return m_stmt_factory.create_continue(zero_loc);
}

// Translate a return-region into HLSL AST.
hlsl::Stmt *HLSLWriterPass::translate_return(llvm::hlsl::RegionReturn const *region)
{
    llvm::hlsl::Region *head = region->getHead();
    hlsl::Stmt *head_stmt = translate_region(head);

    llvm::ReturnInst *ret_inst = region->get_return_inst();
    llvm::Value *ret_val = ret_inst->getReturnValue();

    hlsl::Expr *hlsl_ret_expr = nullptr;
    if (ret_val != nullptr) {
        hlsl_ret_expr = translate_expr(ret_val);
    }

    hlsl::Stmt *ret_stmt = nullptr;
    if (m_out_def != nullptr && hlsl_ret_expr != nullptr) {
        // return through an out parameter
        hlsl::Stmt *expr_stmt = create_assign_stmt(m_out_def, hlsl_ret_expr);
        ret_stmt = m_stmt_factory.create_return(convert_location(ret_inst), /*expr=*/nullptr);
        ret_stmt = join_statements(expr_stmt, ret_stmt);
    } else {
        ret_stmt = m_stmt_factory.create_return(convert_location(ret_inst), hlsl_ret_expr);
    }

    return join_statements(head_stmt, ret_stmt);
}

// Return the base pointer of the given pointer after skipping all bitcasts and GEPs.
llvm::Value *HLSLWriterPass::get_base_pointer(llvm::Value *pointer)
{
    while (true) {
        if (llvm::BitCastInst *bitcast = llvm::dyn_cast<llvm::BitCastInst>(pointer)) {
            pointer = bitcast->getOperand(0);
            continue;
        }
        if (llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(pointer)) {
            pointer = gep->getPointerOperand();
            continue;
        }

        // should be alloca or pointer parameter
        return pointer;
    }
}

// Recursive part of process_pointer_address implementation.
llvm::Value *HLSLWriterPass::process_pointer_address_recurse(
    Type_walk_stack &stack,
    llvm::Value *pointer,
    uint64_t write_size)
{
    llvm::Value *base_pointer;

    // skip bitcasts
    pointer = pointer->stripPointerCasts();

    if (llvm::GEPOperator *gep = llvm::dyn_cast<llvm::GEPOperator>(pointer)) {
        base_pointer = process_pointer_address_recurse(stack, gep->getPointerOperand(), write_size);
        if (base_pointer == nullptr)
            return nullptr;

        llvm::Type *cur_llvm_type = stack.back().field_type;
        uint64_t cur_type_size = 0;

        for (unsigned i = 1, num_indices = gep->getNumIndices(); i < num_indices; ++i) {
            // check whether we can stop going deeper into the compound
            cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);
            if (cur_type_size <= write_size) {
                // we can only stop if all of the remaining indices are zero
                bool all_zero = true;
                for (unsigned j = i; j < num_indices; ++j) {
                    if (llvm::ConstantInt *idx = llvm::dyn_cast<llvm::ConstantInt>(
                            gep->getOperand(j + 1))) {
                        if (!idx->isZero()) {
                            all_zero = false;
                            break;
                        }
                    } else {
                        all_zero = false;
                        break;
                    }
                }
                // we can stop here
                if (all_zero)
                    break;
            }

            llvm::Value *idx_val = gep->getOperand(i + 1);
            if (llvm::ConstantInt *idx = llvm::dyn_cast<llvm::ConstantInt>(idx_val))
            {
                unsigned idx_imm = unsigned(idx->getZExtValue());
                llvm::Type *field_type =
                    llvm::cast<llvm::CompositeType>(cur_llvm_type)->getTypeAtIndex(idx_imm);
                stack.push_back(Type_walk_element(cur_llvm_type, i, idx_val, 0, field_type));
                cur_llvm_type = field_type;
            } else {
                if (!llvm::isa<llvm::ArrayType>(cur_llvm_type) &&
                    !llvm::isa<llvm::VectorType>(cur_llvm_type))
                {
                    stack.clear();
                    return nullptr;
                }
                llvm::Type *field_type =
                    llvm::cast<llvm::CompositeType>(cur_llvm_type)->getTypeAtIndex(0u);
                stack.push_back(Type_walk_element(
                    cur_llvm_type, i, idx_val, 0, field_type));
                cur_llvm_type = field_type;
            }
        }
    } else {
        // should be alloca or pointer parameter
        base_pointer = pointer;

        llvm::Type *compound_type = pointer->getType()->getPointerElementType();
        stack.push_back(Type_walk_element(compound_type, 0, nullptr, 0, compound_type));
    }

    return base_pointer;
}

// Initialize the type walk stack for the given pointer and the size to be written
// and return the base pointer.
llvm::Value *HLSLWriterPass::process_pointer_address(
    Type_walk_stack &stack,
    llvm::Value *pointer,
    uint64_t write_size)
{
    llvm::Value *base_pointer = process_pointer_address_recurse(stack, pointer, write_size);

    llvm::Type *cur_llvm_type = stack.back().field_type;
    uint64_t cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);

    // still too big? check whether we can go deeper
    // example: %10 = bitcast [16 x <4 x float>]* %0 to i32*
    while (cur_type_size > write_size) {
        // check for array, vector or struct

        if (!llvm::isa<llvm::CompositeType>(cur_llvm_type))
            break;

        llvm::Type *first_type = cur_llvm_type->getContainedType(0);
        stack.push_back(Type_walk_element(
            cur_llvm_type, ~0, nullptr, 0, first_type));
        cur_llvm_type = first_type;
        cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);
    }

    return base_pointer;
}

// Create an lvalue expression for the compound element given by the type walk stack.
hlsl::Expr *HLSLWriterPass::create_compound_elem_expr(
    Type_walk_stack &stack,
    llvm::Value     *base_pointer)
{
    hlsl::Expr *cur_expr = translate_expr(base_pointer);
    hlsl::Type *cur_type = cur_expr->get_type()->skip_type_alias();

    // start after base element
    for (size_t i = 1, n = stack.size(); i < n; ++i) {
        hlsl::Type::Kind type_kind = cur_type->get_kind();
        if (type_kind == hlsl::Type::TK_ARRAY ||
            type_kind == hlsl::Type::TK_VECTOR ||
            type_kind == hlsl::Type::TK_MATRIX)
        {
            if (type_kind == hlsl::Type::TK_VECTOR &&
                    (stack[i].field_index_val == nullptr ||
                        llvm::isa<llvm::ConstantInt>(stack[i].field_index_val))) {
                unsigned idx_imm;
                if (stack[i].field_index_val) {
                    llvm::ConstantInt *idx = llvm::cast<llvm::ConstantInt>(stack[i].field_index_val);
                    idx_imm = unsigned(idx->getZExtValue()) + stack[i].field_index_offs;
                } else
                    idx_imm = stack[i].field_index_offs;
                cur_expr = create_vector_access(cur_expr, idx_imm);
            } else {
                hlsl::Expr *array_index;
                if (stack[i].field_index_val) {
                    array_index = translate_expr(stack[i].field_index_val);
                    if (stack[i].field_index_offs != 0) {
                        hlsl::Expr *offs = m_expr_factory.create_literal(
                            zero_loc,
                            m_value_factory.get_int32(int32_t(stack[i].field_index_offs)));
                        array_index = m_expr_factory.create_binary(
                            Expr_binary::OK_PLUS,
                            array_index,
                            offs);
                    }
                } else {
                    array_index = m_expr_factory.create_literal(
                        zero_loc, m_value_factory.get_int32(int32_t(stack[i].field_index_offs)));
                }
                cur_expr = m_expr_factory.create_binary(
                    Expr_binary::OK_ARRAY_SUBSCRIPT, cur_expr, array_index);
            }

            cur_type = hlsl::cast<hlsl::Type_compound>(cur_type)->get_compound_type(0);
            cur_type = cur_type->skip_type_alias();
            cur_expr->set_type(cur_type);
            continue;
        }

        if (hlsl::Type_struct *struct_type = hlsl::as<hlsl::Type_struct>(cur_type)) {
            unsigned idx_imm = stack[i].field_index_offs;
            if (stack[i].field_index_val) {
                llvm::ConstantInt *idx = llvm::cast<llvm::ConstantInt>(stack[i].field_index_val);
                idx_imm += unsigned(idx->getZExtValue());
            }
            hlsl::Type_struct::Field *field = struct_type->get_field(idx_imm);
            hlsl::Expr *field_ref = create_reference(
                field->get_symbol(), field->get_type());
            cur_expr = m_expr_factory.create_binary(
                Expr_binary::OK_SELECT, cur_expr, field_ref);

            cur_type = field->get_type()->skip_type_alias();
            cur_expr->set_type(cur_type);
            continue;
        }

        MDL_ASSERT(!"unexpected type kind");
        return m_expr_factory.create_invalid(zero_loc);
    }

    return cur_expr;
}

// Return true, if the index is valid for the given composite type.
bool HLSLWriterPass::is_valid_composite_index(llvm::Type *type, size_t index)
{
    if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(type)) {
        return index < st->getNumElements();
    }
    if (llvm::SequentialType *st = llvm::dyn_cast<llvm::SequentialType>(type)) {
        return index < st->getNumElements();
    }
    MDL_ASSERT(!"unexpected composite type");
    return false;
}

// Go to the next element in the stack.
bool HLSLWriterPass::move_to_next_compound_elem(Type_walk_stack &stack)
{
    Type_walk_element *cur_elem = &stack.back();
    while (true) {
        unsigned index;
        if (cur_elem->field_index_val) {
            MDL_ASSERT(llvm::isa<llvm::ConstantInt>(cur_elem->field_index_val) &&
                "field_index_val must be a constant to move to next compound element");
            llvm::ConstantInt *idx = llvm::cast<llvm::ConstantInt>(cur_elem->field_index_val);
            index = unsigned(idx->getZExtValue()) + cur_elem->field_index_offs;
        } else {
            index = cur_elem->field_index_offs;
        }

        if (is_valid_composite_index(cur_elem->llvm_type, index + 1)) {
            llvm::CompositeType *comp_type =
                llvm::cast<llvm::CompositeType>(cur_elem->llvm_type);
            ++cur_elem->field_index_offs;
            cur_elem->field_type = comp_type->getTypeAtIndex(index + 1);
            return true;
        }

        // try parent type
        stack.pop_back();
        if (stack.empty()) {
            MDL_ASSERT(!"memset out of bounds?");
            return false;
        }
        cur_elem = &stack.back();
    }
}

// Add statements to zero initializes the given lvalue expression
template <unsigned N>
void HLSLWriterPass::create_zero_init(
    llvm::SmallVector<hlsl::Stmt *, N> &stmts,
    hlsl::Expr *lval_expr)
{
    hlsl::Type *dest_type = lval_expr->get_type();
    if (hlsl::Type_array *arr_type = hlsl::as<hlsl::Type_array>(dest_type)) {
        hlsl::Type *elem_type = arr_type->get_element_type();
        for (size_t i = 0, n = arr_type->get_size(); i < n; ++i) {
            // as arrays of arrays are not allowed in MDL, we can use the "cast 0" approach
            hlsl::Expr *res = create_cast(
                elem_type,
                m_expr_factory.create_literal(zero_loc, m_value_factory.get_int32(0)));

            hlsl::Expr *array_index = m_expr_factory.create_literal(
                zero_loc, m_value_factory.get_int32(int32_t(i)));
            hlsl::Expr *array_elem = m_expr_factory.create_binary(
                Expr_binary::OK_ARRAY_SUBSCRIPT, lval_expr, array_index);
            array_elem->set_type(elem_type);

            stmts.push_back(create_assign_stmt(array_elem, res));
        }
    }
    else {
        // implement as casting 0 to the requested type
        hlsl::Expr *res = create_cast(
            dest_type,
            m_expr_factory.create_literal(zero_loc, m_value_factory.get_int32(0)));

        stmts.push_back(create_assign_stmt(lval_expr, res));
    }
}

// Translate a call into one or more statements, if it is a supported intrinsic call.
template <unsigned N>
bool HLSLWriterPass::translate_intrinsic_call(
    llvm::SmallVector<hlsl::Stmt *, N> &stmts,
    llvm::CallInst *call)
{
    llvm::Function *called_func = call->getCalledFunction();
    if (called_func == nullptr)
        return false;

    // we need to handle memset and memcpy intrinsics
    switch (called_func->getIntrinsicID()) {
    case llvm::Intrinsic::memset:
        {
            // expect the destination pointer to be a bitcast
            llvm::BitCastInst *bitcast = llvm::dyn_cast<llvm::BitCastInst>(call->getOperand(0));
            if (bitcast == nullptr)
                return false;

            // only handle as zero memory
            llvm::ConstantInt *fill_val = llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(1));
            if (fill_val == nullptr || !fill_val->isZero())
                return false;

            // only allow constant size
            llvm::ConstantInt *size_val = llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2));
            if (size_val == nullptr)
                return false;
            int64_t write_size = int64_t(size_val->getZExtValue());

            llvm::Value *dest = bitcast->getOperand(0);

            // get base element and go to largest sub-element which is smaller or equal to
            //     the write size
            // while (true)
            //   construct expression and write it and reduce write size
            //   if all written, done
            //   go to next element (maybe pop until there is a next element)

            Type_walk_stack stack;
            llvm::Value *base_pointer = process_pointer_address(stack, dest, write_size);

            uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
            while (write_size > 0) {
                hlsl::Expr *lval_expr = create_compound_elem_expr(stack, base_pointer);
                create_zero_init(stmts, lval_expr);

                Type_walk_element &cur_elem = stack.back();

                // reduce write_size by alignment and alloc size (includes internal padding)
                uint64_t total_size = cur_elem.get_total_size_and_update_offset(
                    m_cur_data_layout, cur_offs);
                write_size -= total_size;

                if (total_size == 0) {
                    MDL_ASSERT(!"invalid type size");
                    break;
                }

                // done?
                if (write_size <= 0)
                    break;

                // go to next element
                if (!move_to_next_compound_elem(stack))
                    return false;
            }
            return true;
        }
    case llvm::Intrinsic::memcpy:
        {
            // expect the destination pointer to be a bitcast
            llvm::BitCastInst *bitcast_dst = llvm::dyn_cast<llvm::BitCastInst>(call->getOperand(0));
            if (bitcast_dst == nullptr)
                return false;

            // expect the source pointer to be a bitcast (may be a bitcast instruction or value
            // (in case of a constant))
            llvm::Operator *bitcast_src = llvm::dyn_cast<llvm::Operator>(call->getOperand(1));
            if (bitcast_src->getOpcode() != llvm::Instruction::BitCast)
                return false;

            // only allow constant size
            llvm::ConstantInt *size_val = llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2));
            if (size_val == nullptr)
                return false;
            int64_t write_size = int64_t(size_val->getZExtValue());

            llvm::Value *dest = bitcast_dst->getOperand(0);
            llvm::Value *src = bitcast_src->getOperand(0);

            // get base element and go to largest sub-element which is smaller or equal to
            //     the write size
            // while (true)
            //   construct expression and write it and reduce write size
            //   if all written, done
            //   go to next element (maybe pop until there is a next element)

            Type_walk_stack stack_dest;
            Type_walk_stack stack_src;
            llvm::Value *base_pointer_dest = process_pointer_address(stack_dest, dest, write_size);
            llvm::Value *base_pointer_src = process_pointer_address(stack_src, src, write_size);

            uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
            while (write_size > 0) {
                hlsl::Expr *lval_expr = create_compound_elem_expr(stack_dest, base_pointer_dest);
                hlsl::Expr *rval_expr = create_compound_elem_expr(stack_src, base_pointer_src);
                stmts.push_back(create_assign_stmt(lval_expr, rval_expr));

                Type_walk_element &cur_elem_dest = stack_dest.back();

                // reduce write_size by alignment and alloc size (includes internal padding)
                MDL_ASSERT(cur_elem_dest.field_type == stack_src.back().field_type);
                uint64_t total_size = cur_elem_dest.get_total_size_and_update_offset(
                    m_cur_data_layout, cur_offs);
                write_size -= total_size;

                if (total_size == 0) {
                    MDL_ASSERT(!"invalid type size");
                    break;
                }

                // done?
                if (write_size <= 0)
                    break;

                // go to next element in both stacks
                if (!move_to_next_compound_elem(stack_dest))
                    return false;
                if (!move_to_next_compound_elem(stack_src))
                    return false;
            }
            return true;
        }
    default:
        return false;
    }
}

// Translate a struct select into an if statement writing to the given variable.
hlsl::Stmt *HLSLWriterPass::translate_struct_select(
    llvm::SelectInst *select,
    hlsl::Def_variable *dst_var)
{
    hlsl::Expr *cond       = translate_expr(select->getCondition());
    hlsl::Expr *true_expr  = translate_expr(select->getTrueValue());
    hlsl::Expr *false_expr = translate_expr(select->getFalseValue());

    hlsl::Location loc = convert_location(select);
    hlsl::Stmt *res = m_stmt_factory.create_if(
        loc,
        cond,
        create_assign_stmt(dst_var, true_expr),
        create_assign_stmt(dst_var, false_expr));
    return res;
}

// Convert the given HLSL value to the given HLSL type and return an HLSL expression for it.
hlsl::Expr *HLSLWriterPass::convert_to(hlsl::Expr *val, hlsl::Type *dest_type)
{
    hlsl::Type *src_type = val->get_type()->skip_type_alias();
    dest_type = dest_type->skip_type_alias();

    // first convert bool to int, if necessary
    if (hlsl::is<hlsl::Type_bool>(src_type) && hlsl::is<hlsl::Type_float>(dest_type)) {
        hlsl::Type *int_type = m_type_factory.get_int();
        val = create_cast(int_type, val);
        src_type = int_type;
    }

    if (hlsl::is<hlsl::Type_int>(src_type) && hlsl::is<hlsl::Type_float>(dest_type)) {
        hlsl::Type *float_type = m_type_factory.get_float();
        hlsl::Expr *func_ref = create_reference(get_sym("asfloat"), float_type);
        hlsl::Expr *expr_call = m_expr_factory.create_call(func_ref, { val });
        expr_call->set_type(float_type);
        return expr_call;
    }

    if (hlsl::is<hlsl::Type_float>(src_type) && hlsl::is<hlsl::Type_int>(dest_type)) {
        hlsl::Type *int_type = m_type_factory.get_int();
        hlsl::Expr *func_ref = create_reference(get_sym("asint"), int_type);
        hlsl::Expr *expr_call = m_expr_factory.create_call(func_ref, { val });
        expr_call->set_type(int_type);
        return expr_call;
    }

    return nullptr;
}

// Convert the given LLVM value to the given LLVM type and return an HLSL expression for it.
hlsl::Expr *HLSLWriterPass::convert_to(llvm::Value *val, llvm::Type *dest_type)
{
    llvm::Type *src_type = val->getType();

    if ((src_type->isIntegerTy() && dest_type->isFloatingPointTy()) ||
            (src_type->isFloatingPointTy() && dest_type->isIntegerTy())) {
        hlsl::Expr *hlsl_val = translate_expr(val);
        hlsl::Type *hlsl_dest_type = convert_type(dest_type);
        return convert_to(hlsl_val, hlsl_dest_type);
    }

    return nullptr;
}

// Translate an LLVM store instruction into one or more HLSL statements.
template <unsigned N>
void HLSLWriterPass::translate_store(
    llvm::SmallVector<hlsl::Stmt *, N> &stmts,
    llvm::StoreInst *inst)
{
    llvm::Value *value   = inst->getValueOperand();
    llvm::Value *pointer = inst->getPointerOperand();

    llvm::BitCastInst *nearest_bitcast = nullptr;
    int64_t target_size = 0;
    int64_t orig_size = 0;
    while (llvm::BitCastInst *bitcast = llvm::dyn_cast<llvm::BitCastInst>(pointer)) {
        // skip bitcasts but remember first target size
        if (target_size == 0) {
            nearest_bitcast = bitcast;
            target_size = m_cur_data_layout->getTypeStoreSize(
                bitcast->getDestTy()->getPointerElementType());
            orig_size = m_cur_data_layout->getTypeStoreSize(
                bitcast->getSrcTy()->getPointerElementType());
        }
        pointer = bitcast->getOperand(0);
    }

    if (target_size == 0) {
        // no bitcasts, so we can do a direct assignment
        hlsl::Expr *lvalue = translate_lval_expression(pointer);
        stmts.push_back(create_assign_stmt(lvalue, translate_expr(value)));
        return;
    }

    Type_walk_stack stack;
    llvm::Value *base_pointer = process_pointer_address(stack, pointer, target_size);

    // handle cases like:
    //   - bitcast float* %0 to i32*
    //   - bitcast [2 x float]* %weights.i.i.i to i32*
    //   - bitcast [16 x <4 x float>]* %0 to i32*
    if (m_cur_data_layout->getTypeStoreSize(stack.back().field_type) == target_size) {
        llvm::Type *src_type = stack.back().field_type;

        if (hlsl::Expr *rvalue = convert_to(value, src_type)) {
            hlsl::Expr *lvalue = create_compound_elem_expr(stack, base_pointer);
            stmts.push_back(create_assign_stmt(lvalue, rvalue));
            return;
        }
    }

    hlsl::Expr  *expr = nullptr;
    hlsl::Type  *expr_elem_type = nullptr;

    if (llvm::isa<llvm::VectorType>(value->getType())) {
        expr = translate_expr(value);
        if (hlsl::Type_vector *vt = hlsl::as<hlsl::Type_vector>(expr->get_type())) {
            expr_elem_type = vt->get_element_type();
        }
        if (expr_elem_type == nullptr) {
            MDL_ASSERT(!"only store with bitcast to vectors supported");
            return;
        }
    } else if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(value)) {
        if (!ci->isZero()) {
            MDL_ASSERT(!"currently only zero initialization supported");
            return;
        }
    } else {
        MDL_ASSERT(!"unexpected value type");
        return;
    }

    uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
    unsigned cur_vector_index = 0;
    while (target_size > 0) {
        hlsl::Expr *lval = create_compound_elem_expr(stack, base_pointer);
        if (hlsl::Type_vector *lval_vt = hlsl::as<hlsl::Type_vector>(lval->get_type())) {
            for (size_t i = 0, n = lval_vt->get_size(); i < n; ++i) {
                hlsl::Expr *lhs = create_vector_access(lval, i);
                hlsl::Expr *rhs;
                if (expr == nullptr) {
                    // zero initialization
                    rhs = m_expr_factory.create_literal(
                        convert_location(inst),
                        m_value_factory.get_zero_initializer(lhs->get_type()));
                } else {
                    rhs = create_vector_access(expr, cur_vector_index++);
                    expr = translate_expr(value);   // create new AST for the expression
                }
                stmts.push_back(create_assign_stmt(lhs, rhs));
            }
        } else {
            hlsl::Expr *rhs;
            if (expr == nullptr) {
                // zero initialization
                rhs = m_expr_factory.create_literal(
                    convert_location(inst),
                    m_value_factory.get_zero_initializer(lval->get_type()));
            } else {
                MDL_ASSERT(expr_elem_type == lval->get_type());
                rhs = create_vector_access(expr, cur_vector_index++);
                expr = translate_expr(value);   // create new AST for the expression
            }

            stmts.push_back(create_assign_stmt(lval, rhs));
        }

        Type_walk_element &cur_elem = stack.back();

        // reduce target_size by alignment and alloc size (includes internal padding)
        uint64_t total_size = cur_elem.get_total_size_and_update_offset(
            m_cur_data_layout, cur_offs);
        target_size -= total_size;

        if (total_size == 0) {
            MDL_ASSERT(!"invalid type size");
            break;
        }

        // done?
        if (target_size <= 0)
            break;

        // go to next element
        if (!move_to_next_compound_elem(stack)) {
            MDL_ASSERT(!"moving to next element failed");
            return;
        }
    }
}

// Translate an LLVM ConstantInt value to an HLSL value.
hlsl::Value *HLSLWriterPass::translate_constant_int(llvm::ConstantInt *ci)
{
    unsigned int bit_width = ci->getBitWidth();
    if (bit_width > 1 && bit_width < 16) {
        // allow comparison of "trunc i32 %Y to i2" with "i2 -1"
        // TODO: sign
        return m_value_factory.get_int16(int16_t(ci->getSExtValue()) & ((1 << bit_width) - 1));
    }

    if (bit_width > 16 && bit_width < 32) {
        // allow comparison of "trunc i32 %Y to i2" with "i2 -1"
        // TODO: sign
        return m_value_factory.get_int32(int32_t(ci->getSExtValue()) & ((1 << bit_width) - 1));
    }

    switch (bit_width) {
    case 1:
        return m_value_factory.get_bool(!ci->isZero());
    case 16:
        // TODO: sign
        return m_value_factory.get_int16(int16_t(ci->getSExtValue()));
    case 32:
        // TODO: sign
        return m_value_factory.get_int32(int32_t(ci->getSExtValue()));
    case 64:  // TODO: always treat as 32-bit, maybe not a good idea
        return m_value_factory.get_int32(int32_t(ci->getZExtValue()));
    }
    MDL_ASSERT(!"unexpected LLVM integer type");
    return m_value_factory.get_bad();
}

// Translate an LLVM ConstantFP value to an HLSL Expression.
hlsl::Value *HLSLWriterPass::translate_constant_fp(llvm::ConstantFP *cf)
{
    llvm::APFloat const &apfloat = cf->getValueAPF();

    switch (cf->getType()->getTypeID()) {
    case llvm::Type::HalfTyID:
        return m_value_factory.get_half(apfloat.convertToFloat());
    case llvm::Type::FloatTyID:
        return m_value_factory.get_float(apfloat.convertToFloat());
    case llvm::Type::DoubleTyID:
        return m_value_factory.get_double(apfloat.convertToDouble());
    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        MDL_ASSERT(!"unexpected float literal type");
        return m_value_factory.get_double(apfloat.convertToDouble());
    default:
        break;
    }
    MDL_ASSERT(!"invalid float literal type");
    return m_value_factory.get_bad();
}

// Translate an LLVM ConstantDataVector value to an HLSL Value.
hlsl::Value *HLSLWriterPass::translate_constant_data_vector(llvm::ConstantDataVector *cv)
{
    size_t num_elems = size_t(cv->getNumElements());
    hlsl::Type_vector *hlsl_type;
    Small_VLA<hlsl::Value_scalar *, 8> values(m_alloc, num_elems);

    switch (cv->getElementType()->getTypeID()) {
    case llvm::Type::IntegerTyID:
        {
            llvm::IntegerType *int_type = llvm::cast<llvm::IntegerType>(cv->getElementType());
            switch (int_type->getBitWidth()) {
            case 1:
            case 8:
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_bool(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_value_factory.get_bool(cv->getElementAsInteger(i) != 0);
                }
                break;
            case 16:
                // TODO: sign
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_min16int(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_value_factory.get_int16(
                        int16_t(cv->getElementAsAPInt(i).getSExtValue()));
                }
                break;
            case 32:
                // TODO: sign
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_int(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_value_factory.get_int32(
                        int32_t(cv->getElementAsAPInt(i).getSExtValue()));
                }
                break;
            case 64:  // always cast to signed 32-bit
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_int(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_value_factory.get_int32(
                        int32_t(cv->getElementAsAPInt(i).getZExtValue()));
                }
                break;
            default:
                MDL_ASSERT(!"invalid integer vector literal type");
                return m_value_factory.get_bad();
            }
            break;
        }

    case llvm::Type::HalfTyID:
        hlsl_type = m_type_factory.get_vector(m_type_factory.get_half(), num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = m_value_factory.get_half(cv->getElementAsFloat(i));
        }
        break;
    case llvm::Type::FloatTyID:
        hlsl_type = m_type_factory.get_vector(m_type_factory.get_float(), num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = m_value_factory.get_float(cv->getElementAsFloat(i));
        }
        break;

    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        MDL_ASSERT(!"unexpected float literal type");
        // fallthrough
    case llvm::Type::DoubleTyID:
        hlsl_type = m_type_factory.get_vector(m_type_factory.get_double(), num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = m_value_factory.get_double(cv->getElementAsDouble(i));
        }
        break;

    default:
        MDL_ASSERT(!"invalid vector literal type");
        return m_value_factory.get_bad();
    }

    return m_value_factory.get_vector(hlsl_type, values);
}

// Translate an LLVM ConstantDataArray value to an HLSL compound initializer.
hlsl::Expr *HLSLWriterPass::translate_constant_data_array(llvm::ConstantDataArray *cv)
{
    size_t num_elems = size_t(cv->getNumElements());
    hlsl::Type_vector *hlsl_type;
    Small_VLA<hlsl::Expr *, 8> values(m_alloc, num_elems);

    switch (cv->getElementType()->getTypeID()) {
    case llvm::Type::IntegerTyID:
        {
            llvm::IntegerType *int_type = llvm::cast<llvm::IntegerType>(cv->getElementType());
            switch (int_type->getBitWidth()) {
            case 1:
            case 8:
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_bool(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_expr_factory.create_literal(
                        zero_loc, m_value_factory.get_bool(cv->getElementAsInteger(i) != 0));
                }
                break;
            case 16:
                // TODO: sign
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_min16int(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_expr_factory.create_literal(
                        zero_loc,
                        m_value_factory.get_int16(
                            int16_t(cv->getElementAsAPInt(i).getSExtValue())));
                }
                break;
            case 32:
                // TODO: sign
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_int(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_expr_factory.create_literal(
                        zero_loc,
                        m_value_factory.get_int32(
                            int32_t(cv->getElementAsAPInt(i).getSExtValue())));
                }
                break;
            case 64:  // always cast to signed 32-bit
                hlsl_type = m_type_factory.get_vector(m_type_factory.get_int(), num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = m_expr_factory.create_literal(
                        zero_loc,
                        m_value_factory.get_int32(
                            int32_t(cv->getElementAsAPInt(i).getZExtValue())));
                }
                break;
            default:
                MDL_ASSERT(!"invalid integer vector literal type");
                return m_expr_factory.create_invalid(zero_loc);
            }
            break;
        }

    case llvm::Type::HalfTyID:
        hlsl_type = m_type_factory.get_vector(m_type_factory.get_half(), num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = m_expr_factory.create_literal(
                zero_loc,
                m_value_factory.get_half(cv->getElementAsFloat(i)));
        }
        break;
    case llvm::Type::FloatTyID:
        hlsl_type = m_type_factory.get_vector(m_type_factory.get_float(), num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = m_expr_factory.create_literal(
                zero_loc,
                m_value_factory.get_float(cv->getElementAsFloat(i)));
        }
        break;

    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        MDL_ASSERT(!"unexpected float literal type");
        // fallthrough
    case llvm::Type::DoubleTyID:
        hlsl_type = m_type_factory.get_vector(m_type_factory.get_double(), num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = m_expr_factory.create_literal(
                zero_loc,
                m_value_factory.get_double(cv->getElementAsDouble(i)));
        }
        break;

    default:
        MDL_ASSERT(!"invalid vector literal type");
        return m_expr_factory.create_invalid(zero_loc);
    }

    hlsl::Expr *res = m_expr_factory.create_compound(zero_loc, values);
    res->set_type(hlsl_type);
    return res;
}

// Translate an LLVM ConstantStruct value to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_constant_struct_expr(llvm::ConstantStruct *cv, bool is_global)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Small_VLA<hlsl::Expr *, 8> agg_elems(m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        agg_elems[i] = translate_constant_expr(elem, is_global);
    }

    hlsl::Type *res_type = convert_type(cv->getType());
    if (is_global) {
        hlsl::Expr *res = m_expr_factory.create_compound(zero_loc, agg_elems);
        res->set_type(res_type);
        return res;
    }
    return create_constructor_call(res_type, agg_elems, zero_loc);
}

// Translate an LLVM ConstantVector value to an HLSL Value.
hlsl::Value *HLSLWriterPass::translate_constant_vector(llvm::ConstantVector *cv)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Small_VLA<hlsl::Value_scalar *, 8> values(m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        values[i] = hlsl::cast<hlsl::Value_scalar>(translate_constant(elem));
    }

    return m_value_factory.get_vector(
        hlsl::cast<hlsl::Type_vector>(convert_type(cv->getType())), values);
}

// Translate an LLVM ConstantArray value to an HLSL compound expression.
hlsl::Expr *HLSLWriterPass::translate_constant_array(llvm::ConstantArray *cv, bool is_global)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Small_VLA<hlsl::Expr *, 8> values(m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        values[i] = translate_constant_expr(elem, is_global);
    }

    hlsl::Expr *res = m_expr_factory.create_compound(zero_loc, values);
    res->set_type(convert_type(cv->getType()));
    return res;
}

// Translate an LLVM ConstantAggregateZero value to an HLSL compound expression.
hlsl::Expr *HLSLWriterPass::translate_constant_array(
    llvm::ConstantAggregateZero *cv, bool is_global)
{
    llvm::ArrayType *at = llvm::cast<llvm::ArrayType>(cv->getType());
    size_t num_elems = size_t(at->getArrayNumElements());
    Small_VLA<hlsl::Expr *, 8> agg_elems(m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getElementValue(i);
        agg_elems[i] = translate_constant_expr(elem, is_global);
    }

    hlsl::Expr *res = m_expr_factory.create_compound(zero_loc, agg_elems);
    res->set_type(convert_type(cv->getType()));
    return res;
}

// Translate an LLVM ConstantArray value to an HLSL matrix value.
hlsl::Value *HLSLWriterPass::translate_constant_matrix(llvm::ConstantArray *cv)
{
    size_t num_elems = size_t(cv->getNumOperands());
    hlsl::Value_vector *values[4];

    hlsl::Type_matrix *mt = hlsl::cast<Type_matrix>(convert_type(cv->getType()));

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        values[i] = hlsl::cast<hlsl::Value_vector>(translate_constant(elem));
    }
    return m_value_factory.get_matrix(mt, Array_ref<hlsl::Value_vector *>(values, num_elems));
}

// Translate an LLVM ConstantAggregateZero value to an HLSL Value.
hlsl::Value *HLSLWriterPass::translate_constant_aggregate_zero(llvm::ConstantAggregateZero *cv)
{
    hlsl::Type *type = convert_type(cv->getType());
    return m_value_factory.get_zero_initializer(type);
}

// Translate an LLVM Constant value to an HLSL Value.
hlsl::Value *HLSLWriterPass::translate_constant(llvm::Constant *c)
{
    if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(c)) {
        return translate_constant_vector(cv);
    }
    if (llvm::isa<llvm::UndefValue>(c)) {
        hlsl::Type *type = convert_type(c->getType());

#if 0
        // use special undef values for debugging
        if (hlsl::is<hlsl::Type_float>(type)) {
            return m_value_factory.get_float(-107374176.0f);
        } else if(hlsl::is<hlsl::Type_double>(type)) {
            return m_value_factory.get_double(-9.25596313493e+61);
        } else if (hlsl::is<hlsl::Type_int>(type)) {
            return m_value_factory.get_int32(0xcccccccc);
        } else if (hlsl::is<hlsl::Type_uint>(type)) {
            return m_value_factory.get_uint32(0xcccccccc);
        }
#endif

        return m_value_factory.get_zero_initializer(type);
    }
    if (llvm::ConstantAggregateZero *cv = llvm::dyn_cast<llvm::ConstantAggregateZero>(c)) {
        return translate_constant_aggregate_zero(cv);
    }
    if (llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(c)) {
        return translate_constant_data_vector(cv);
    }
    if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(c)) {
        return translate_constant_int(ci);
    }
    if (llvm::ConstantFP *cf = llvm::dyn_cast<llvm::ConstantFP>(c)) {
        return translate_constant_fp(cf);
    }
    if (llvm::ConstantArray *cv = llvm::dyn_cast<llvm::ConstantArray>(c)) {
        llvm::ArrayType *at = cv->getType();
        if (is_matrix_type(at)) {
            return translate_constant_matrix(cv);
        }
    }
    MDL_ASSERT(!"Unsupported Constant kind");
    return m_value_factory.get_bad();
}

// Translate an LLVM Constant value to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_constant_expr(llvm::Constant *c, bool is_global)
{
    if (llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(c)) {
        if (gv->isConstant() && gv->hasInitializer()) {
            MDL_ASSERT(!gv->isExternallyInitialized());
            c = gv->getInitializer();

            hlsl::Def_variable *var = create_global_const(c);
            return create_reference(var);
        }
    }

    if (llvm::ConstantStruct *cv = llvm::dyn_cast<llvm::ConstantStruct>(c)) {
        return translate_constant_struct_expr(cv, is_global);
    }

    if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(c->getType())) {
        if (llvm::ConstantAggregateZero *cv = llvm::dyn_cast<llvm::ConstantAggregateZero>(c)) {
            size_t num_elems = size_t(cv->getNumElements());
            Small_VLA<hlsl::Expr *, 8> agg_elems(m_alloc, num_elems);

            for (size_t i = 0; i < num_elems; ++i) {
                llvm::Constant *elem = cv->getElementValue(i);
                agg_elems[i] = translate_constant_expr(elem, is_global);
            }

            hlsl::Type *res_type = convert_type(st);
            if (is_global) {
                hlsl::Expr *res = m_expr_factory.create_compound(zero_loc, agg_elems);
                res->set_type(res_type);
                return res;
            }
            return create_constructor_call(res_type, agg_elems, zero_loc);
        }

        if (llvm::UndefValue *undef = llvm::dyn_cast<llvm::UndefValue>(c)) {
            size_t num_elems = size_t(undef->getNumElements());
            Small_VLA<hlsl::Expr *, 8> agg_elems(m_alloc, num_elems);

            for (size_t i = 0; i < num_elems; ++i) {
                llvm::Constant *elem = undef->getElementValue(i);
                agg_elems[i] = translate_constant_expr(elem, is_global);
            }

            hlsl::Type *res_type = convert_type(st);
            if (is_global) {
                hlsl::Expr *res = m_expr_factory.create_compound(zero_loc, agg_elems);
                res->set_type(res_type);
                return res;
            }
            return create_constructor_call(res_type, agg_elems, zero_loc);
        }
    }

    if (llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(c->getType())) {
        if (!is_matrix_type(at)) {
            if (llvm::ConstantDataArray *cv = llvm::dyn_cast<llvm::ConstantDataArray>(c)) {
                return translate_constant_data_array(cv);
            }
            if (llvm::ConstantArray *cv = llvm::dyn_cast<llvm::ConstantArray>(c)) {
                return translate_constant_array(cv, is_global);
            }
            if (llvm::ConstantAggregateZero *cv = llvm::dyn_cast<llvm::ConstantAggregateZero>(c)) {
                return translate_constant_array(cv, is_global);
            }
        }
    }

    return m_expr_factory.create_literal(zero_loc, translate_constant(c));
}

// Translate an LLVM value to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr(llvm::Value *value)
{
    // check whether a local variable was generated for this instruction
    auto it = m_local_var_map.find(value);
    if (it != m_local_var_map.end()) {
        return create_reference(it->second);
    }

    if (llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(value)) {
        if (inst->isBinaryOp() ||
            llvm::isa<llvm::ICmpInst>(inst) ||
            llvm::isa<llvm::FCmpInst>(inst))
        {
            return translate_expr_bin(inst);
        }

        if (inst->isCast()) {
            return translate_expr_cast(llvm::cast<llvm::CastInst>(inst));
        }

        hlsl::Expr *result = nullptr;

        switch (inst->getOpcode()) {
        case llvm::Instruction::Alloca:
            {
                hlsl::Def_variable *var_def = create_local_var(inst);
                hlsl::Symbol       *var_sym = var_def->get_symbol();
                llvm::PointerType  *ptr_type = llvm::cast<llvm::PointerType>(inst->getType());
                llvm::Type         *var_type = ptr_type->getElementType();
                result = create_reference(var_sym, var_type);
            }
            break;

        case llvm::Instruction::Load:
            result = translate_expr_load(llvm::cast<llvm::LoadInst>(inst));
            break;

        case llvm::Instruction::Store:
            result = translate_expr_store(llvm::cast<llvm::StoreInst>(inst));
            break;

        case llvm::Instruction::GetElementPtr:
            {
                llvm::GetElementPtrInst *gep = llvm::cast<llvm::GetElementPtrInst>(inst);
                MDL_ASSERT(gep->hasAllZeroIndices());

                // treat as pointer cast, skip it
                return translate_expr(gep->getOperand(0));
            }


        case llvm::Instruction::PHI:
            {
                llvm::PHINode *phi = llvm::cast<llvm::PHINode>(inst);
                MDL_ASSERT(!"unexpected PHI node, a local variable should have been registered");

                hlsl::Def_variable *phi_out_var = get_phi_out_var(phi);
                return create_reference(phi_out_var);
            }

        case llvm::Instruction::Call:
            return translate_expr_call(llvm::cast<llvm::CallInst>(inst), nullptr);

        case llvm::Instruction::Select:
            result = translate_expr_select(llvm::cast<llvm::SelectInst>(inst));
            break;

        case llvm::Instruction::ShuffleVector:
            result = translate_expr_shufflevector(llvm::cast<llvm::ShuffleVectorInst>(inst));
            break;

        case llvm::Instruction::ExtractElement:
            result = translate_expr_extractelement(llvm::cast<llvm::ExtractElementInst>(inst));
            break;

        case llvm::Instruction::ExtractValue:
            result = translate_expr_extractvalue(llvm::cast<llvm::ExtractValueInst>(inst));
            break;

        case llvm::Instruction::InsertElement:
            result = translate_expr_insertelement(llvm::cast<llvm::InsertElementInst>(inst));
            break;

        case llvm::Instruction::InsertValue:
            result = translate_expr_insertvalue(llvm::cast<llvm::InsertValueInst>(inst));
            break;

        case llvm::Instruction::Fence:
        case llvm::Instruction::AtomicCmpXchg:
        case llvm::Instruction::AtomicRMW:
        case llvm::Instruction::CleanupPad:
        case llvm::Instruction::CatchPad:
        case llvm::Instruction::VAArg:
        case llvm::Instruction::LandingPad:
            MDL_ASSERT(!"unexpected LLVM instruction");
            result = m_expr_factory.create_invalid(convert_location(inst));
            break;

        default:
            break;
        }

        if (result != nullptr) {
            return result;
        }
    }

    if (llvm::Constant *ci = llvm::dyn_cast<llvm::Constant>(value)) {
        // check, if the constant can be expressed as a HLSL constant expression, otherwise
        // create a HLSL constant declaration
        if (llvm::ArrayType *a_type = llvm::dyn_cast<llvm::ArrayType>(ci->getType())) {
            if (!is_matrix_type(a_type)) {
                hlsl::Def_variable *var = create_local_const(ci);
                return create_reference(var);
            }
        }
        return translate_constant_expr(ci, /*is_global=*/ false);
    }
    MDL_ASSERT(!"unexpected LLVM value");
    return m_expr_factory.create_invalid(zero_loc);
}

// If a given type has an unsigned variant, return it.
hlsl::Type *HLSLWriterPass::to_unsigned_type(hlsl::Type *type)
{
    switch (type->get_kind()) {
    case hlsl::Type::TK_ALIAS:
        return to_unsigned_type(type->skip_type_alias());

    case hlsl::Type::TK_VOID:
    case hlsl::Type::TK_BOOL:
    case hlsl::Type::TK_HALF:
    case hlsl::Type::TK_FLOAT:
    case hlsl::Type::TK_DOUBLE:
    case hlsl::Type::TK_MIN10FLOAT:
    case hlsl::Type::TK_MIN16FLOAT:
    case hlsl::Type::TK_ARRAY:
    case hlsl::Type::TK_STRUCT:
    case hlsl::Type::TK_FUNCTION:
    case hlsl::Type::TK_TEXTURE:
    case hlsl::Type::TK_ERROR:
        return nullptr;

    case hlsl::Type::TK_INT:
        return m_type_factory.get_uint();

    case hlsl::Type::TK_UINT:
        return type;

    case hlsl::Type::TK_MIN12INT:
        // no unsigned variant
        return nullptr;

    case hlsl::Type::TK_MIN16INT:
        return m_type_factory.get_min16uint();

    case hlsl::Type::TK_MIN16UINT:
        return type;

    case hlsl::Type::TK_VECTOR:
        {
            hlsl::Type_vector *v_type = hlsl::cast<hlsl::Type_vector>(type);
            hlsl::Type_scalar *e_type = v_type->get_element_type();
            hlsl::Type_scalar *u_type = hlsl::cast<hlsl::Type_scalar>(to_unsigned_type(e_type));

            if (u_type != nullptr) {
                if (u_type != e_type)
                    return m_type_factory.get_vector(u_type, v_type->get_size());
                return type;
            }
        }
        return nullptr;

    case hlsl::Type::TK_MATRIX:
        {
            hlsl::Type_matrix *m_type = hlsl::cast<hlsl::Type_matrix>(type);
            hlsl::Type_vector *e_type = m_type->get_element_type();
            hlsl::Type_vector *u_type = hlsl::cast<hlsl::Type_vector>(to_unsigned_type(e_type));

            if (u_type != nullptr) {
                if (u_type != e_type)
                    return m_type_factory.get_matrix(u_type, m_type->get_columns());
                return type;
            }
        }
        return nullptr;
    }
    MDL_ASSERT("!unexpected type kind");
    return nullptr;
}

// Translate a binary LLVM instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_bin(llvm::Instruction *inst)
{
    hlsl::Expr *left  = translate_expr(inst->getOperand(0));
    hlsl::Expr *right = translate_expr(inst->getOperand(1));

    hlsl::Expr_binary::Operator hlsl_op;

    bool need_unsigned_operands = false;

    switch (inst->getOpcode()) {
    // "Standard binary operators"
    case llvm::Instruction::Add:
    case llvm::Instruction::FAdd:
        hlsl_op = hlsl::Expr_binary::OK_PLUS;
        break;
    case llvm::Instruction::Sub:
    case llvm::Instruction::FSub:
        hlsl_op = hlsl::Expr_binary::OK_MINUS;
        break;
    case llvm::Instruction::Mul:
    case llvm::Instruction::FMul:
        hlsl_op = hlsl::Expr_binary::OK_MULTIPLY;
        break;
    case llvm::Instruction::UDiv:
    case llvm::Instruction::SDiv:
    case llvm::Instruction::FDiv:
        hlsl_op = hlsl::Expr_binary::OK_DIVIDE;
        break;
    case llvm::Instruction::URem:
    case llvm::Instruction::SRem:
    case llvm::Instruction::FRem:
        hlsl_op = hlsl::Expr_binary::OK_MODULO;
        break;

    // "Logical operators (integer operands)"
    case llvm::Instruction::Shl:
        hlsl_op = hlsl::Expr_binary::OK_SHIFT_LEFT;
        break;
    case llvm::Instruction::LShr:  // unsigned
        need_unsigned_operands = true;
        hlsl_op = hlsl::Expr_binary::OK_SHIFT_RIGHT;
        break;
    case llvm::Instruction::AShr:  // signed
        hlsl_op = hlsl::Expr_binary::OK_SHIFT_RIGHT;
        break;
    case llvm::Instruction::And:
        if (hlsl::is<hlsl::Type_bool>(left->get_type()->skip_type_alias()) &&
            hlsl::is<hlsl::Type_bool>(right->get_type()->skip_type_alias()))
        {
            // map bitwise AND on boolean values to logical and
            hlsl_op = hlsl::Expr_binary::OK_LOGICAL_AND;
        } else {
            hlsl_op = hlsl::Expr_binary::OK_BITWISE_AND;
        }
        break;
    case llvm::Instruction::Or:
        if (hlsl::is<hlsl::Type_bool>(left->get_type()->skip_type_alias()) &&
            hlsl::is<hlsl::Type_bool>(right->get_type()->skip_type_alias()))
        {
            // map bitwise OR on boolean values to logical OR
            hlsl_op = hlsl::Expr_binary::OK_LOGICAL_OR;
        } else {
            hlsl_op = hlsl::Expr_binary::OK_BITWISE_OR;
        }
        break;
    case llvm::Instruction::Xor:
        if (hlsl::is<hlsl::Type_bool>(left->get_type()->skip_type_alias()) &&
            hlsl::is<hlsl::Type_bool>(right->get_type()->skip_type_alias()))
        {
            // map XOR on boolean values to NOT-EQUAL to be compatible to SLANG
            hlsl_op = hlsl::Expr_binary::OK_NOT_EQUAL;
        } else
            hlsl_op = hlsl::Expr_binary::OK_BITWISE_XOR;
        break;

    // "Other operators"
    case llvm::Instruction::ICmp:
    case llvm::Instruction::FCmp:
        {
            llvm::CmpInst *cmp = llvm::cast<llvm::CmpInst>(inst);
            switch (cmp->getPredicate()) {
            case llvm::CmpInst::FCMP_OEQ:
            case llvm::CmpInst::FCMP_UEQ:
            case llvm::CmpInst::ICMP_EQ:
                hlsl_op = hlsl::Expr_binary::OK_EQUAL;
                break;
            case llvm::CmpInst::FCMP_ONE:
            case llvm::CmpInst::FCMP_UNE:
            case llvm::CmpInst::ICMP_NE:
                hlsl_op = hlsl::Expr_binary::OK_NOT_EQUAL;
                break;

            case llvm::CmpInst::FCMP_OGT:
            case llvm::CmpInst::FCMP_UGT:
            case llvm::CmpInst::ICMP_SGT:
                hlsl_op = hlsl::Expr_binary::OK_GREATER;
                break;
            case llvm::CmpInst::ICMP_UGT:
                need_unsigned_operands = true;
                hlsl_op = hlsl::Expr_binary::OK_GREATER;
                break;

            case llvm::CmpInst::FCMP_OGE:
            case llvm::CmpInst::FCMP_UGE:
            case llvm::CmpInst::ICMP_SGE:
                hlsl_op = hlsl::Expr_binary::OK_GREATER_OR_EQUAL;
                break;
            case llvm::CmpInst::ICMP_UGE:
                need_unsigned_operands = true;
                hlsl_op = hlsl::Expr_binary::OK_GREATER_OR_EQUAL;
                break;

            case llvm::CmpInst::FCMP_OLT:
            case llvm::CmpInst::FCMP_ULT:
            case llvm::CmpInst::ICMP_SLT:
                hlsl_op = hlsl::Expr_binary::OK_LESS;
                break;
            case llvm::CmpInst::ICMP_ULT:
                need_unsigned_operands = true;
                hlsl_op = hlsl::Expr_binary::OK_LESS;
                break;

            case llvm::CmpInst::FCMP_OLE:
            case llvm::CmpInst::FCMP_ULE:
            case llvm::CmpInst::ICMP_SLE:
                hlsl_op = hlsl::Expr_binary::OK_LESS_OR_EQUAL;
                break;
            case llvm::CmpInst::ICMP_ULE:
                need_unsigned_operands = true;
                hlsl_op = hlsl::Expr_binary::OK_LESS_OR_EQUAL;
                break;

            case llvm::CmpInst::FCMP_ORD:
            case llvm::CmpInst::FCMP_UNO:
                {
                    hlsl::Type *bool_type = m_type_factory.get_bool();
                    hlsl::Expr *isnan_ref = create_reference(get_sym("isnan"), left->get_type());

                    hlsl::Expr *left_isnan = m_expr_factory.create_call(isnan_ref, { left });
                    left_isnan->set_type(bool_type);

                    hlsl::Expr *right_isnan = m_expr_factory.create_call(isnan_ref, { right });
                    right_isnan->set_type(bool_type);

                    hlsl::Expr *res = m_expr_factory.create_binary(
                        Expr_binary::OK_LOGICAL_OR,
                        left_isnan,
                        right_isnan);
                    if (cmp->getPredicate() == llvm::CmpInst::FCMP_ORD) {
                        res = m_expr_factory.create_unary(
                            convert_location(inst), Expr_unary::OK_LOGICAL_NOT, res);
                    }
                    res->set_type(bool_type);
                    return res;
                }

            default:
                MDL_ASSERT(!"unexpected LLVM comparison predicate");
                return m_expr_factory.create_invalid(convert_location(inst));
            }
            break;
        }

    default:
        MDL_ASSERT(!"unexpected LLVM binary instruction");
        return m_expr_factory.create_invalid(convert_location(inst));
    }

    // LLVM does not have an unary minus, but HLSL has
    bool convert_to_unary_minus = false;
    if (hlsl_op == hlsl::Expr_binary::OK_MINUS) {
        if (hlsl::Expr_literal *l = as<hlsl::Expr_literal>(left)) {
            hlsl::Value *lv = l->get_value();

            if (lv->is_zero()) {
                if (hlsl::Value_fp *f = as<hlsl::Value_fp>(lv)) {
                    if (f->is_minus_zero()) {
                        // -0.0 - x == -x
                        convert_to_unary_minus = true;
                    }
                } else {
                    // 0 - x = -x
                    convert_to_unary_minus = true;
                }
            }
        }
    }

    hlsl::Expr *res = nullptr;
    if (convert_to_unary_minus) {
        res = m_expr_factory.create_unary(
            convert_location(inst), hlsl::Expr_unary::OK_NEGATIVE, right);
    } else {
        if (need_unsigned_operands) {
            hlsl::Type *l_tp   = left->get_type()->skip_type_alias();
            hlsl::Type *u_l_tp = to_unsigned_type(l_tp);

            if (u_l_tp != nullptr && u_l_tp != l_tp) {
                left = create_cast(u_l_tp, left);
            }

            hlsl::Type *r_tp = right->get_type()->skip_type_alias();
            hlsl::Type *u_r_tp = to_unsigned_type(r_tp);

            if (u_r_tp != nullptr && u_r_tp != r_tp) {
                right = create_cast(u_r_tp, right);
            }
        }
        res = m_expr_factory.create_binary(hlsl_op, left, right);
    }
    res->set_type(convert_type(inst->getType()));

    return res;
}

// Find a function parameter of the given type in the current function.
hlsl::Definition *HLSLWriterPass::find_parameter_of_type(llvm::Type *t)
{
    for (llvm::Argument &arg_it : m_curr_func->args()) {
        llvm::Type *arg_llvm_type = arg_it.getType();

        if (arg_llvm_type == t) {
            return m_local_var_map[&arg_it];
        }
    }
    return nullptr;
}

// Translate an LLVM select instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_select(llvm::SelectInst *select)
{
    hlsl::Expr *cond       = translate_expr(select->getCondition());
    hlsl::Expr *true_expr  = translate_expr(select->getTrueValue());
    hlsl::Expr *false_expr = translate_expr(select->getFalseValue());
    hlsl::Expr *res        = m_expr_factory.create_conditional(cond, true_expr, false_expr);

    res->set_location(convert_location(select));
    return res;
}

// Translate an LLVM cast instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_call(
    llvm::CallInst   *call,
    hlsl::Definition *dst_var)
{
    llvm::Function *func = call->getCalledFunction();
    if (func == NULL) {
        MDL_ASSERT(func && "indirection function calls not supported");
        return m_expr_factory.create_invalid(convert_location(call));
    }

    hlsl::Symbol *func_sym      = nullptr;
    hlsl::Type   *ret_type      = nullptr;
    bool         has_out_return = false;

    if (hlsl::Def_function *func_def = get_definition(func)) {
        DG_node *call_node = m_dg.get_node(func_def);

        if (m_curr_node != nullptr)
            m_dg.add_edge(m_curr_node, call_node);

        hlsl::Type_function *func_type = func_def->get_type();

        func_sym = func_def->get_symbol();
        ret_type = func_type->get_return_type();

        // handle converted out parameter
        if (hlsl::is<hlsl::Type_void>(func_type->get_return_type()) &&
            !call->getType()->isVoidTy())
        {
            hlsl::Type_function::Parameter *param = func_type->get_parameter(0);
            if (param->get_modifier() == hlsl::Type_function::Parameter::PM_OUT)
                has_out_return = true;
        }
    } else {
        // call to an unknown entity
        llvm::StringRef name = func->getName();

        bool is_llvm_intrinsic = name.startswith("llvm.");
        if (is_llvm_intrinsic || name.startswith("hlsl.")) {
            // handle HLSL or LLVM intrinsics
            name = name.drop_front(5);

            if (is_llvm_intrinsic) {
                size_t pos = name.find('.');
                name = name.slice(0, pos);

                // need some mapping between LLVM intrinsics and HLSL
                if (name == "fabs")
                    name = "abs";
            }
        }
        func_sym = get_sym(name);
        ret_type = convert_type(func->getReturnType());
    }

    hlsl::Expr *func_ref = create_reference(func_sym, ret_type);

    // count all zero-sized array arguments
    unsigned num_void_args = 0;
    for (llvm::Value *arg : call->arg_operands()) {
        llvm::Type *arg_llvm_type = arg->getType();
        if (llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(arg_llvm_type))
            arg_llvm_type = pt->getPointerElementType();

        hlsl::Type *arg_type = convert_type(arg_llvm_type);
        if (hlsl::is<hlsl::Type_void>(arg_type))
            ++num_void_args;
    }

    size_t n_args = call->getNumArgOperands() - num_void_args;
    if (has_out_return)
        ++n_args;

    Small_VLA<hlsl::Expr *, 8> args(m_alloc, n_args);

    size_t i = 0;

    hlsl::Def_variable *result_tmp = nullptr;
    if (has_out_return) {
        if (dst_var != nullptr) {
            args[i++] = create_reference(dst_var);
            dst_var = nullptr;
        } else {
            result_tmp = create_local_var(get_unique_hlsl_sym("tmp", "tmp"), call->getType());
            args[i++] = create_reference(result_tmp);
        }
    }

    for (llvm::Value *arg : call->arg_operands()) {
        llvm::Type *arg_llvm_type = arg->getType();
        if (llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(arg_llvm_type))
            arg_llvm_type = pt->getPointerElementType();

        hlsl::Type *arg_type = convert_type(arg_llvm_type);
        if (hlsl::is<hlsl::Type_void>(arg_type))
            continue;  // skip void typed parameters

        hlsl::Expr *arg_expr = nullptr;
        if (llvm::isa<llvm::UndefValue>(arg)) {
            // the called function does not use this argument
            llvm::Type *t = arg->getType();

            // we call a function with an argument, try to find
            // a matching parameter and replace it
            // In our case, this typically "reverts" the optimization
            if (hlsl::Definition *param_def = find_parameter_of_type(t)) {
                arg_expr = create_reference(param_def);
            }
        }

        if (arg_expr == nullptr) {
            arg_expr = translate_expr(arg);
        }
        args[i++] = arg_expr;
    }

    hlsl::Expr *expr_call = m_expr_factory.create_call(func_ref, args);
    expr_call->set_type(ret_type);
    expr_call->set_location(convert_location(call));

    if (result_tmp != nullptr) {
        hlsl::Expr *t = create_reference(result_tmp);
        hlsl::Expr *r = m_expr_factory.create_binary(
            hlsl::Expr_binary::OK_SEQUENCE,
            expr_call,
            t);
        r->set_type(t->get_type());
        r->set_location(expr_call->get_location());
        expr_call = r;
    }

    if (dst_var != nullptr)
        return create_assign_expr(dst_var, expr_call);
    return expr_call;
}

// Translate an LLVM cast instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_cast(llvm::CastInst *inst)
{
    hlsl::Expr *expr = translate_expr(inst->getOperand(0));

    llvm::Type *src_type = inst->getSrcTy();
    llvm::Type *dest_type = inst->getDestTy();

    switch (inst->getOpcode()) {
    case llvm::Instruction::ZExt:
        {
            if (inst->isIntegerCast()) {
                unsigned src_bits  = src_type->getIntegerBitWidth();
                if (src_bits == 1) {
                    // i1 -> i*
                    // Note: HLSL can implicitly convert from bool to integer, but it doesn't
                    //    work when resolving overloads (for example for asfloat())
                    hlsl::Type *hlsl_type = convert_type(inst->getType());
                    return create_cast(hlsl_type, expr);
                }

                unsigned dest_bits = dest_type->getIntegerBitWidth();
                if (src_bits == 32 && dest_bits == 64) {
                    // FIXME: i32 -> i64: ignore is not true in general
                    return expr;
                }
            }
            MDL_ASSERT(!"unsupported LLVM ZExt cast instruction");
            return expr;
        }

    case llvm::Instruction::SExt:
        {
            if (inst->isIntegerCast()) {
                unsigned src_bits  = src_type->getIntegerBitWidth();
                unsigned dest_bits = dest_type->getIntegerBitWidth();
                if (src_bits == 1 && dest_bits == 32) {
                    // bool "sign" cast, i.e. b ? T(-1) : T(0)
                    hlsl::Location loc = convert_location(inst);

                    hlsl::Type  *hlsl_type = convert_type(inst->getType());
                    hlsl::Value *m1        = m_value_factory.get_int32(-1);
                    hlsl::Value *z         = m_value_factory.get_int32(0);
                    hlsl::Expr  *t         = m_expr_factory.create_literal(loc, m1);
                    hlsl::Expr  *f         = m_expr_factory.create_literal(loc, z);
                    hlsl::Expr  *res       = m_expr_factory.create_conditional(expr, t, f);
                    res->set_type(hlsl_type);
                    return res;
                }

                if (src_bits == 32 && dest_bits == 64) {
                    // FIXME: i32 -> i64: ignore is not true in general
                    return expr;
                }
            }
            MDL_ASSERT(!"unsupported LLVM SExt cast instruction");
            return m_expr_factory.create_invalid(convert_location(inst));
        }

    case llvm::Instruction::Trunc:
        {
            hlsl::Type *hlsl_type = convert_type(inst->getType());
            if (expr->get_type() == hlsl_type)
                return expr;

            hlsl::Expr *casted_expr = create_cast(hlsl_type, expr);
            if (llvm::IntegerType *dest_int_type = llvm::dyn_cast<llvm::IntegerType>(dest_type)) {
                int trunc_mask = (1 << dest_int_type->getBitWidth()) - 1;
                hlsl::Expr *trunk_mask_expr = m_expr_factory.create_literal(
                    zero_loc, m_value_factory.get_int32(trunc_mask));
                return m_expr_factory.create_binary(
                    Expr_binary::OK_BITWISE_AND, casted_expr, trunk_mask_expr);
            }
            MDL_ASSERT(!"Probably unsupported trunc instruction");
            return casted_expr;
        }

    case llvm::Instruction::FPToUI:
    case llvm::Instruction::FPToSI:
    case llvm::Instruction::UIToFP:
    case llvm::Instruction::SIToFP:
    case llvm::Instruction::FPTrunc:
    case llvm::Instruction::FPExt:
        {
            hlsl::Type *hlsl_type = convert_type(inst->getType());
            return create_cast(hlsl_type, expr);
        }

    case llvm::Instruction::BitCast:
        {
            hlsl::Type *hlsl_type = convert_type(inst->getType());
            if (hlsl::Expr *res = convert_to(expr, hlsl_type))
                return res;

            if (src_type->isVectorTy() && dest_type->isVectorTy() &&
                    src_type->getVectorNumElements() == dest_type->getVectorNumElements()) {
                llvm::Type *src_vet = src_type->getVectorElementType();
                llvm::Type *dest_vet = dest_type->getVectorElementType();

                if (src_vet->isFloatingPointTy() && dest_vet->isIntegerTy()) {
                    hlsl::Type_scalar *int_type = m_type_factory.get_int();
                    hlsl::Type *vt = m_type_factory.get_vector(
                        int_type, src_type->getVectorNumElements());
                    hlsl::Expr *func_ref = create_reference(get_sym("asint"), vt);
                    hlsl::Expr *expr_call = m_expr_factory.create_call(func_ref, { expr });
                    expr_call->set_type(vt);
                    return expr_call;
                }

                if (src_vet->isIntegerTy() && dest_vet->isFloatingPointTy()) {
                    hlsl::Type_scalar *float_type = m_type_factory.get_float();
                    hlsl::Type *vt = m_type_factory.get_vector(
                        float_type, src_type->getVectorNumElements());

                    hlsl::Expr *func_ref = create_reference(get_sym("asfloat"), vt);
                    hlsl::Expr *expr_call = m_expr_factory.create_call(func_ref, { expr });
                    expr_call->set_type(vt);
                    return expr_call;
                }
            }

            if (src_type->isPointerTy() && dest_type->isPointerTy()) {
                llvm::StructType *src_elem_st =
                    llvm::dyn_cast<llvm::StructType>(src_type->getPointerElementType());
                llvm::StructType *dest_elem_st =
                    llvm::dyn_cast<llvm::StructType>(dest_type->getPointerElementType());

                // skip "bitcast %State_core* %1 to %class.State*" and
                // "bitcast %class.State* %1 to %State_core*"
                // (can appear when optimization is disabled)
                if (src_elem_st && dest_elem_st &&
                        src_elem_st->hasName() && dest_elem_st->hasName() && (
                            (
                                src_elem_st->getName() == "State_core" &&
                                dest_elem_st->getName() == "class.State"
                            ) || (
                                src_elem_st->getName() == "class.State" &&
                                dest_elem_st->getName() == "State_core"
                            )
                        ) ) {
                    return expr;
                }
            }

            MDL_ASSERT(!"unexpected LLVM BitCast instruction");
            return m_expr_factory.create_invalid(convert_location(inst));
        }

    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::AddrSpaceCast:
    default:
        MDL_ASSERT(!"unexpected LLVM cast instruction");
        return m_expr_factory.create_invalid(convert_location(inst));
    }
}

// Creates a HLSL cast expression to the given destination type.
hlsl::Expr *HLSLWriterPass::create_cast(hlsl::Type *dst_type, hlsl::Expr *expr)
{
    // only use C-style cast for non-scalar types to simplify string manipulations
    // on generated code
    hlsl::Expr *res;
    hlsl::Expr *dest_type_ref = create_reference(
        get_type_name(dst_type), dst_type);
    if (!hlsl::is<hlsl::Type_scalar>(dst_type)) {
        res = m_expr_factory.create_typecast(dest_type_ref, expr);
    } else {
        res = m_expr_factory.create_call(dest_type_ref, { expr });
    }
    res->set_type(dst_type);
    return res;
}

// Translate an LLVM load instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_load(llvm::LoadInst *inst)
{
    llvm::Value *pointer = inst->getPointerOperand();

    int64_t target_size = 0;
    while (llvm::BitCastInst *bitcast = llvm::dyn_cast<llvm::BitCastInst>(pointer)) {
        // skip bitcasts but remember first target size
        if (target_size == 0) {
            target_size = m_cur_data_layout->getTypeStoreSize(
                bitcast->getDestTy()->getPointerElementType());
        }
        pointer = bitcast->getOperand(0);
    }

    if (target_size == 0) {
        target_size = m_cur_data_layout->getTypeStoreSize(
            pointer->getType()->getPointerElementType());
    }

    Type_walk_stack stack;
    llvm::Value *base_pointer = process_pointer_address(stack, pointer, target_size);

    hlsl::Type *res_type = convert_type(inst->getType());
    hlsl::Type *res_elem_type = nullptr;
    hlsl::Type *conv_to_type = res_type;
    if (hlsl::Type_vector *vt = hlsl::as<hlsl::Type_vector>(res_type)) {
        res_elem_type = vt->get_element_type();
        conv_to_type = res_elem_type;
    }

    std::vector<hlsl::Expr *> expr_parts;
    uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
    while (target_size > 0) {
        hlsl::Expr *cur_expr = create_compound_elem_expr(stack, base_pointer);
        if (cur_expr->get_type() != conv_to_type) {
            if (hlsl::Expr *converted_expr = convert_to(cur_expr, conv_to_type)) {
                cur_expr = converted_expr;
            }
        }
        expr_parts.push_back(cur_expr);

        Type_walk_element &cur_elem = stack.back();

        // reduce target_size by alignment and alloc size (includes internal padding)
        uint64_t total_size = cur_elem.get_total_size_and_update_offset(
            m_cur_data_layout, cur_offs);
        target_size -= total_size;

        if (total_size == 0) {
            MDL_ASSERT(!"invalid type size");
            break;
        }

        // need to split the part into multiple ones to fit into the result vector?
        hlsl::Type *part_type = expr_parts.back()->get_type();
        if (res_elem_type && part_type != res_elem_type) {
            if (hlsl::Type_vector *part_vt = hlsl::as<hlsl::Type_vector>(part_type)) {
                expr_parts.back() = create_vector_access(expr_parts.back(), 0);
                for (size_t i = 1, n = part_vt->get_size(); i < n; ++i) {
                    hlsl::Expr *sub_part = create_compound_elem_expr(stack, base_pointer);
                    expr_parts.push_back(create_vector_access(sub_part, unsigned(i)));
                }
            } else {
                MDL_ASSERT(!"unexpected part type");
            }
        }

        // done?
        if (target_size <= 0)
            break;

        // go to next element
        if (!move_to_next_compound_elem(stack))
            return m_expr_factory.create_invalid(zero_loc);
    }

    // atomic element read, just return it
    if (expr_parts.size() == 1) {
        return expr_parts[0];
    }

    // we need to construct the result from multiple parts
    MDL_ASSERT(expr_parts.size() > 0);
    return create_constructor_call(res_type, expr_parts, convert_location(inst));
}

// Translate an LLVM store instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_store(llvm::StoreInst *inst)
{
    llvm::Value *pointer = inst->getPointerOperand();
    hlsl::Expr *lvalue   = translate_lval_expression(pointer);

    llvm::Value *value   = inst->getValueOperand();
    hlsl::Expr  *expr    = translate_expr(value);

    return m_expr_factory.create_binary(hlsl::Expr_binary::OK_ASSIGN, lvalue, expr);
}

// Translate an LLVM pointer value to an HLSL lvalue expression.
hlsl::Expr *HLSLWriterPass::translate_lval_expression(llvm::Value *pointer)
{
    uint64_t target_size = 0;
    while (llvm::BitCastInst *bitcast = llvm::dyn_cast<llvm::BitCastInst>(pointer)) {
        // skip bitcasts but remember first target size
        if (target_size == 0) {
            target_size = m_cur_data_layout->getTypeStoreSize(
                bitcast->getDestTy()->getPointerElementType());
        }
        pointer = bitcast->getOperand(0);
    }

    if (llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(pointer)) {
        llvm::ConstantInt *zero_idx = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(1));
        if (zero_idx == nullptr || !zero_idx->isZero()) {
            MDL_ASSERT(!"invalid gep, first index not zero");
            return m_expr_factory.create_invalid(convert_location(gep));
        }

        // walk the base expression using select operations according to the indices of the gep
        llvm::Type *cur_llvm_type = gep->getPointerOperandType()->getPointerElementType();
        hlsl::Expr *cur_expr = translate_lval_expression(gep->getPointerOperand());
        hlsl::Type *cur_type = cur_expr->get_type()->skip_type_alias();

        for (unsigned i = 1, num_indices = gep->getNumIndices(); i < num_indices; ++i) {
            // check whether a bitcast wants us to stop going deeper into the compound
            if (target_size != 0) {
                uint64_t cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);
                if (cur_type_size <= target_size)
                    break;
            }

            if (hlsl::is<hlsl::Type_array>(cur_type) ||
                hlsl::is<hlsl::Type_vector>(cur_type) ||
                hlsl::is<hlsl::Type_matrix>(cur_type))
            {
                llvm::Value *gep_index = gep->getOperand(i + 1);
                if (hlsl::is<hlsl::Type_vector>(cur_type) &&
                        llvm::isa<llvm::ConstantInt>(gep_index)) {
                    llvm::ConstantInt *idx = llvm::cast<llvm::ConstantInt>(gep_index);
                    cur_expr = create_vector_access(cur_expr, unsigned(idx->getZExtValue()));
                } else {
                    hlsl::Expr *array_index = translate_expr(gep_index);
                    cur_expr = m_expr_factory.create_binary(
                        Expr_binary::OK_ARRAY_SUBSCRIPT, cur_expr, array_index);
                }

                cur_type = hlsl::cast<hlsl::Type_compound>(cur_type)->get_compound_type(0);
                cur_type = cur_type->skip_type_alias();
                cur_expr->set_type(cur_type);
                cur_llvm_type =
                    llvm::cast<llvm::CompositeType>(cur_llvm_type)->getTypeAtIndex(0u);

                continue;
            }

            if (hlsl::Type_struct *struct_type = hlsl::as<hlsl::Type_struct>(cur_type)) {
                llvm::ConstantInt *idx = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(i + 1));
                if (idx == nullptr) {
                    MDL_ASSERT(!"invalid field index for a struct type");
                    return m_expr_factory.create_invalid(convert_location(gep));
                }
                unsigned idx_val = unsigned(idx->getZExtValue());
                hlsl::Type_struct::Field *field = struct_type->get_field(idx_val);
                hlsl::Expr *field_ref = create_reference(field->get_symbol(), field->get_type());
                cur_expr = m_expr_factory.create_binary(
                    Expr_binary::OK_SELECT, cur_expr, field_ref);

                cur_type = field->get_type()->skip_type_alias();
                cur_expr->set_type(cur_type);
                cur_llvm_type =
                    llvm::cast<llvm::CompositeType>(cur_llvm_type)->getTypeAtIndex(idx_val);
                continue;
            }

            MDL_ASSERT(!"Unexpected element type for GEP");
        }
        return cur_expr;
    }

    // should be alloca or pointer parameter
    auto it = m_local_var_map.find(pointer);
    if (it != m_local_var_map.end()) {
        return create_reference(it->second);
    }
    MDL_ASSERT(!"unexpected unmapped alloca or pointer parameter");
    return m_expr_factory.create_invalid(zero_loc);
}

// Returns the given expression as a call expression, if it is a call to a vector constructor.
static hlsl::Expr_call *as_vector_constructor_call(hlsl::Expr *expr)
{
    hlsl::Expr_call *call = hlsl::as<hlsl::Expr_call>(expr);
    if (call == nullptr)
        return nullptr;
    hlsl::Expr_ref *callee_ref = hlsl::as<hlsl::Expr_ref>(call->get_callee());
    if (callee_ref == nullptr)
        return nullptr;

    hlsl::Type *res_type = call->get_type();
    if (res_type == nullptr || !hlsl::is<hlsl::Type_vector>(res_type->skip_type_alias()))
        return nullptr;

    hlsl::Definition *def = callee_ref->get_definition();
    if (def == nullptr)
        return nullptr;

    hlsl::Def_function *call_def = hlsl::as<hlsl::Def_function>(def);
    if (call_def == nullptr)
        return nullptr;

    if (call_def->get_semantics() != hlsl::Def_function::DS_ELEM_CONSTRUCTOR)
        return nullptr;

    return call;
}

// Translate an LLVM shufflevector instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_shufflevector(llvm::ShuffleVectorInst *inst)
{
    hlsl::Expr *v1_expr = translate_expr(inst->getOperand(0));

    // is this shuffle a swizzle?
    if (llvm::isa<llvm::UndefValue>(inst->getOperand(1))) {
        if (hlsl::Expr_call *constr_call = as_vector_constructor_call(v1_expr)) {
            hlsl::Type_vector *call_vt = hlsl::cast<hlsl::Type_vector>(constr_call->get_type());
            size_t len = inst->getShuffleMask().size();
            hlsl::Type_vector *res_type = m_type_factory.get_vector(
                call_vt->get_element_type(), len);

            Small_VLA<hlsl::Expr *, 4> vec_elems(m_alloc, len);
            unsigned cur_elem_idx = 0;
            for (int index : inst->getShuffleMask()) {
                vec_elems[cur_elem_idx] = constr_call->get_argument(index);
                ++cur_elem_idx;
            }

            return create_constructor_call(res_type, vec_elems, convert_location(inst));
        }

        string shuffle_name(m_alloc);
        int first_valid_index = -1;
        for (int index : inst->getShuffleMask()) {
            // handle undef indices
            if (index < 0) {
                // first time we see an undef?
                if (first_valid_index < 0) {
                    // default to index 0
                    first_valid_index = 0;

                    // find first valid index, to reuse already used values if possible
                    for (int i : inst->getShuffleMask()) {
                        if (i >= 0) {
                            first_valid_index = i;
                            break;
                        }
                    }
                }

                index = first_valid_index;
            }

            if (char const *index_name = get_vector_index_str(uint64_t(index))) {
                shuffle_name.append(index_name);
            } else {
                MDL_ASSERT(!"invalid shuffle mask");
                return m_expr_factory.create_invalid(convert_location(inst));
            }
        }

        hlsl::Type *res_type  = convert_type(inst->getType());
        hlsl::Expr *index_ref = create_reference(get_sym(shuffle_name.c_str()), res_type);
        hlsl::Expr *res       = m_expr_factory.create_binary(
            Expr_binary::OK_SELECT, v1_expr, index_ref);
        res->set_type(res_type);
        res->set_location(convert_location(inst));

        return res;
    }

    // no, use constructor for translation
    // collect elements from both LLVM values via the shuffle matrix
    hlsl::Expr *v2_expr = translate_expr(inst->getOperand(1));

    uint64_t num_elems = inst->getType()->getNumElements();
    int v1_size = inst->getOperand(0)->getType()->getVectorNumElements();
    Small_VLA<hlsl::Expr *, 4> vec_elems(m_alloc, num_elems);
    unsigned cur_elem_idx = 0;
    int first_valid_index = -1;
    for (int index : inst->getShuffleMask()) {
        // handle undef indices
        if (index < 0) {
            // first time we see an undef?
            if (first_valid_index < 0) {
                // default to index 0
                first_valid_index = 0;

                // find first valid index, to reuse already used values if possible
                for (int i : inst->getShuffleMask()) {
                    if (i >= 0) {
                        first_valid_index = i;
                        break;
                    }
                }
            }

            index = first_valid_index;
        }

        if (index >= v1_size)
            vec_elems[cur_elem_idx] = create_vector_access(v2_expr, unsigned(index - v1_size));
        else
            vec_elems[cur_elem_idx] = create_vector_access(v1_expr, unsigned(index));
        ++cur_elem_idx;
    }

    return create_constructor_call(
        convert_type(inst->getType()), vec_elems, convert_location(inst));
}

// Translate an HLSL vector element access to an HLSL expression.
hlsl::Expr *HLSLWriterPass::create_vector_access(hlsl::Expr *vec, unsigned index)
{
    hlsl::Type_vector *vt = hlsl::as<hlsl::Type_vector>(vec->get_type());
    if (vt == nullptr) {
        MDL_ASSERT(!"create_vector_access on non-vector expression");
        return m_expr_factory.create_invalid(zero_loc);
    }

    if (hlsl::Expr_literal *literal_expr = as<hlsl::Expr_literal>(vec)) {
        if (hlsl::Value_vector *val = as<hlsl::Value_vector>(literal_expr->get_value())) {
            hlsl::Value *extracted_value = val->extract(m_value_factory, index);
            return m_expr_factory.create_literal(zero_loc, extracted_value);
        }
    }

    hlsl::Expr *res;

    if (hlsl::Symbol *index_sym = get_vector_index_sym(index)) {
        hlsl::Expr *index_ref = m_expr_factory.create_reference(get_type_name(index_sym));

        index_ref->set_type(vt->get_element_type());

        res = m_expr_factory.create_binary(Expr_binary::OK_SELECT, vec, index_ref);
    } else {
        hlsl::Expr *index_expr = m_expr_factory.create_literal(
            zero_loc,
            m_value_factory.get_int32(index));
        res = m_expr_factory.create_binary(Expr_binary::OK_ARRAY_SUBSCRIPT, vec, index_expr);
    }

    res->set_type(vt->get_element_type());
    return res;
}

// Translate an LLVM InsertElement instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_insertelement(llvm::InsertElementInst *inst)
{
    // collect all values for the vector by following the InsertElement instruction chain
    uint64_t num_elems = inst->getType()->getNumElements();
    uint64_t remaining_elems = num_elems;
    Small_VLA<hlsl::Expr *, 4> vec_elems(m_alloc, num_elems);
    memset(vec_elems.data(), 0, vec_elems.size() * sizeof(hlsl::Expr *));

    llvm::InsertElementInst *cur_insert = inst;
    llvm::Value *cur_value = cur_insert;
    while (!has_local_var(cur_value)) {
        llvm::Value *index = cur_insert->getOperand(2);
        if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
            uint64_t index_val = ci->getZExtValue();
            if (index_val >= num_elems) {
                MDL_ASSERT(!"invalid InsertElement instruction");
                return m_expr_factory.create_invalid(zero_loc);
            }
            if (vec_elems[index_val] == nullptr) {
                vec_elems[index_val] = translate_expr(cur_insert->getOperand(1));

                // all vector element initializers found?
                if (--remaining_elems == 0)
                    break;
            }
        } else {
            MDL_ASSERT(!"InsertElement with non-constant index not supported");
            return m_expr_factory.create_invalid(zero_loc);
        }

        cur_value = cur_insert->getOperand(0);
        cur_insert = llvm::dyn_cast<llvm::InsertElementInst>(cur_value);
        if (cur_insert == nullptr)
            break;
    }

    // not all elements found? -> extract them from the current remaining value
    if (remaining_elems) {
        if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(cur_value)) {
            for (uint64_t i = 0; i < num_elems; ++i) {
                if (vec_elems[i] != nullptr)
                    continue;
                vec_elems[i] = translate_expr(cv->getOperand(unsigned(i)));
            }
        } else {
            hlsl::Expr *base_expr = translate_expr(cur_value);
            for (uint64_t i = 0; i < num_elems; ++i) {
                if (vec_elems[i] != nullptr)
                    continue;

                vec_elems[i] = create_vector_access(base_expr, unsigned(i));
            }
        }
    }

    return create_constructor_call(
        convert_type(inst->getType()), vec_elems, convert_location(inst));
}

/// Helper class to process insertvalue instructions of nested types.
class InsertValueObject
{
public:
    /// Constructor.
    InsertValueObject(Arena_builder *arena_builder, llvm::Type *llvm_type)
    : m_expr(nullptr)
    , m_children(*arena_builder->get_arena(), get_num_children_for_type(llvm_type))
    {
        if (llvm::CompositeType *ct = llvm::dyn_cast<llvm::CompositeType>(llvm_type)) {
            // Add children if get_num_children_for_type() said, we need them
            for (size_t i = 0, n = m_children.size(); i < n; ++i) {
                llvm::Type *child_type = ct->getTypeAtIndex(unsigned(i));
                m_children[i] = arena_builder->create<InsertValueObject>(arena_builder, child_type);
            }
        }
    }

    /// Get the child at the given index.
    ///
    /// \returns nullptr, if the index is invalid
    InsertValueObject *get_child(unsigned index)
    {
        MDL_ASSERT(index < m_children.size());
        if (index >= m_children.size())
            return nullptr;
        return m_children[index];
    }

    /// Returns true, if the object is already fully set by later InsertValue instructions.
    bool has_expr()
    {
        return m_expr != nullptr;
    }

    /// Set the expression for this object.
    void set_expr(hlsl::Expr *expr)
    {
        m_expr = expr;
    }

    /// Translate the object into an expression.
    ///
    /// \param alloc      an allocator used for temporary arrays
    /// \param writer     the HLSLWriterPass to create AST
    /// \param base_expr  an expression to use, when the expression for this object has not been set
    hlsl::Expr *translate(IAllocator *alloc, HLSLWriterPass *writer, hlsl::Expr *base_expr)
    {
        if (m_expr)
            return m_expr;

        if (m_children.size() == 0) {
            return base_expr;
        }

        Small_VLA<hlsl::Expr *, 4> agg_elems(alloc, m_children.size());
        for (size_t i = 0, n = m_children.size(); i < n; ++i) {
            hlsl::Expr *child_base_expr = writer->create_compound_access(base_expr, unsigned(i));

            agg_elems[i] = m_children[i]->translate(alloc, writer, child_base_expr);
        }

        return writer->create_constructor_call(base_expr->get_type(), agg_elems, zero_loc);
    }

private:
    /// Returns the number of children objects which should be created for the given type.
    static size_t get_num_children_for_type(llvm::Type *llvm_type)
    {
        if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(llvm_type)) {
            return st->getNumElements();
        } else if (llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(llvm_type)) {
            return at->getNumElements();
        } else
            return 0;
    }

private:
    hlsl::Expr *m_expr;
    Arena_VLA<InsertValueObject *> m_children;
};

// Translate an LLVM InsertValue instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_insertvalue(llvm::InsertValueInst *inst)
{
    // collect all values for the struct or array by following the InsertValue instruction chain
    llvm::Type *comp_type = inst->getType();

    Memory_arena arena(m_alloc);
    Arena_builder builder(arena);
    InsertValueObject root(&builder, comp_type);

    llvm::InsertValueInst *cur_insert = inst;
    llvm::Value *cur_value = cur_insert;
    while (!has_local_var(cur_value)) {
        InsertValueObject *cur_obj = &root;
        for (unsigned i = 0, n = cur_insert->getNumIndices(); i < n && !cur_obj->has_expr(); ++i) {
            unsigned cur_index = cur_insert->getIndices()[i];
            cur_obj = cur_obj->get_child(cur_index);
        }

        // only overwrite the value, if it has not been set, yet (here or in a parent level)
        if (!cur_obj->has_expr()) {
            hlsl::Expr *cur_expr = translate_expr(cur_insert->getInsertedValueOperand());
            cur_obj->set_expr(cur_expr);
        }

        cur_value = cur_insert->getAggregateOperand();
        cur_insert = llvm::dyn_cast<llvm::InsertValueInst>(cur_value);
        if (cur_insert == nullptr)
            break;
    }

    // translate collected values into an expression
    hlsl::Expr *base_expr = translate_expr(cur_value);
    hlsl::Expr *res = root.translate(m_alloc, this, base_expr);
    return res;
}

// Translate an LLVM ExtractElement instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_extractelement(llvm::ExtractElementInst *extract)
{
    hlsl::Expr  *expr  = translate_expr(extract->getVectorOperand());
    llvm::Value *index = extract->getIndexOperand();

    hlsl::Type *res_type = convert_type(extract->getType());
    hlsl::Expr *res;

    if (hlsl::Symbol *index_sym = get_vector_index_sym(index)) {
        hlsl::Expr *index_ref = create_reference(index_sym, res_type);
        res = m_expr_factory.create_binary(
            Expr_binary::OK_SELECT, expr, index_ref);
    } else {
        hlsl::Expr *index_expr = translate_expr(index);
        res = m_expr_factory.create_binary(
            Expr_binary::OK_ARRAY_SUBSCRIPT, expr, index_expr);
    }

    res->set_type(res_type);
    res->set_location(convert_location(extract));

    return res;
}

// Translate an LLVM ExtractValue instruction to an HLSL expression.
hlsl::Expr *HLSLWriterPass::translate_expr_extractvalue(llvm::ExtractValueInst *extract)
{
    hlsl::Expr *res = translate_expr(extract->getAggregateOperand());
    hlsl::Type *cur_type = res->get_type()->skip_type_alias();

    for (unsigned i : extract->getIndices()) {
        if (hlsl::is<hlsl::Type_array>(cur_type) ||
            hlsl::is<hlsl::Type_matrix>(cur_type))
        {
            hlsl::Expr *index_expr = m_expr_factory.create_literal(
                zero_loc,
                m_value_factory.get_int32(i));
            res = m_expr_factory.create_binary(
                Expr_binary::OK_ARRAY_SUBSCRIPT, res, index_expr);
        } else if (hlsl::is<hlsl::Type_vector>(cur_type)) {
            // due to type mapping, this could also be a vector
            res = create_vector_access(res, i);
        } else {
            hlsl::Type_struct *s_type = hlsl::cast<hlsl::Type_struct>(cur_type);

            if (hlsl::Type_struct::Field *field = s_type->get_field(i)) {
                hlsl::Expr *index_ref = create_reference(field->get_symbol(), field->get_type());
                res = m_expr_factory.create_binary(
                    Expr_binary::OK_SELECT, res, index_ref);
            } else {
                MDL_ASSERT(!"ExtractValue index too high");
                res = m_expr_factory.create_invalid(zero_loc);
            }
        }

        cur_type = get_compound_sub_type(hlsl::cast<hlsl::Type_compound>(cur_type), i);
        cur_type = cur_type->skip_type_alias();
        res->set_type(cur_type);
    }

    res->set_location(convert_location(extract));

    return res;
}

// Get the type of the i-th subelement.
hlsl::Type *HLSLWriterPass::get_compound_sub_type(hlsl::Type_compound *comp_type, unsigned i)
{
    if (hlsl::Type_vector *tp = hlsl::as<hlsl::Type_vector>(comp_type)) {
        return tp->get_element_type();
    } else if (hlsl::Type_matrix *tp = hlsl::as<hlsl::Type_matrix>(comp_type)) {
        return tp->get_element_type();
    } else if (hlsl::Type_array *tp = hlsl::as<hlsl::Type_array>(comp_type)) {
        return tp->get_element_type();
    } else {
        hlsl::Type_struct *struct_tp = hlsl::cast<hlsl::Type_struct>(comp_type);
        return struct_tp->get_compound_type(i);
    }
}

// Check if a given LLVM array type is the representation of the HLSL matrix type.
bool HLSLWriterPass::is_matrix_type(llvm::ArrayType *array_type) const
{
    // map some floating point arrays to matrices:
    //   float2[2] -> float2x2
    //   float2[3] -> float3x2
    //   float2[4] -> float4x2
    //   float3[2] -> float2x3
    //   float3[3] -> float3x3
    //   float3[4] -> float4x3
    //   float4[2] -> float2x4
    //   float4[3] -> float3x4
    //   float4[4] -> float4x4

    llvm::Type *array_elem_type = array_type->getElementType();
    if (llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(array_elem_type)) {
        llvm::Type         *vt_elem_type = vt->getElementType();
        llvm::Type::TypeID type_id = vt_elem_type->getTypeID();

        if (type_id == llvm::Type::FloatTyID || type_id == llvm::Type::DoubleTyID) {
            size_t cols = array_type->getNumElements();
            size_t rows = vt->getNumElements();

            if (2 <= cols && cols <= 4 && 2 <= rows && rows <= 4) {
                // map to a matrix type
                return true;
            }
        }
    }
    return false;
}

// Load and process type debug info.
void HLSLWriterPass::process_type_debug_info(llvm::Module const &module)
{
    llvm::DebugInfoFinder finder;

    finder.processModule(module);

    for (llvm::DIType *dt : finder.types()) {
        if (dt->getTag() == llvm::dwarf::DW_TAG_structure_type) {
            llvm::DICompositeType *st  = llvm::cast<llvm::DICompositeType>(dt);
            llvm::StringRef       name = st->getName();

            string s_name(name.begin(), name.end(), m_alloc);

            m_struct_dbg_info[s_name] = st;
        }
    }
}

// Convert an LLVM type to an HLSL type.
hlsl::Type *HLSLWriterPass::convert_type(llvm::Type *type)
{
    switch (type->getTypeID()) {
    case llvm::Type::VoidTyID:
        return m_type_factory.get_void();
    case llvm::Type::HalfTyID:
        return m_type_factory.get_half();
    case llvm::Type::FloatTyID:
        return m_type_factory.get_float();
    case llvm::Type::DoubleTyID:
        return m_type_factory.get_double();
    case llvm::Type::IntegerTyID:
        {
            llvm::IntegerType *int_type = llvm::cast<llvm::IntegerType>(type);
            unsigned int bit_width = int_type->getBitWidth();

            // Support such constructs
            // %X = trunc i32 %Y to i2
            // %Z = icmp i2 %X, 1
            if (bit_width > 1 && bit_width <= 16)
                return m_type_factory.get_min16int();
            if (bit_width > 16 && bit_width <= 32)
                return m_type_factory.get_int();

            switch (int_type->getBitWidth()) {
            case 1:
                return m_type_factory.get_bool();
            case 64:  // TODO: maybe not a good idea
                return m_type_factory.get_int();
            default:
                MDL_ASSERT(!"unexpected LLVM integer type");
                return m_type_factory.get_int();
            }
        }
    case llvm::Type::StructTyID:
        return convert_struct_type(llvm::cast<llvm::StructType>(type));

    case llvm::Type::ArrayTyID:
        {
            llvm::ArrayType *array_type = llvm::cast<llvm::ArrayType>(type);
            size_t          n_elem      = array_type->getNumElements();

            if (n_elem == 0) {
                // map zero length array to void, we cannot handle that in HLSL
                return m_type_factory.get_void();
            }

            // map some floating point arrays to matrices:
            //   float2[2] -> float2x2
            //   float2[3] -> float3x2
            //   float2[4] -> float4x2
            //   float3[2] -> float2x3
            //   float3[3] -> float3x3
            //   float3[4] -> float4x3
            //   float4[2] -> float2x4
            //   float4[3] -> float3x4
            //   float4[4] -> float4x4

            llvm::Type *array_elem_type = array_type->getElementType();
            if (llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(array_elem_type)) {
                llvm::Type         *vt_elem_type = vt->getElementType();
                llvm::Type::TypeID type_id       = vt_elem_type->getTypeID();

                if (type_id == llvm::Type::FloatTyID || type_id == llvm::Type::DoubleTyID) {
                    size_t cols = n_elem;
                    size_t rows = vt->getNumElements();

                    if (2 <= cols && cols <= 4 && 2 <= rows && rows <= 4) {
                        // map to a matrix type
                        hlsl::Type_scalar *res_elem_type =
                            type_id == llvm::Type::FloatTyID ?
                            (hlsl::Type_scalar *)m_type_factory.get_float() :
                            (hlsl::Type_scalar *)m_type_factory.get_double();
                        hlsl::Type_vector *res_vt_type =
                            m_type_factory.get_vector(res_elem_type, rows);
                        return m_type_factory.get_matrix(res_vt_type, cols);
                    }
                }
            }

            hlsl::Type *res_elem_type = convert_type(array_elem_type);
            return m_type_factory.get_array(res_elem_type, n_elem);
        }

    case llvm::Type::VectorTyID:
        {
            llvm::VectorType *vector_type = cast<llvm::VectorType>(type);
            hlsl::Type       *elem_type   = convert_type(vector_type->getElementType());
            if (hlsl::Type_scalar *scalar_type = hlsl::as<hlsl::Type_scalar>(elem_type)) {
                hlsl::Type *res = m_type_factory.get_vector(
                    scalar_type, vector_type->getNumElements());
                if (res != nullptr) {
                    return res;
                }
            }
            MDL_ASSERT(!"invalid vector type");
            return m_type_factory.get_error();
        }

    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        MDL_ASSERT(!"unexpected LLVM type");
        return m_type_factory.get_double();

    case llvm::Type::PointerTyID:
        if (llvm::ArrayType *array_type = llvm::dyn_cast<llvm::ArrayType>(
            type->getPointerElementType()))
        {
            uint64_t size = array_type->getNumElements();
            if (size == 0) {
                // map zero length array to void, we cannot handle that in HLSL
                return m_type_factory.get_void();
            }

            hlsl::Type *base_type = convert_type(array_type->getElementType());
            return m_type_factory.get_array(base_type, size_t(size));
        }
        if (llvm::StructType *struct_type = llvm::dyn_cast<llvm::StructType>(
            type->getPointerElementType()))
        {
            // for pointers to structs, just skip the pointer
            return convert_struct_type(struct_type);
        }
        MDL_ASSERT(!"pointer types not supported, yet");
        return m_type_factory.get_error();

    case llvm::Type::LabelTyID:
    case llvm::Type::MetadataTyID:
    case llvm::Type::X86_MMXTyID:
    case llvm::Type::TokenTyID:
    case llvm::Type::FunctionTyID:
        MDL_ASSERT(!"unexpected LLVM type");
        return m_type_factory.get_error();
    }

    MDL_ASSERT(!"unknown LLVM type");
    return m_type_factory.get_error();
}

// Add a field to a struct declaration.
hlsl::Type_struct::Field HLSLWriterPass::add_struct_field(
    hlsl::Declaration_struct *decl_struct,
    hlsl::Type               *type,
    hlsl::Symbol             *sym)
{
    hlsl::Declaration_field *decl_field =
        m_decl_factory.create_field_declaration(get_type_name(type));
    hlsl::Field_declarator *field_declarator = m_decl_factory.create_field(zero_loc);
    field_declarator->set_name(get_name(zero_loc, sym));
    add_array_specifiers(field_declarator, type);

    decl_field->add_field(field_declarator);
    decl_struct->add(decl_field);

    return hlsl::Type_struct::Field(type, sym);
}

// Add a field to a struct declaration.
hlsl::Type_struct::Field HLSLWriterPass::add_struct_field(
    hlsl::Declaration_struct *decl_struct,
    hlsl::Type               *type,
    char const               *name)
{
    hlsl::Symbol *sym = m_symbol_table.get_symbol(name);
    return add_struct_field(decl_struct, type, sym);
}

// Convert an LLVM struct type to an HLSL struct type.
hlsl::Type *HLSLWriterPass::convert_struct_type(
    llvm::StructType *s_type)
{
    auto it = m_type_cache.find(s_type);
    if (it != m_type_cache.end())
        return it->second;

    string struct_name(m_alloc);

    llvm::DICompositeType *di_type = nullptr;

    if (s_type->hasName()) {
        llvm::StringRef name = s_type->getName();
        if (name == "State_core")
            return create_state_core_struct_type(s_type);
        if (name == "State_environment")
            return create_state_env_struct_type(s_type);
        if (name == "Res_data")
            return create_res_data_struct_type(s_type);
        if (name.startswith("struct.BSDF_")) {
            if (name == "struct.BSDF_sample_data")
                return create_bsdf_sample_data_struct_types(s_type);
            if (name == "struct.BSDF_evaluate_data")
                return create_bsdf_evaluate_data_struct_types(s_type);
            if (name == "struct.BSDF_pdf_data")
                return create_bsdf_pdf_data_struct_types(s_type);
            if (name == "struct.BSDF_auxiliary_data")
                return create_bsdf_auxiliary_data_struct_types(s_type);
        }
        if (name.startswith("struct.EDF_")) {
            if (name == "struct.EDF_sample_data")
                return create_edf_sample_data_struct_types(s_type);
            if (name == "struct.EDF_evaluate_data")
                return create_edf_evaluate_data_struct_types(s_type);
            if (name == "struct.EDF_pdf_data")
                return create_edf_pdf_data_struct_types(s_type);
            if (name == "struct.EDF_auxiliary_data")
                return create_edf_auxiliary_data_struct_types(s_type);
        }
        if (name == "struct.float3") {
            hlsl::Type_scalar *float_type = m_type_factory.get_float();
            hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);
            m_type_cache[s_type] = float3_type;
            return float3_type;
        }
        if (name == "struct.float4") {
            hlsl::Type_scalar *float_type = m_type_factory.get_float();
            hlsl::Type_vector *float4_type = m_type_factory.get_vector(float_type, 4);
            m_type_cache[s_type] = float4_type;
            return float4_type;
        }

        if (name.startswith("struct."))
            name = name.substr(7);

        auto it = m_struct_dbg_info.find(string(name.begin(), name.end(), m_alloc));
        if (it != m_struct_dbg_info.end())
            di_type = it->second;

        // strip the leading "::" from the fully qualified MDL type name
        size_t skip = 0;
        if (name.startswith("::"))
            skip = 2;

        struct_name = string(name.begin() + skip, name.end(), m_alloc);

        // replace all ':' and '.' by '_'
        for (char &ch : struct_name) {
            if (ch == ':' || ch == '.')
                ch = '_';
        }
    }


    unsigned n_fields = s_type->getNumElements();

    // check if we have at least ONE non-void field
    bool has_non_void_field = false;
    for (unsigned i = 0; i < n_fields; ++i) {
        hlsl::Type *field_type = convert_type(s_type->getElementType(i));
        if (hlsl::is<hlsl::Type_void>(field_type))
            continue;

        has_non_void_field = true;
        break;
    }

    if (!has_non_void_field) {
        // an empty struct, map to void
        return m_type_cache[s_type] = m_type_factory.get_void();
    }

    bool is_deriv = m_type_mapper.is_deriv_type(s_type);

    // Some derivative types are part of the API and need to have fixed names.
    // While HLSL supports implicit conversion of different struct types with matching types,
    // Slang does not
    Symbol *struct_sym;
    bool is_api_type = false;
    if (is_deriv && s_type == m_type_mapper.get_deriv_float2_type())
        struct_sym = get_sym("Derived_float2"), is_api_type = true;
    else if (is_deriv && s_type == m_type_mapper.get_deriv_float3_type())
        struct_sym = get_sym("Derived_float3"), is_api_type = true;
    else
        struct_sym = get_unique_hlsl_sym(
            struct_name.c_str(),
            is_deriv ? "deriv_type" : "structtype");

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    Small_VLA<hlsl::Type_struct::Field, 8> fields(m_alloc, n_fields);

    static char const * const deriv_names[] = { "val", "dx", "dy" };

    unsigned n = 0;
    for (unsigned i = 0; i < n_fields; ++i) {
        hlsl::Type *field_type = convert_type(s_type->getElementType(i));
        if (hlsl::is<hlsl::Type_void>(field_type))
            continue;

        hlsl::Symbol *sym = nullptr;

        if (is_deriv) {
            // use predefined field names for derived types
            MDL_ASSERT(i < llvm::array_lengthof(deriv_names));
            sym = get_sym(deriv_names[i]);
        } else if (di_type != nullptr) {
            // get original struct member name from the debug information
            llvm::DINodeArray elements = di_type->getElements();

            if (i < elements.size()) {
                llvm::DIType    *e_tp = llvm::cast<llvm::DIType>(elements[i]);
                llvm::StringRef s     = e_tp->getName();

                string field_name = string(s.begin(), s.end(), m_alloc);

                if (!field_name.empty()) {
                    sym = m_symbol_table.get_symbol(field_name.c_str());
                    if (sym->get_id() < hlsl::Symbol::SYM_USER) {
                        // cannot be used
                        sym = get_unique_hlsl_sym(field_name.c_str(), field_name.c_str());
                    }
                }
            }
        }

        if (sym == nullptr) {
            char name_buf[16];
            snprintf(name_buf, sizeof(name_buf), "m_%u", i);
            sym = m_symbol_table.get_symbol(name_buf);
        }
        fields[n++] = add_struct_field(decl_struct, field_type, sym);
    }

    hlsl::Type_struct *res = m_type_factory.get_struct(
        Array_ref<hlsl::Type_struct::Field>(fields.data(), n), struct_sym);

    hlsl::Definition_table::Scope_transition scope_trans(m_def_tab, m_def_tab.get_global_scope());
    m_def_tab.enter_definition(
        hlsl::Definition::DK_TYPE,
        struct_sym,
        res,
        hlsl::Def_function::DS_UNKNOWN,
        &zero_loc);

    m_type_cache[s_type] = res;

    // don't add API types to the unit to avoid printing
    if (!is_api_type)
        m_unit->add_decl(decl_struct);

    return res;
}

// Create an HLSL struct with the given names for the given LLVM struct type.
hlsl::Type_struct *HLSLWriterPass::create_struct_from_llvm(
    llvm::StructType *type,
    char const *type_name,
    size_t num_field_names,
    char const * const *field_names,
    bool add_to_unit)
{
    if (num_field_names != type->getNumContainedTypes()) {
        MDL_ASSERT(num_field_names == type->getNumContainedTypes());
        return nullptr;
    }

    Small_VLA<hlsl::Type_struct::Field, 16> fields(m_alloc, num_field_names);
    hlsl::Declaration_struct *s_decl = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol             *s_sym  = m_symbol_table.get_symbol(type_name);
    s_decl->set_name(get_name(zero_loc, s_sym));

    for (size_t i = 0; i < num_field_names; ++i) {
        hlsl::Type *f_tp = convert_type(type->getContainedType(unsigned(i)));

        fields[i] = add_struct_field(s_decl, f_tp, field_names[i]);
    }

    hlsl::Type_struct *res = m_type_factory.get_struct(fields, s_sym);
    m_type_cache[type] = res;

    if (add_to_unit)
        m_unit->add_decl(s_decl);

    return res;
}

// Create the HLSL state core struct for the corresponding LLVM struct type.
hlsl::Type_struct *HLSLWriterPass::create_state_core_struct_type(
    llvm::StructType *type)
{
    static char const * const field_names[] = {
        "normal",
        "geom_normal",
        "position",
        "animation_time",
        "text_coords",
        "tangent_u",
        "tangent_v",
        "text_results",
        "ro_data_segment_offset",
        "world_to_object",
        "object_to_world",
        "object_id",
        "meters_per_scene_unit",
        "arg_block_offset",
    };

    size_t n_fields = llvm::array_lengthof(field_names);
    return create_struct_from_llvm(
        type, "Shading_state_material", n_fields, field_names, /*add_to_unit=*/ false);
}

// Create the HLSL state environment struct for the corresponding LLVM struct type.
hlsl::Type_struct *HLSLWriterPass::create_state_env_struct_type(
    llvm::StructType *type)
{
    static char const * const field_names[] = {
        "direction",
        "ro_data_segment_offset",
    };

    size_t n_fields = llvm::array_lengthof(field_names);
    return create_struct_from_llvm(
        type, "Shading_state_environment", n_fields, field_names, /*add_to_unit=*/ false);
}

// Create the HLSL resource data struct for the corresponding LLVM struct type.
hlsl::Type_struct *HLSLWriterPass::create_res_data_struct_type(llvm::StructType *type)
{
    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Res_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type *int_type = m_type_factory.get_int();

    hlsl::Type_struct::Field dummy_field[1];
    dummy_field[0] = add_struct_field(decl_struct, int_type, "dummy");

    hlsl::Type_struct *res = m_type_factory.get_struct(dummy_field, struct_sym);
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Bsdf_sample_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_bsdf_sample_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);
    hlsl::Type_vector *float4_type = m_type_factory.get_vector(float_type, 4);
    hlsl::Type_scalar *int_type = m_type_factory.get_int();

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Bsdf_sample_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct::Field fields[9];

    fields[0] = add_struct_field(decl_struct, float3_type, "ior1");
    fields[1] = add_struct_field(decl_struct, float3_type, "ior2");
    fields[2] = add_struct_field(decl_struct, float3_type, "k1");

    fields[3] = add_struct_field(decl_struct, float3_type, "k2");
    fields[4] = add_struct_field(decl_struct, float4_type, "xi");
    fields[5] = add_struct_field(decl_struct, float_type,  "pdf");
    fields[6] = add_struct_field(decl_struct, float3_type, "bsdf_over_pdf");
    fields[7] = add_struct_field(decl_struct, int_type,    "event_type");
    fields[8] = add_struct_field(decl_struct, int_type,    "handle");

    hlsl::Type_struct *res = m_type_factory.get_struct(fields, struct_sym);
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Bsdf_evaluate_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_bsdf_evaluate_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Bsdf_evaluate_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    MDL_ASSERT(m_df_handle_slot_mode != mi::mdl::DF_HSM_POINTER &&
               "df_handle_slot_mode POINTER is not supported for HLSL");

    hlsl::Type_struct *res;
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE)
    {
        hlsl::Type_struct::Field fields[7];

        fields[0] = add_struct_field(decl_struct, float3_type, "ior1");
        fields[1] = add_struct_field(decl_struct, float3_type, "ior2");
        fields[2] = add_struct_field(decl_struct, float3_type, "k1");
        fields[3] = add_struct_field(decl_struct, float3_type, "k2");
        fields[4] = add_struct_field(decl_struct, float3_type, "bsdf_diffuse");
        fields[5] = add_struct_field(decl_struct, float3_type, "bsdf_glossy");
        fields[6] = add_struct_field(decl_struct, float_type,  "pdf");
        res = m_type_factory.get_struct(fields, struct_sym);
    }
    else // DF_HSM_FIXED (no pointers in HLSL)
    {
        size_t fixed_array_size = static_cast<size_t>(m_df_handle_slot_mode);
        hlsl::Type *float3_array_type = m_type_factory.get_array(float3_type, fixed_array_size);
        hlsl::Type_scalar *int_type = m_type_factory.get_int();

        hlsl::Type_struct::Field fields[8];

        fields[0] = add_struct_field(decl_struct, float3_type, "ior1");
        fields[1] = add_struct_field(decl_struct, float3_type, "ior2");
        fields[2] = add_struct_field(decl_struct, float3_type, "k1");
        fields[3] = add_struct_field(decl_struct, float3_type, "k2");
        fields[4] = add_struct_field(decl_struct, int_type, "handle_offset");
        fields[5] = add_struct_field(decl_struct, float3_array_type, "bsdf_diffuse");
        fields[6] = add_struct_field(decl_struct, float3_array_type, "bsdf_glossy");
        fields[7] = add_struct_field(decl_struct, float_type, "pdf");
        res = m_type_factory.get_struct(fields, struct_sym);
    }

    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Bsdf_pdf_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_bsdf_pdf_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Bsdf_pdf_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct::Field fields[5];

    fields[0] = add_struct_field(decl_struct, float3_type, "ior1");
    fields[1] = add_struct_field(decl_struct, float3_type, "ior2");
    fields[2] = add_struct_field(decl_struct, float3_type, "k1");
    fields[3] = add_struct_field(decl_struct, float3_type, "k2");
    fields[4] = add_struct_field(decl_struct, float_type,  "pdf");

    hlsl::Type_struct *res = m_type_factory.get_struct(fields, struct_sym);
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Bsdf_auxiliary_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_bsdf_auxiliary_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Bsdf_auxiliary_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct *res;
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE)
    {
        hlsl::Type_struct::Field fields[5];

        fields[0] = add_struct_field(decl_struct, float3_type, "ior1");
        fields[1] = add_struct_field(decl_struct, float3_type, "ior2");
        fields[2] = add_struct_field(decl_struct, float3_type, "k1");
        fields[3] = add_struct_field(decl_struct, float3_type, "albedo");
        fields[4] = add_struct_field(decl_struct, float3_type, "normal");
        res = m_type_factory.get_struct(fields, struct_sym);
    }
    else // DF_HSM_FIXED (no pointers in HLSL)
    {
        size_t fixed_array_size = static_cast<size_t>(m_df_handle_slot_mode);
        hlsl::Type *float3_array_type = m_type_factory.get_array(float3_type, fixed_array_size);
        hlsl::Type_scalar *int_type = m_type_factory.get_int();

        hlsl::Type_struct::Field fields[6];

        fields[0] = add_struct_field(decl_struct, float3_type, "ior1");
        fields[1] = add_struct_field(decl_struct, float3_type, "ior2");
        fields[2] = add_struct_field(decl_struct, float3_type, "k1");
        fields[3] = add_struct_field(decl_struct, int_type, "handle_offset");
        fields[4] = add_struct_field(decl_struct, float3_array_type, "albedo");
        fields[5] = add_struct_field(decl_struct, float3_array_type, "normal");
        res = m_type_factory.get_struct(fields, struct_sym);
    }
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Edf_sample_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_edf_sample_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);
    hlsl::Type_vector *float4_type = m_type_factory.get_vector(float_type, 4);
    hlsl::Type_scalar *int_type = m_type_factory.get_int();

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Edf_sample_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct::Field fields[6];

    fields[0] = add_struct_field(decl_struct, float4_type, "xi");
    fields[1] = add_struct_field(decl_struct, float3_type, "k1");
    fields[2] = add_struct_field(decl_struct, float_type,  "pdf");
    fields[3] = add_struct_field(decl_struct, float3_type, "edf_over_pdf");
    fields[4] = add_struct_field(decl_struct, int_type,    "event_type");
    fields[5] = add_struct_field(decl_struct, int_type,    "handle");

    hlsl::Type_struct *res = m_type_factory.get_struct(fields, struct_sym);
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Edf_evaluate_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_edf_evaluate_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Edf_evaluate_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    MDL_ASSERT(m_df_handle_slot_mode != mi::mdl::DF_HSM_POINTER &&
               "df_handle_slot_mode POINTER is not supported for HLSL");

    hlsl::Type_struct *res;
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE)
    {
        hlsl::Type_struct::Field fields[4];

        fields[0] = add_struct_field(decl_struct, float3_type, "k1");
        fields[1] = add_struct_field(decl_struct, float_type, "cos");
        fields[2] = add_struct_field(decl_struct, float3_type, "edf");
        fields[3] = add_struct_field(decl_struct, float_type, "pdf");
        res = m_type_factory.get_struct(fields, struct_sym);
    }
    else // DF_HSM_FIXED (no pointers in HLSL)
    {
        size_t fixed_array_size = static_cast<size_t>(m_df_handle_slot_mode);
        hlsl::Type *float3_array_type = m_type_factory.get_array(float3_type, fixed_array_size);
        hlsl::Type_scalar *int_type = m_type_factory.get_int();

        hlsl::Type_struct::Field fields[5];

        fields[0] = add_struct_field(decl_struct, float3_type, "k1");
        fields[1] = add_struct_field(decl_struct, int_type, "handle_offset");
        fields[2] = add_struct_field(decl_struct, float_type, "cos");
        fields[3] = add_struct_field(decl_struct, float3_array_type, "edf");
        fields[4] = add_struct_field(decl_struct, float_type, "pdf");
        res = m_type_factory.get_struct(fields, struct_sym);
    }
    
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Edf_pdf_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_edf_pdf_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Edf_pdf_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct::Field fields[2];

    fields[0] = add_struct_field(decl_struct, float3_type, "k1");
    fields[1] = add_struct_field(decl_struct, float_type,  "pdf");

    hlsl::Type_struct *res = m_type_factory.get_struct(fields, struct_sym);
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Create the Edf_auxiliary_data struct type used by libbsdf.
hlsl::Type_struct *HLSLWriterPass::create_edf_auxiliary_data_struct_types(llvm::StructType *type)
{
    hlsl::Type_scalar *float_type = m_type_factory.get_float();
    hlsl::Type_vector *float3_type = m_type_factory.get_vector(float_type, 3);

    hlsl::Declaration_struct *decl_struct = m_decl_factory.create_struct(zero_loc);
    hlsl::Symbol *struct_sym = m_symbol_table.get_symbol("Edf_auxiliary_data");
    decl_struct->set_name(get_name(zero_loc, struct_sym));

    hlsl::Type_struct *res;
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE)
    {
        hlsl::Type_struct::Field fields[1];

        fields[0] = add_struct_field(decl_struct, float3_type, "k1");

        res = m_type_factory.get_struct(fields, struct_sym);
    }
    else // DF_HSM_FIXED (no pointers in HLSL)
    {
        hlsl::Type_scalar *int_type = m_type_factory.get_int();

        hlsl::Type_struct::Field fields[2];

        fields[0] = add_struct_field(decl_struct, float3_type, "k1");
        fields[1] = add_struct_field(decl_struct, int_type, "handle_offset");

        res = m_type_factory.get_struct(fields, struct_sym);
    }
    m_type_cache[type] = res;

    // do not add to the unit to avoid printing it

    return res;
}

// Get an HLSL symbol for the given location and LLVM string.
hlsl::Symbol *HLSLWriterPass::get_sym(llvm::StringRef const &str)
{
    return m_symbol_table.get_symbol(str.str().c_str());
}

/// Get an HLSL symbol for the given location and string.
hlsl::Symbol *HLSLWriterPass::get_sym(char const *str)
{
    return m_symbol_table.get_symbol(str);
}

/// Get a valid HLSL symbol for an LLVM string and a template.
hlsl::Symbol *HLSLWriterPass::get_unique_hlsl_sym(
    llvm::StringRef const &str,
    char const            *templ)
{
    return get_unique_hlsl_sym(str.str().c_str(), templ);
}

/// Get a valid HLSL symbol for an LLVM string and a template.
hlsl::Symbol *HLSLWriterPass::get_unique_hlsl_sym(
    char const *str,
    char const *templ)
{
    bool valid = true;
    char const *name = str;

    if (!isalpha(str[0]) && str[0] != '_') {
        valid = false;
    } else {
        for (char const *p = &str[1]; *p != '\0'; ++p) {
            if (!isalnum(*p) && *p != '_') {
                valid = false;
                break;
            }
        }
    }

    if (!valid) {
        str = templ;
        name = nullptr;  // skip lookup and append id before trying
    }

    // check scope
    hlsl::Symbol *sym = nullptr;

    char buffer[65];
    while (true) {
        if (name != nullptr) {
            sym = m_symbol_table.lookup_symbol(name);
            if (sym == nullptr) {
                // this is the first occurrence of this symbol
                sym = m_symbol_table.get_symbol(name);
                break;
            }
            size_t id = sym->get_id();
            if (id >= Symbol::SYM_USER && m_def_tab.get_definition(sym) == nullptr) {
                // symbol exists, but is user defined and no definition in this scope, good
                break;
            }
        }

        // rename it and try again
        strncpy(buffer, str, 58);
        buffer[58] = '\0';
        snprintf(buffer + strlen(buffer), 6, "%u", m_next_unique_name_id);
        buffer[64] = '\0';
        name = buffer;
        ++m_next_unique_name_id;
    }
    return sym;
}

// Get an HLSL type name for an LLVM type.
hlsl::Type_name *HLSLWriterPass::get_type_name(llvm::Type *type)
{
    hlsl::Type *hlsl_type = convert_type(type);
    hlsl::Type_name *type_name = get_type_name(hlsl_type);
    return type_name;
}

// Get an HLSL type name for an HLSL type.
hlsl::Type_name *HLSLWriterPass::get_type_name(hlsl::Type *type)
{
    hlsl::Type::Modifiers mod = type->get_type_modifiers();

    // in HLSL type names have no array specifiers, so skip all arrays
    while (hlsl::Type_array *a_type = hlsl::as<hlsl::Type_array>(type)) {
        type = a_type->get_element_type();
    }

    hlsl::Name      *name      = m_decl_factory.create_name(zero_loc, type->get_sym());
    hlsl::Type_name *type_name = m_decl_factory.create_type_name(zero_loc);
    type_name->set_name(name);
    type_name->set_type(type);

    if (mod & hlsl::Type::MK_CONST) {
        type_name->get_qualifier().set_type_modifier(TM_CONST);
    }
    if (mod & hlsl::Type::MK_COL_MAJOR) {
        type_name->get_qualifier().set_type_modifier(TM_COLUMN_MAJOR);
    }
    if (mod & hlsl::Type::MK_ROW_MAJOR) {
        type_name->get_qualifier().set_type_modifier(TM_ROW_MAJOR);
    }

    return type_name;
}

// Get an HLSL type name for an HLSL name.
hlsl::Type_name *HLSLWriterPass::get_type_name(hlsl::Symbol *sym)
{
    hlsl::Type_name *type_name = m_decl_factory.create_type_name(zero_loc);
    hlsl::Name      *name = get_name(zero_loc, sym);
    type_name->set_name(name);

    return type_name;
}

// Get an HLSL name for the given location and symbol.
hlsl::Name *HLSLWriterPass::get_name(Location loc, Symbol *sym)
{
    return m_decl_factory.create_name(loc, sym);
}

// Get an HLSL name for the given location and string.
hlsl::Name *HLSLWriterPass::get_name(Location loc, const char *str)
{
    hlsl::Symbol *sym = m_symbol_table.get_symbol(str);
    return get_name(loc, sym);
}

// Get an HLSL name for the given location and LLVM string.
hlsl::Name *HLSLWriterPass::get_name(Location loc, llvm::StringRef const &str)
{
    return get_name(loc, str.str().c_str());
}

// Add a parameter to the given function and the current definition table
// and return its symbol.
hlsl::Symbol *HLSLWriterPass::add_func_parameter(
    hlsl::Declaration_function *func,
    char const *param_name,
    hlsl::Type *param_type)
{
    hlsl::Declaration_param *decl_param = m_decl_factory.create_param(
        get_type_name(param_type));
    hlsl::Symbol *param_sym = get_unique_hlsl_sym(param_name, param_name);
    hlsl::Name *param_hlsl_name = get_name(zero_loc, param_sym);
    decl_param->set_name(param_hlsl_name);
    func->add_param(decl_param);

    hlsl::Def_param *param_elem_def = m_def_tab.enter_parameter_definition(
        param_sym, param_type, &param_hlsl_name->get_location());
    param_elem_def->set_declaration(decl_param);
    param_hlsl_name->set_definition(param_elem_def);

    return param_sym;
}

// Get the HLSL symbol of a generated struct constructor.
// This generates the struct constructor, if it does not exist, yet.
hlsl::Def_function *HLSLWriterPass::get_struct_constructor(hlsl::Type_struct *type)
{
    auto it = m_struct_constructor_map.find(type);
    if (it != m_struct_constructor_map.end())
        return it->second;

    // declare the constructor function
    string constr_name("constr_", m_alloc);
    constr_name += type->get_sym()->get_name();

    // generate name in current scope to avoid name clashes
    hlsl::Symbol *func_sym = get_unique_hlsl_sym(constr_name.c_str(), "constr");

    hlsl::Definition_table::Scope_transition transition(m_def_tab, m_def_tab.get_global_scope());

    Declaration_function *decl_func = m_decl_factory.create_function(
        get_type_name(type), get_name(zero_loc, constr_name.c_str()));

    // build the function type
    vector<hlsl::Type_function::Parameter>::Type params(m_alloc);
    for (size_t i = 0, n = type->get_field_count(); i < n; ++i) {
        hlsl::Type_struct::Field *field = type->get_field(i);
        hlsl::Type *field_type = field->get_type();
        if (hlsl::Type_array *at = hlsl::as<hlsl::Type_array>(field_type)) {
            // add one parameter per array element
            hlsl::Type *elem_type = at->get_element_type();
            for (size_t j = 0, num_elems = at->get_size(); j < num_elems; ++j) {
                params.push_back(hlsl::Type_function::Parameter(
                    elem_type, hlsl::Type_function::Parameter::PM_IN));
            }
        } else {
            params.push_back(hlsl::Type_function::Parameter(
                field->get_type(), hlsl::Type_function::Parameter::PM_IN));
        }
    }

    // create the function definition
    hlsl::Type_function *func_type = m_type_factory.get_function(type, params);
    hlsl::Def_function  *func_def  = m_def_tab.enter_function_definition(
        func_sym, func_type, hlsl::Def_function::DS_ELEM_CONSTRUCTOR, &zero_loc);

    func_def->set_declaration(decl_func);

    // create the body
    {
        hlsl::Definition_table::Scope_enter enter(m_def_tab, func_def);

        hlsl::Stmt_compound *block = m_stmt_factory.create_compound(zero_loc);

        // declare the struct "res" variable
        hlsl::Declaration_variable *decl_var  = m_decl_factory.create_variable(get_type_name(type));
        hlsl::Init_declarator      *init_decl = m_decl_factory.create_init_declarator(zero_loc);
        hlsl::Symbol               *var_sym   = get_unique_hlsl_sym("res", "res");
        hlsl::Name                 *var_name  = get_name(zero_loc, var_sym);
        init_decl->set_name(var_name);
        decl_var->add_init(init_decl);

        hlsl::Def_variable *var_def = m_def_tab.enter_variable_definition(
            var_sym, type, &var_name->get_location());
        var_def->set_declaration(decl_var);
        var_name->set_definition(var_def);

        block->add_stmt(m_stmt_factory.create_declaration(decl_var));

        // add a parameter and an assignment to our "res" variable per struct field
        for (size_t i = 0, n = type->get_field_count(); i < n; ++i) {
            hlsl::Type_struct::Field *field = type->get_field(i);

            if (hlsl::Type_array *at = hlsl::as<hlsl::Type_array>(field->get_type())) {
                // add one parameter per array element and fill the array in "res" with them
                hlsl::Type *elem_type = at->get_element_type();
                char const *field_name = field->get_symbol()->get_name();
                for (size_t j = 0, num_elems = at->get_size(); j < num_elems; ++j) {
                    hlsl::Symbol *param_elem_sym =
                        add_func_parameter(decl_func, field_name, elem_type);

                    hlsl::Expr *lvalue_expr = create_field_access(
                        create_reference(var_sym, type),
                        unsigned(i));
                    lvalue_expr = create_array_access(lvalue_expr, unsigned(j));
                    hlsl::Stmt *assign_stmt = create_assign_stmt(
                        lvalue_expr,
                        create_reference(param_elem_sym, elem_type));
                    block->add_stmt(assign_stmt);
                }
                continue;
            }

            hlsl::Symbol *param_sym =
                add_func_parameter(decl_func, field->get_symbol()->get_name(), field->get_type());

            hlsl::Expr *lvalue_expr = create_field_access(
                create_reference(var_sym, type),
                unsigned(i));
            hlsl::Stmt *assign_stmt = create_assign_stmt(
                lvalue_expr,
                create_reference(param_sym, field->get_type()));
            block->add_stmt(assign_stmt);
        }

        // return the struct
        block->add_stmt(m_stmt_factory.create_return(zero_loc, create_reference(var_sym, type)));

        decl_func->set_body(block);
        m_unit->add_decl(decl_func);
    }

    m_struct_constructor_map[type] = func_def;
    return func_def;
}

// Get a name for a given vector index, if possible.
// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
char const *HLSLWriterPass::get_vector_index_str(uint64_t index)
{
    switch (index) {
    case 0: return "x";
    case 1: return "y";
    case 2: return "z";
    case 3: return "w";
    }
    return nullptr;
}

// Get an HLSL symbol for a given vector index, if possible.
// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
hlsl::Symbol *HLSLWriterPass::get_vector_index_sym(uint64_t index)
{
    if (char const *index_name = get_vector_index_str(index)) {
        return get_sym(index_name);
    }
    return nullptr;
}

// Get an HLSL symbol for a given LLVM vector index, if possible.
// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
hlsl::Symbol *HLSLWriterPass::get_vector_index_sym(llvm::Value *index)
{
    if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
        return get_vector_index_sym(ci->getZExtValue());
    }
    return nullptr;
}

// Create a reference to a variable of the given type.
hlsl::Expr *HLSLWriterPass::create_reference(hlsl::Type_name *type_name, hlsl::Type *type)
{
    hlsl::Expr *ref = m_expr_factory.create_reference(type_name);
    ref->set_type(type);
    return ref;
}

// Create a reference to a variable of the given type.
hlsl::Expr *HLSLWriterPass::create_reference(hlsl::Symbol *var_sym, hlsl::Type *type)
{
    return create_reference(get_type_name(var_sym), type);
}

// Create a reference to a variable of the given type.
hlsl::Expr *HLSLWriterPass::create_reference(hlsl::Symbol *var_sym, llvm::Type *type)
{
    return create_reference(var_sym, convert_type(type));
}

// Create a reference to an entity.
hlsl::Expr *HLSLWriterPass::create_reference(hlsl::Definition *def)
{
    hlsl::Expr *expr = create_reference(def->get_symbol(), def->get_type());
    if (hlsl::Expr_ref *ref = hlsl::as<hlsl::Expr_ref>(expr)) {
        ref->set_definition(def);
    }
    return expr;
}

// Create a select expression on a field of a struct.
hlsl::Expr *HLSLWriterPass::create_field_access(hlsl::Expr *struct_expr, unsigned field_index)
{
    hlsl::Type_struct *type = hlsl::as<hlsl::Type_struct>(struct_expr->get_type());
    if (type == nullptr) {
        MDL_ASSERT(!"create_field_access on non-struct expression");
        return m_expr_factory.create_invalid(zero_loc);
    }

    // fold a field access on a constructor call to the corresponding constructor argument
    if (hlsl::Expr_call *call = hlsl::as<hlsl::Expr_call>(struct_expr)) {
        if (hlsl::Expr_ref *callee_ref = hlsl::as<hlsl::Expr_ref>(call->get_callee())) {
            if (hlsl::Def_function *callee_def =
                    hlsl::as<hlsl::Def_function>(callee_ref->get_definition())) {
                if (callee_def->get_semantics() == hlsl::Def_function::DS_ELEM_CONSTRUCTOR) {
                    // if the struct contains any array types, they were flattened,
                    // so we need to get the right argument index
                    size_t arg_idx = 0;
                    for (size_t i = 0, n = type->get_compound_size(); i < n; ++i) {
                        hlsl::Type *field_type = type->get_compound_type(i);
                        if (hlsl::Type_array *at = hlsl::as<hlsl::Type_array>(field_type)) {
                            if (i == field_index) {
                                break;  // we cannot fold this
                            }
                            arg_idx += at->get_size();
                        } else {
                            if (i == field_index) {
                                return call->get_argument(arg_idx);
                            }
                            ++arg_idx;
                        }
                    }
                }
            }
        }
    }

    hlsl::Type_struct::Field *field = type->get_field(field_index);
    if (field == NULL) {
        return m_expr_factory.create_invalid(zero_loc);
    }

    hlsl::Expr *field_ref = create_reference(field->get_symbol(), field->get_type());

    hlsl::Expr *res = m_expr_factory.create_binary(
        Expr_binary::OK_SELECT, struct_expr, field_ref);
    res->set_type(field->get_type());
    return res;
}

// Create an array_subscript expression on an element of an array.
hlsl::Expr *HLSLWriterPass::create_array_access(hlsl::Expr *array, unsigned index)
{
    hlsl::Type_array *type = hlsl::as<hlsl::Type_array>(array->get_type());
    if (type == nullptr) {
        MDL_ASSERT(!"create_array_access on non-array expression");
        return m_expr_factory.create_invalid(zero_loc);
    }

    if (hlsl::Expr_literal *literal_expr = hlsl::as<hlsl::Expr_literal>(array)) {
        if (hlsl::Value_array *val = hlsl::as<hlsl::Value_array>(literal_expr->get_value())) {
            hlsl::Value *extracted_value = val->extract(m_value_factory, index);
            return m_expr_factory.create_literal(zero_loc, extracted_value);
        }
    }

    if (hlsl::Expr_compound *comp_expr = hlsl::as<hlsl::Expr_compound>(array)) {
        return comp_expr->get_element(index);
    }

    hlsl::Expr *index_expr = m_expr_factory.create_literal(
        zero_loc,
        m_value_factory.get_int32(int32_t(index)));
    hlsl::Expr *res = m_expr_factory.create_binary(
        Expr_binary::OK_ARRAY_SUBSCRIPT, array, index_expr);
    res->set_type(type->get_element_type());
    return res;
}

// Create an array_subscript expression on a matrix.
hlsl::Expr *HLSLWriterPass::create_matrix_access(hlsl::Expr *matrix, unsigned index)
{
    hlsl::Type_matrix *type = hlsl::as<hlsl::Type_matrix>(matrix->get_type());
    if (type == nullptr) {
        MDL_ASSERT(!"create_matrix_access on non-matrix expression");
        return m_expr_factory.create_invalid(zero_loc);
    }

    if (hlsl::Expr_literal *literal_expr = hlsl::as<hlsl::Expr_literal>(matrix)) {
        if (hlsl::Value_matrix *val = hlsl::as<hlsl::Value_matrix>(literal_expr->get_value())) {
            hlsl::Value *extracted_value = val->extract(m_value_factory, index);
            return m_expr_factory.create_literal(zero_loc, extracted_value);
        }
    }

    hlsl::Expr *index_expr = m_expr_factory.create_literal(
        zero_loc,
        m_value_factory.get_int32(int32_t(index)));
    hlsl::Expr *res = m_expr_factory.create_binary(
        Expr_binary::OK_ARRAY_SUBSCRIPT, matrix, index_expr);
    res->set_type(type->get_element_type());
    return res;
}

// Create a select expression on a field of a struct.
hlsl::Expr *HLSLWriterPass::create_compound_access(hlsl::Expr *comp_expr, unsigned comp_index)
{
    hlsl::Type *comp_type = comp_expr->get_type()->skip_type_alias();
    if (hlsl::is<hlsl::Type_struct>(comp_type))
        return create_field_access(comp_expr, comp_index);

    if (hlsl::is<hlsl::Type_array>(comp_type))
        return create_array_access(comp_expr, comp_index);

    if (hlsl::is<hlsl::Type_vector>(comp_type))
        return create_vector_access(comp_expr, comp_index);

    if (hlsl::is<hlsl::Type_matrix>(comp_type))
        return create_matrix_access(comp_expr, comp_index);

    MDL_ASSERT(!"create_compound_access on invalid expression");
    return m_expr_factory.create_invalid(zero_loc);
}

// Create an assign statement, assigning an expression to an lvalue expression.
hlsl::Stmt *HLSLWriterPass::create_assign_stmt(
    hlsl::Expr *lvalue,
    hlsl::Expr *expr)
{
    hlsl::Expr *assign = m_expr_factory.create_binary(
        hlsl::Expr_binary::OK_ASSIGN, lvalue, expr);
    assign->set_location(lvalue->get_location());
    assign->set_type(expr->get_type());
    return m_stmt_factory.create_expression(lvalue->get_location(), assign);
}

// Create an assign expression, assigning an expression to a variable.
hlsl::Expr *HLSLWriterPass::create_assign_expr(
    hlsl::Definition *var_def,
    hlsl::Expr       *expr)
{
    hlsl::Expr *lvalue = create_reference(var_def);
    hlsl::Expr *assign = m_expr_factory.create_binary(
        hlsl::Expr_binary::OK_ASSIGN, lvalue, expr);
    assign->set_location(expr->get_location());
    assign->set_type(expr->get_type());
    return assign;
}

// Create an assign statement, assigning an expression to a variable.
hlsl::Stmt *HLSLWriterPass::create_assign_stmt(
    hlsl::Definition *var_def,
    hlsl::Expr       *expr)
{
    hlsl::Expr *assign = create_assign_expr(var_def, expr);
    return m_stmt_factory.create_expression(assign->get_location(), assign);
}

// Add array specifier to an init declarator if necessary.
template<typename Decl_type>
void HLSLWriterPass::add_array_specifiers(Decl_type *decl, hlsl::Type *type)
{
    while (hlsl::Type_array *a_type = hlsl::as<hlsl::Type_array>(type)) {
        type = a_type->get_element_type();

        hlsl::Expr *size = nullptr;
        if (!a_type->is_unsized()) {
            size = this->m_expr_factory.create_literal(
                zero_loc,
                this->m_value_factory.get_int32(int(a_type->get_size())));
        }
        hlsl::Array_specifier *as = this->m_decl_factory.create_array_specifier(zero_loc, size);
        decl->add_array_specifier(as);
    }
}

// Get the constructor for the given HLSL type.
hlsl::Def_function *HLSLWriterPass::lookup_constructor(hlsl::Type *type)
{
    // FIXME: this implementation is wrong, it works only for the fake fillPredefinedEntities()
    if (hlsl::Def_function *def = hlsl::as_or_null<hlsl::Def_function>(
        m_def_tab.get_predef_scope()->find_definition_in_scope(type->get_sym())))
    {
        if (def->get_semantics() == hlsl::Def_function::DS_ELEM_CONSTRUCTOR)
            return def;
    }
    return NULL;
}

// Generates a constructor call for the given HLSL type.
hlsl::Expr *HLSLWriterPass::create_constructor_call(
    hlsl::Type *type,
    Array_ref<hlsl::Expr *> const &args,
    hlsl::Location loc)
{
    hlsl::Expr *ref;

    type = type->skip_type_alias();

    if (hlsl::Type_struct *res_type = hlsl::as<hlsl::Type_struct>(type)) {
        hlsl::Def_function *constr_def = get_struct_constructor(res_type);
        ref = create_reference(constr_def);
    } else if (hlsl::is<hlsl::Type_array>(type)) {
        // this should only appear as part of a complex insertvalue chain
        // and should be broken apart again by a constructor call using this as an argument
        hlsl::Expr *res = m_expr_factory.create_compound(loc, args);
        res->set_type(type);
        return res;
    } else {
        hlsl::Def_function *def = lookup_constructor(type);
        if (def != NULL) {
            ref = create_reference(def);
        } else {
            hlsl::Type_name *constr_name = get_type_name(type);
            ref = create_reference(constr_name, type);
        }
    }

    // for struct types, check whether we have any array arguments
    bool has_array_args = false;
    size_t num_args = 0;
    if (hlsl::is<hlsl::Type_struct>(type)) {
        for (hlsl::Expr *expr : args) {
            if (hlsl::Type_array *at = hlsl::as<hlsl::Type_array>(expr->get_type())) {
                has_array_args = true;
                num_args += at->get_size();
            } else {
                ++num_args;
            }
        }
    }

    hlsl::Expr *res;
    if (has_array_args) {
        // arrays are split into one parameter per array element
        Small_VLA<hlsl::Expr *, 8> split_args(m_alloc, num_args);
        size_t cur_idx = 0;
        for (hlsl::Expr *expr : args) {
            if (hlsl::Type_array *at = hlsl::as<hlsl::Type_array>(expr->get_type())) {
                for (size_t i = 0, n = at->get_size(); i < n; ++i) {
                    split_args[cur_idx++] = create_array_access(expr, unsigned(i));
                }
            } else {
                split_args[cur_idx++] = expr;
            }
        }
        res = m_expr_factory.create_call(ref, split_args);
    } else {
        res = m_expr_factory.create_call(ref, args);
    }

    res->set_type(type);
    res->set_location(loc);
    return res;
}

// Generates a new local variable for an HLSL symbol and an LLVM type.
hlsl::Def_variable *HLSLWriterPass::create_local_var(
    hlsl::Symbol *var_sym,
    llvm::Type   *type,
    bool          add_decl_statement)
{
    hlsl::Type      *var_type      = convert_type(type);
    hlsl::Type_name *var_type_name = get_type_name(var_type);

    hlsl::Declaration_variable *decl_var = m_decl_factory.create_variable(var_type_name);

    hlsl::Init_declarator *init_decl = m_decl_factory.create_init_declarator(zero_loc);
    hlsl::Name *var_name = get_name(zero_loc, var_sym);
    init_decl->set_name(var_name);
    add_array_specifiers(init_decl, var_type);
    decl_var->add_init(init_decl);

    hlsl::Def_variable *var_def = m_def_tab.enter_variable_definition(
        var_sym, var_type, &var_name->get_location());
    var_def->set_declaration(decl_var);
    var_name->set_definition(var_def);

    if (add_decl_statement) {
        // so far, add all declarations to the function scope
        m_cur_start_block->add_stmt(m_stmt_factory.create_declaration(decl_var));
    }

    return var_def;
}

// Generates a new local variable.
hlsl::Def_variable *HLSLWriterPass::create_local_var(
    llvm::Value *value,
    bool        do_not_register,
    bool        add_decl_statement)
{
    llvm::Type *type = value->getType();
    if (llvm::isa<llvm::AllocaInst>(value)) {
        // alloc instructions return a pointer to the allocated space
        llvm::PointerType *p_type = llvm::cast<llvm::PointerType>(type);
        type = p_type->getElementType();
    }

    if (type->isVoidTy())
        return nullptr;

    auto it = m_local_var_map.find(value);
    if (it != m_local_var_map.end()) {
        // this should be a variable, otherwise something did go really wrong
        return hlsl::cast<hlsl::Def_variable>(it->second);
    }

    hlsl::Symbol *var_sym = get_unique_hlsl_sym(value->getName(), "tmp");

    hlsl::Def_variable *var_def = create_local_var(var_sym, type, add_decl_statement);

    if (!do_not_register) {
        m_local_var_map[value] = var_def;
    }
    return var_def;
}

// Generates a new local const variable to hold an LLVM value.
hlsl::Def_variable *HLSLWriterPass::create_local_const(llvm::Constant *cv)
{
    auto it = m_local_var_map.find(cv);
    if (it != m_local_var_map.end())
        return hlsl::as<hlsl::Def_variable>(it->second);

    llvm::Type   *type = cv->getType();
    hlsl::Symbol *cnst_sym = get_unique_hlsl_sym(cv->getName(), "cnst");
    hlsl::Type   *cnst_type = convert_type(type);

    cnst_type = m_type_factory.get_alias(cnst_type, hlsl::Type::MK_CONST);

    hlsl::Type_name *cnst_type_name = get_type_name(cnst_type);

    hlsl::Declaration_variable *decl_cnst = m_decl_factory.create_variable(cnst_type_name);

    hlsl::Init_declarator *init_decl = m_decl_factory.create_init_declarator(zero_loc);
    hlsl::Name *var_name = get_name(zero_loc, cnst_sym);
    init_decl->set_name(var_name);
    add_array_specifiers(init_decl, cnst_type);
    decl_cnst->add_init(init_decl);

    init_decl->set_initializer(translate_constant_expr(cv, /*is_global=*/ false));

    hlsl::Def_variable *cnst_def = m_def_tab.enter_variable_definition(
        cnst_sym, cnst_type, &var_name->get_location());
    cnst_def->set_declaration(decl_cnst);
    var_name->set_definition(cnst_def);

    // so far, add all declarations to the function scope
    m_cur_start_block->add_stmt(m_stmt_factory.create_declaration(decl_cnst));

    m_local_var_map[cv] = cnst_def;
    return cnst_def;
}

// Generates a new local const variable to hold an LLVM value.
hlsl::Def_variable *HLSLWriterPass::create_global_const(llvm::Constant *cv)
{
    auto it = m_global_var_map.find(cv);
    if (it != m_global_var_map.end())
        return hlsl::as<hlsl::Def_variable>(it->second);

    hlsl::Definition_table::Scope_transition scope(m_def_tab, m_def_tab.get_global_scope());

    llvm::Type   *type = cv->getType();
    hlsl::Symbol *cnst_sym = get_unique_hlsl_sym(cv->getName(), "glob_cnst");
    hlsl::Type   *cnst_type = convert_type(type);

    cnst_type = m_type_factory.get_alias(cnst_type, hlsl::Type::MK_CONST);

    hlsl::Type_name *cnst_type_name = get_type_name(cnst_type);
    cnst_type_name->get_qualifier().set_storage_qualifier(hlsl::SQ_STATIC);

    hlsl::Declaration_variable *decl_cnst = m_decl_factory.create_variable(cnst_type_name);

    hlsl::Init_declarator *init_decl = m_decl_factory.create_init_declarator(zero_loc);
    hlsl::Name *var_name = get_name(zero_loc, cnst_sym);
    init_decl->set_name(var_name);
    add_array_specifiers(init_decl, cnst_type);
    decl_cnst->add_init(init_decl);

    init_decl->set_initializer(translate_constant_expr(cv, /*is_global=*/ true));

    hlsl::Def_variable *cnst_def = m_def_tab.enter_variable_definition(
        cnst_sym, cnst_type, &var_name->get_location());
    cnst_def->set_declaration(decl_cnst);
    var_name->set_definition(cnst_def);

    // so far, add all declarations to the function scope
    m_unit->add_decl(decl_cnst);

    m_global_var_map[cv] = cnst_def;
    return cnst_def;
}

// Get or create the in- and out-variables for a PHI node.
std::pair<hlsl::Def_variable *, hlsl::Def_variable *> HLSLWriterPass::get_phi_vars(
    llvm::PHINode *phi)
{
    auto it = m_phi_var_in_out_map.find(phi);
    if (it != m_phi_var_in_out_map.end())
        return it->second;

    // TODO: use debug info to generate a better name
    hlsl::Symbol *in_sym  = get_unique_hlsl_sym("phi_in", "phi_in_");
    hlsl::Symbol *out_sym = get_unique_hlsl_sym("phi_out", "phi_out_");

    // TODO: arrays?
    llvm::Type *type = phi->getType();
    hlsl::Def_variable *phi_in_def  = create_local_var(in_sym, type);
    hlsl::Def_variable *phi_out_def = create_local_var(out_sym, type, true);

    auto res = std::make_pair(phi_in_def, phi_out_def);
    m_phi_var_in_out_map[phi] = res;

    m_local_var_map[phi] = phi_out_def;

    return res;
}

// Get or create the symbol of the in-variable of a PHI node.
hlsl::Def_variable *HLSLWriterPass::get_phi_in_var(llvm::PHINode *phi)
{
    return get_phi_vars(phi).first;
}

// Get or create the symbol of the out-variable of a PHI node, where the in-variable
// will be written to at the start of a block.
hlsl::Def_variable *HLSLWriterPass::get_phi_out_var(llvm::PHINode *phi)
{
    return get_phi_vars(phi).second;
}

/// Check if the given statement is empty.
static bool is_empty_statment(hlsl::Stmt *stmt) {
    if (stmt == nullptr)
        return true;
    if (hlsl::Stmt_expr *expr_stmt = as<hlsl::Stmt_expr>(stmt)) {
        return expr_stmt->get_expression() == nullptr;
    }
    return false;
}

// Joins two statements into a compound statement.
// If one or both statements already are compound statements, they will be merged.
// If one statement is "empty" (nullptr), the other will be returned.
hlsl::Stmt *HLSLWriterPass::join_statements(hlsl::Stmt *head, hlsl::Stmt *tail)
{
    if (is_empty_statment(head))
        return tail;
    if (is_empty_statment(tail))
        return head;

    hlsl::Stmt_compound *block;
    if (head->get_kind() == hlsl::Stmt::SK_COMPOUND) {
        block = hlsl::cast<hlsl::Stmt_compound>(head);
    } else {
        block = m_stmt_factory.create_compound(zero_loc);
        block->add_stmt(head);
    }
    if (tail->get_kind() == hlsl::Stmt::SK_COMPOUND) {
        hlsl::Stmt_compound *tail_list = hlsl::cast<hlsl::Stmt_compound>(tail);
        for (hlsl::Stmt_compound::iterator it = tail_list->begin(); it != tail_list->end(); ) {
            hlsl::Stmt *cur_stmt = it;
            ++it;  // increment iterator before the statement is moved to the other (invasive) list
            block->add_stmt(cur_stmt);
        }
    } else {
        block->add_stmt(tail);
    }
    return block;
}

// Convert the LLVM debug location (if any is attached to the given instruction)
// to an HLSL location.
hlsl::Location HLSLWriterPass::convert_location(llvm::Instruction *I)
{
    if (m_use_dbg) {
        if (llvm::MDNode *md_node = I->getMetadata(llvm::LLVMContext::MD_dbg)) {
            if (llvm::isa<llvm::DILocation>(md_node)) {
                llvm::DebugLoc Loc(md_node);
                unsigned        Line = Loc->getLine();
                unsigned        Column = Loc->getColumn();
                llvm::StringRef fname = Loc->getFilename();

                string s(fname.data(), fname.data() + fname.size(), m_alloc);
                Ref_fname_id_map::const_iterator it = m_ref_fnames.find(s);
                unsigned file_id;
                if (it == m_ref_fnames.end()) {
                    file_id = m_unit->register_filename(s.c_str());
                    m_ref_fnames.insert(Ref_fname_id_map::value_type(s, file_id));
                } else {
                    file_id = it->second;
                }

                return hlsl::Location(Line, Column, file_id);
            }
        }
    }
    return zero_loc;
}

char HLSLWriterPass::ID = 0;

// Creates a HLSL writer pass.
llvm::Pass *createHLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    mi::mdl::IOutput_stream                              &out,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    bool                                                 enable_debug,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list)
{
    return new HLSLWriterPass(
        alloc,
        type_mapper,
        out,
        num_texture_spaces,
        num_texture_results,
        enable_debug,
        exp_func_list,
        df_handle_slot_mode);
}

}  // hlsl
}  // mdl
}  // mi
