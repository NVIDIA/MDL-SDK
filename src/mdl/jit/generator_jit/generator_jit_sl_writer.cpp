/******************************************************************************
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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
#include <mdl/compiler/compilercore/compilercore_errors.h>
#include <mdl/compiler/compilercore/compilercore_messages.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_tools.h>
#include <mdl/compiler/compiler_glsl/compiler_glsl_tools.h>

#include "generator_jit_ast_compute.h"
#include "generator_jit_sl_function.h"
#include "generator_jit_sl_writer.h"
#include "generator_jit_streams.h"

#include "generator_jit_hlsl_writer.h"
#include "generator_jit_glsl_writer.h"

#define NEW_OUT_OF_SSA 1
#define DEBUG_NEW_OUT_OF_SSA 0

#if !NEW_OUT_OF_SSA
#define DEBUG_NEW_OUT_OF_SSA 0
#endif

namespace mi {
namespace mdl {
namespace sl {

// --------------- Generic Writer pass ---------------

// Constructor.
template<typename BasePass>
SLWriterPass<BasePass>::SLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    Generated_code_source                                &code,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    mi::mdl::Options_impl const                          &options,
    mi::mdl::Messages_impl                               &messages,
    bool                                                 enable_debug,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
    mi::mdl::Function_remap const                        &func_remaps,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    bool                                                 enable_opt_remarks)
: Base(ID, alloc, type_mapper, options, messages, enable_debug, enable_opt_remarks)
, m_code(code)
, m_exp_func_list(exp_func_list)
, m_func_remaps(func_remaps)
, m_dg(alloc)
, m_curr_node(nullptr)
, m_curr_func(nullptr)
, m_cur_start_block(nullptr)
, m_local_var_map(
    0, typename Variable_map::hasher(), typename Variable_map::key_equal(), alloc)
, m_global_const_map(
    0, typename Variable_map::hasher(), typename Variable_map::key_equal(), alloc)
, m_phi_var_in_out_map(
    0, typename Phi_map::hasher(), typename Phi_map::key_equal(), alloc)
, m_struct_constructor_map(
    0, typename Struct_map::hasher(), typename Struct_map::key_equal(), alloc)
, m_llvm_function_map(
    0, typename Function_map::hasher(), typename Function_map::key_equal(), alloc)
, m_curr_ast_func(nullptr)
, m_out_def(nullptr)
, m_num_texture_spaces(num_texture_spaces)
, m_num_texture_results(num_texture_results)
, m_df_handle_slot_mode(df_handle_slot_mode)
, m_cur_data_layout(nullptr)
{
    register_api_types(options, df_handle_slot_mode);
}

// Register all API types (structs that are used in the code, but implemented by the user)
template<typename BasePass>
void SLWriterPass<BasePass>::register_api_types(
    mi::mdl::Options_impl const  &options,
    mi::mdl::Df_handle_slot_mode df_handle_slot_mode)
{
    MDL_ASSERT(m_df_handle_slot_mode != mi::mdl::DF_HSM_POINTER &&
        "df_handle_slot_mode POINTER is not supported for GLSL/HLSL");

    // map LLVM State_core type to Shading_state_material
    {
        static struct Core_fields {
            Type_mapper::State_field state_field;
            char const               *name;
        } const field_desc[] = {
            { Type_mapper::STATE_CORE_NORMAL,                "normal" },
            { Type_mapper::STATE_CORE_GEOMETRY_NORMAL,       "geom_normal" },
            { Type_mapper::STATE_CORE_POSITION,              "position" },
            { Type_mapper::STATE_CORE_ANIMATION_TIME,        "animation_time" },
            { Type_mapper::STATE_CORE_TEXTURE_COORDINATE,    "text_coords" },
            { Type_mapper::STATE_CORE_BITANGENTS,            "bitangent" },
            { Type_mapper::STATE_CORE_TANGENT_U,             "tangent_u" },
            { Type_mapper::STATE_CORE_TANGENT_V,             "tangent_v" },
            { Type_mapper::STATE_CORE_TEXT_RESULTS,          "text_results" },
            { Type_mapper::STATE_CORE_RO_DATA_SEG,           "ro_data_segment_offset" },
            { Type_mapper::STATE_CORE_W2O_TRANSFORM,         "world_to_object" },
            { Type_mapper::STATE_CORE_O2W_TRANSFORM,         "object_to_world" },
            { Type_mapper::STATE_CORE_OBJECT_ID,             "object_id" },
            { Type_mapper::STATE_CORE_METERS_PER_SCENE_UNIT, "meters_per_scene_unit" },
            { Type_mapper::STATE_CORE_ARG_BLOCK_OFFSET,      "arg_block_offset" },
        };

        llvm::SmallVector<char const *, 16> fields;

        // check which fields exists and add those to the API description
        for (size_t i = 0; i < llvm::array_lengthof(field_desc); ++i) {
            int idx = Base::m_type_mapper.get_state_index(field_desc[i].state_field);
            if (idx >= 0) {
                fields.push_back(field_desc[i].name);
            }
        }

        char const *core_struct_name =
            options.get_string_option(MDL_JIT_OPTION_SL_CORE_STATE_API_NAME);
        if (core_struct_name == nullptr || core_struct_name[0] == '\0') {
            core_struct_name = "State_core";
        }

        Base::add_api_type_info(
            "State_core",
            core_struct_name,
            Array_ref<char const *>(fields.begin(), fields.end()));
    }

    // map LLVM State_environment type to Shading_state_environment
    {
        static struct Env_fields {
            Type_mapper::State_field state_field;
            char const               *name;
        } const field_desc[] = {
            { Type_mapper::STATE_ENV_DIRECTION,   "direction" },
            { Type_mapper::STATE_ENV_RO_DATA_SEG, "ro_data_segment_offset" },
        };

        llvm::SmallVector<char const *, 4> fields;

        // check which fields exists and add those to the API description
        for (size_t i = 0; i < llvm::array_lengthof(field_desc); ++i) {
            int idx = Base::m_type_mapper.get_state_index(field_desc[i].state_field);
            if (idx >= 0) {
                fields.push_back(field_desc[i].name);
            }
        }

        char const *env_struct_name =
            options.get_string_option(MDL_JIT_OPTION_SL_CORE_STATE_API_NAME);
        if (env_struct_name == nullptr || env_struct_name[0] == '\0') {
            env_struct_name = "State_environment";
        }

        Base::add_api_type_info(
            "State_environment",
            env_struct_name,
            Array_ref<char const *>(fields.begin(), fields.end()));
    }

    // map LLVM struct.BSDF_sample_data to Bsdf_sample_data
    {
        static char const *const fields[] = {
            "ior1",
            "ior2",
            "k1",
            "k2",
            "xi",
            "pdf",
            "bsdf_over_pdf",
            "event_type",
            "handle",
            "flags",
        };
        Base::add_api_type_info("struct.BSDF_sample_data", "Bsdf_sample_data", fields);
    }

    // map LLVM struct.BSDF_evaluate_data to Bsdf_evaluate_data
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
        static char const *const fields[] = {
            "ior1",
            "ior2",
            "k1",
            "k2",
            "bsdf_diffuse",
            "bsdf_glossy",
            "pdf",
            "flags",
        };
        Base::add_api_type_info("struct.BSDF_evaluate_data", "Bsdf_evaluate_data", fields);
    } else /* DF_HSM_FIXED (no pointers in HLSL) */ {
        static char const *const fields[] = {
            "ior1",
            "ior2",
            "k1",
            "k2",
            "handle_offset",
            "bsdf_diffuse",
            "bsdf_glossy",
            "pdf",
            "flags",
        };
        Base::add_api_type_info("struct.BSDF_evaluate_data", "Bsdf_evaluate_data", fields);
    }

    // map LLVM struct.BSDF_pdf_data to Bsdf_pdf_data
    {
        static char const *const fields[] = {
            "ior1",
            "ior2",
            "k1",
            "k2",
            "pdf",
            "flags",
        };
        Base::add_api_type_info("struct.BSDF_pdf_data", "Bsdf_pdf_data", fields);
    }

    /// map struct.BSDF_auxiliary_data to Bsdf_auxiliary_data
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
        static char const *const fields[] = {
            "ior1",
            "ior2",
            "k1",
            "albedo_diffuse",
            "albedo_glossy",
            "normal",
            "roughness",
            "flags",
        };
        Base::add_api_type_info("struct.BSDF_auxiliary_data", "Bsdf_auxiliary_data", fields);
    } else /* DF_HSM_FIXED (no pointers in HLSL) */ {
        static char const *const fields[] = {
            "ior1",
            "ior2",
            "k1",
            "handle_offset",
            "albedo_diffuse",
            "albedo_glossy",
            "normal",
            "roughness",
            "flags",
        };
        Base::add_api_type_info("struct.BSDF_auxiliary_data", "Bsdf_auxiliary_data", fields);
    }

    // map LLVM struct.EDF_sample_data to Edf_sample_data
    {
        static char const *const fields[] = {
            "xi",
            "k1",
            "pdf",
            "edf_over_pdf",
            "event_type",
            "handle",
        };
        Base::add_api_type_info("struct.EDF_sample_data", "Edf_sample_data", fields);
    }

    // map LLVM struct.EDF_evaluate_data to Edf_evaluate_data
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
        static char const *const fields[] = {
            "k1",
            "cos",
            "edf",
            "pdf",
        };
        Base::add_api_type_info("struct.EDF_evaluate_data", "Edf_evaluate_data", fields);
    } else /* DF_HSM_FIXED (no pointers in HLSL) */ {
        static char const *const fields[] = {
            "k1",
            "handle_offset",
            "cos",
            "edf",
            "pdf",
        };
        Base::add_api_type_info("struct.EDF_evaluate_data", "Edf_evaluate_data", fields);
    }

    // map LLVM struct.EDF_pdf_data to Edf_pdf_data
    {
        static char const *const fields[] = {
            "k1",
            "pdf",
        };
        Base::add_api_type_info("struct.EDF_pdf_data", "Edf_pdf_data", fields);
    }

    // Add llvm struct.EDF_auxiliary_data to Edf_auxiliary_data
    if (m_df_handle_slot_mode == mi::mdl::DF_HSM_NONE) {
        static char const *const fields[] = {
            "k1",
        };
        Base::add_api_type_info("struct.EDF_auxiliary_data", "Edf_auxiliary_data", fields);
    } else /* DF_HSM_FIXED (no pointers in HLSL) */ {
        static char const *const fields[] = {
            "k1",
            "handle_offset"
        };
        Base::add_api_type_info("struct.EDF_auxiliary_data", "Edf_auxiliary_data", fields);
    }
}

// Specifies which analysis info is necessary and which is preserved.
template<typename BasePass>
void SLWriterPass<BasePass>::getAnalysisUsage(llvm::AnalysisUsage &usage) const
{
    usage.addRequired<llvm::sl::StructuredControlFlowPass>();
    usage.setPreservesAll();
}

template<typename BasePass>
bool SLWriterPass<BasePass>::runOnModule(llvm::Module &M)
{
    Store<llvm::DataLayout const *> layout_store(m_cur_data_layout, &M.getDataLayout());

    llvm::sl::StructuredControlFlowPass &ast_compute_pass =
        llvm::Pass::getAnalysis<llvm::sl::StructuredControlFlowPass>();

    Base::m_def_tab.transition_to_scope(Base::m_def_tab.get_global_scope());

    // collect type debug info
    Base::enter_type_debug_info(M);

    // create definitions for all user defined functions, so we can create a call graph
    // on the fly
    for (llvm::Function &func : M.functions()) {
        if (func.isDeclaration()) {
            // generate a prototype if necessary
            llvm::StringRef fname = func.getName();

            // ignore LLVM and GLSL/HLSL builtins
            if (!fname.startswith("llvm.") &&
                !fname.startswith("glsl.") &&
                !fname.startswith("hlsl."))
            {
                Def_function *func_def = Base::create_prototype(func);
                if (func_def != nullptr) {
                    m_llvm_function_map[&func] = func_def;
                }
            }
            continue;
        }

        MDL_ASSERT(get_definition(&func) == nullptr && "Definition already exists");

        m_llvm_function_map[&func] = Base::create_definition(&func);
    }

    for (llvm::Function &func : M.functions()) {
        if (!func.isDeclaration()) {
            // create a function definition
            llvm::sl::StructuredFunction const *ast_func =
                ast_compute_pass.getStructuredFunction(&func);
            translate_function(ast_func);
        }
    }

    MDL_ASSERT(
        Base::m_def_tab.get_curr_scope() == Base::m_def_tab.get_global_scope() && "Scope error");

    Enter_func_decl<AST> visitor(Base::m_unit.get());
    m_dg.walk(visitor);

    string prototype(Base::m_alloc);
    String_stream_writer writer(prototype);
    mi::base::Handle<IPrinter> prototype_printer(Base::m_compiler->create_printer(&writer));

    // Beware: This code uses the definitions stored in the m_llvm_function_map which do not
    // survive the core compiler's analyze() call!
    // Hence we must generate them before the code is analyzed. As we print only prototypes here,
    // this should be no problem at all.
    for (mi::mdl::LLVM_code_generator::Exported_function &exp_func : m_exp_func_list) {
        Def_function *def = m_llvm_function_map[exp_func.func];

        // Update function name, which may have been changed due to duplicates or invalid characters
        exp_func.name = def->get_symbol()->get_name();

        prototype.clear();
        prototype_printer->print(def);
        prototype += ';';
        exp_func.set_function_prototype(Base::proto_lang, prototype.c_str());
    }
    // .. so delete it here
    m_llvm_function_map.clear();

    typedef typename list<std::pair<char const *, Symbol *> >::Type Mapped_list;
    Mapped_list mapped(Base::m_alloc);

    for (typename mdl::Function_remap::Remap_entry const *p = m_func_remaps.first();
        p != nullptr;
        p = p->next)
    {
        if (p->used) {
            if (Symbol *dst_sym = Base::m_symbol_table.lookup_symbol(p->to)) {
                mapped.push_back(std::make_pair(p->from, dst_sym));
            }
        }
    }

    // optimize the compilation unit and write it to the output stream
    Base::finalize(M, &m_code, mapped);

    return false;
}

// FIXME: base
// Get the definition for a LLVM function, if one exists.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_function *SLWriterPass<BasePass>::get_definition(
    llvm::Function *func)
{
    typename Function_map::iterator it = m_llvm_function_map.find(func);
    if (it != m_llvm_function_map.end()) {
        return it->second;
    }
    return nullptr;
}

// Generate an AST for the given structured function.
template<typename BasePass>
void SLWriterPass<BasePass>::translate_function(
    llvm::sl::StructuredFunction const *ast_func)
{
    Store<llvm::sl::StructuredFunction const *> ast_store(m_curr_ast_func, ast_func);

    llvm::Function *llvm_func = &ast_func->getFunction();

    Store<llvm::Function *> func_store(m_curr_func, llvm_func);

    Def_function  *func_def  = get_definition(llvm_func);
    Type_function *func_type = func_def->get_type();
    Type          *ret_type  = func_type->get_return_type();
    Type          *out_type  = nullptr;

    // reset the name IDs
    Base::m_next_unique_name_id = 0;

    if (is<Type_void>(ret_type) &&
        !llvm_func->getFunctionType()->getReturnType()->isVoidTy())
    {
        // return type was converted into out parameter
        out_type = Base::return_type_from_first_parameter(func_type);
    }

    // create a new node for this function and make it current
    Store<DG_node<Definition> *> curr_node(m_curr_node, m_dg.get_node(func_def));

    // create the declaration for the function
    Type_name *ret_type_name = Base::get_type_name(ret_type);

    Symbol               *func_sym   = func_def->get_symbol();
    Name                 *func_name  = Base::get_name(Base::zero_loc, func_sym);
    Declaration_function *decl_func  = Base::m_decl_factory.create_function(
        ret_type_name, func_name);

    // copy the noinline attribute to the target language
    if (llvm_func->hasFnAttribute(llvm::Attribute::NoInline)) {
        Base::add_noinline_attribute(decl_func);
    }

    func_def->set_declaration(decl_func);
    func_name->set_definition(func_def);

    // create the function body
    {
        typename Definition_table::Scope_enter enter(Base::m_def_tab, func_def);

        // now create the declarations

        // Note: HLSL does not support array specifiers on types while GLSL does.
        // So far, we always create the "C-Syntax", by adding the array specifiers
        // to the parameter name
        unsigned first_param_ofs = 0;
        if (out_type != nullptr) {
            Type_name         *param_type_name = Base::get_type_name(inner_element_type(out_type));
            Declaration_param *decl_param = Base::m_decl_factory.create_param(param_type_name);
            Base::add_array_specifiers(decl_param, out_type);
            Base::make_out_parameter(param_type_name);


            Symbol *param_sym  = Base::get_unique_sym("p_result", "p_result");
            Name   *param_name = Base::get_name(Base::zero_loc, param_sym);
            decl_param->set_name(param_name);

            m_out_def = Base::m_def_tab.enter_parameter_definition(
                param_sym, out_type, &param_name->get_location());
            m_out_def->set_declaration(decl_param);
            param_name->set_definition(m_out_def);

            decl_func->add_param(decl_param);

            ++first_param_ofs;
        }

        unsigned param_idx = 0;
        for (llvm::Argument &arg_it : llvm_func->args()) {
            llvm::Type *arg_llvm_type = arg_it.getType();
            Type *param_type    = Base::convert_type(arg_llvm_type);

            if (is<Type_void>(param_type)) {
                // skip void typed parameters
                continue;
            }

            Type_name         *param_type_name =
                Base::get_type_name(inner_element_type(param_type));
            Declaration_param *decl_param = Base::m_decl_factory.create_param(param_type_name);
            Base::add_array_specifiers(decl_param, param_type);

            // add parameter qualifier for targets that have it
            Base::add_param_qualifier(param_type_name, func_type, param_idx + first_param_ofs);

            char templ[16];
            snprintf(templ, sizeof(templ), "p_%u",param_idx);

            Symbol *param_sym  = Base::get_unique_sym(arg_it.getName(), templ);
            Name   *param_name = Base::get_name(Base::zero_loc, param_sym);
            decl_param->set_name(param_name);

            Def_param *param_def = Base::m_def_tab.enter_parameter_definition(
                param_sym, param_type, &param_name->get_location());
            param_def->set_declaration(decl_param);
            param_name->set_definition(param_def);

            m_local_var_map[&arg_it] = param_def;

            decl_func->add_param(decl_param);

            ++param_idx;
        }

        // local variables possibly used in outer scopes will be declared in the
        // first block of the function
        m_cur_start_block = Base::m_stmt_factory.create_compound(Base::zero_loc);

        Stmt *stmt = translate_region(ast_func->getBody());
        Stmt *body = join_statements(m_cur_start_block, stmt);
        decl_func->set_body(body);
    }

    // cleanup
    m_cur_start_block = nullptr;
    m_out_def         = nullptr;
    m_local_var_map.clear();
}

// Translate a region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_region(
    llvm::sl::Region const *region)
{
    switch (region->get_kind()) {
    case llvm::sl::Region::SK_BLOCK:
        return translate_block(region);
    case llvm::sl::Region::SK_SEQUENCE:
        {
            llvm::sl::RegionSequence const *seq =
                llvm::sl::cast<llvm::sl::RegionSequence>(region);
            Stmt *stmt = translate_region(seq->getHead());
            for (llvm::sl::Region const *next_region: seq->getTail()) {
                Stmt *next_stmt = translate_region(next_region);
                stmt = join_statements(stmt, next_stmt);
            }
            return stmt;
        }
    case llvm::sl::Region::SK_NATURAL_LOOP:
        return translate_natural(llvm::sl::cast<llvm::sl::RegionNaturalLoop>(region));
    case llvm::sl::Region::SK_IF_THEN:
        return translate_if_then(llvm::sl::cast<llvm::sl::RegionIfThen>(region));
    case llvm::sl::Region::SK_IF_THEN_ELSE:
        return translate_if_then_else(llvm::sl::cast<llvm::sl::RegionIfThenElse>(region));
    case llvm::sl::Region::SK_BREAK:
        return translate_break(llvm::sl::cast<llvm::sl::RegionBreak>(region));
    case llvm::sl::Region::SK_CONTINUE:
        return translate_continue(llvm::sl::cast<llvm::sl::RegionContinue>(region));
    case llvm::sl::Region::SK_RETURN:
        return translate_return(llvm::sl::cast<llvm::sl::RegionReturn>(region));
    default:
        MDL_ASSERT(!"Region kind not supported, yet");
        Base::error(mi::mdl::INTERNAL_JIT_BACKEND_ERROR);
        return Base::m_stmt_factory.create_invalid(Base::zero_loc);
    }
}

/// Check if the given instruction is a InsertValue instruction and all its users are
/// InsertValues too.
static bool part_of_InsertValue_chain(llvm::Instruction &inst)
{
    llvm::InsertValueInst *iv = llvm::dyn_cast<llvm::InsertValueInst>(&inst);
    if (iv == nullptr) {
        return false;
    }

    for (llvm::Value *user : iv->users()) {
        if (!llvm::isa<llvm::InsertValueInst>(user)) {
            return false;
        }
    }
    return true;
}

#if NEW_OUT_OF_SSA

#if DEBUG_NEW_OUT_OF_SSA
void print_ssa_value(llvm::Value *v) {
    if (v == reinterpret_cast<llvm::Value*>(static_cast<intptr_t>(1))) {
        printf("swp");
        return;
    }
    llvm::StringRef name = v->getName();
    if (name.size() > 0) {
        printf("%s", name.str().c_str());
    } else if (llvm::isa<llvm::Constant>(v)) {
        printf("<constant:%p>", v);
    } else {
        printf("<value:%p>", v);
    }
}
#endif

/// Register transfer graph. This is inspired by the graph described in
/// Chapter 4 of Sebastian Hack, "Register Allocation for Programs in SSA Form",
/// PhD Thesis, Universitaet Fridericiana zu Karlsruhe (TH), 2006.
/// 
/// The graph models the value transfer between SSA values on the edges of
/// the CFG. Each edge is a copy operation between a PHI node input and the
/// corresponding PHI node output.
/// 
/// The algorithm for copying inputs to outputs works as follows:
/// 
/// 1. When a node with an in-degree of 0 exists, a copy instruction to
///    one of the successors can be emitted, and the edge removed.
///    Copies to the same variable can be ignored.
/// 2. If no such node exists, there exists a cycle in the graph. In
///    that case, we select an arbitrary edge, create a temporary, copy
///    one the source node on the edge to the temporary, remove the edge to
///    break the cycle,
///    process the rest of the graph and emit a copy from the temporary
///    to the target node of the original edge.
/// 3. Repeat until there are no more edges.
/// 
/// In contrast to the original algorithm, we do not have to care about
/// spilling, as we can simply create temporaries. Also, when removing an edge,
/// we do not move successors of `from` to `to` (which the original algoritm
/// does), because we do not have to care about the number of temporaries.
class Rtg {
public:
    using Id = unsigned;
    using IdValueMap = std::map<Id, llvm::Value *>;
    using ValueIdMap = std::map<llvm::Value *, Id>;

    using RtgSet = std::set<Id>;
    using PredMap = std::map<Id, Id>;
    using SuccMap = std::map<Id, RtgSet>;

private:
    /// Map from internal node ids to LLVM values.
    IdValueMap ids_to_values;
    /// Map from LLVM values to internal node ids.
    ValueIdMap values_to_ids;

    /// Map from nodes to a set of their successors.
    SuccMap m_succ_map;
    /// Map from nodes to their respective predecessor. Nodes can only have
    /// 0 or 1 predecessors.
    PredMap m_pred_map;
    /// Number of edges in the graph.
    unsigned m_edge_count{ 0 };
    /// Counter to generate internal node ids.
    Id m_next_id{ 0 };

public:
    Id find_id(llvm::Value *v) {
        auto it = values_to_ids.find(v);
        if (it != values_to_ids.end()) {
            return it->second;
        } else {
            Id id = m_next_id++;
            values_to_ids[v] = id;
            ids_to_values[id] = v;
            return id;
        }
    }

    /// Add an edge to the register transfer graph, if it doesn't already exist.
    void add_edge(llvm::Value* from_val, llvm::Value* to_val) {
        Id from = find_id(from_val);
        Id to = find_id(to_val);
        auto p = m_pred_map.find(to);
        if (p == m_pred_map.end()) {
            m_succ_map[from].insert(to);
            m_pred_map[to] = from;
            m_edge_count += 1;
        } else {
            MDL_ASSERT(p->second == from);
        }
    }

    /// Remove an edge from the register transfer graph.
    /// Precondition: the edge must exist in the graph.
    void remove_edge(llvm::Value* from_val, llvm::Value* to_val) {
        Id from = find_id(from_val);
        Id to = find_id(to_val);
        MDL_ASSERT(m_pred_map.find(to)->second == from);
        m_succ_map[from].erase(m_succ_map[from].find(to));
        m_pred_map.erase(m_pred_map.find(to));
        m_edge_count -= 1;
    }

    /// Return the number of edges in the graph.
    size_t edge_count() {
        return m_edge_count;
    }

#if DEBUG_NEW_OUT_OF_SSA
    void dump() {
        printf("\n");
        for (auto from_to : m_succ_map) {
            for (auto to : from_to.second) {
                printf("[-] ");
                print_ssa_value(ids_to_values[from_to.first]);
                printf(" -> ");
                print_ssa_value(ids_to_values[to]);
                printf("\n");
            }
        }
    }
#endif

    llvm::Value *predecessor(llvm::Value* node) {
        return ids_to_values[m_pred_map[values_to_ids[node]]];
    }

    /// Select a node that has no successors, but does have at least
    /// one predecessor. Note that the returned node's predecessor can
    /// be itself. The caller has to handle that case (e.g. by simply
    /// ignoring it, since a copy would be useless).
    /// 
    /// Return nullptr if no such edge can be found.
    llvm::Value *pick_without_outgoing_edges() {
        // Pick edge r->s, with outdegree of s == 0 by iterating over
        // m_pred_map (which means there is a predecessor) and then
        // checking whether the successor set is empty.
        for (auto &it : m_pred_map) {
            Id s = it.first;
            if (m_succ_map[s].size() == 0) {
                auto it = ids_to_values.find(s);
                MDL_ASSERT(it != ids_to_values.end());
                return it->second;
            }
        }
        return nullptr;
    }

    /// Return an arbitrary edge from the graph. If there are no edges,
    /// return a pair (nullptr, nullptr).
    std::pair <llvm::Value *, llvm::Value *> pick_arbitrary() {
        // We just take the first entry in the predecessor map, exchanging
        // the from and to nodes.
        if (m_pred_map.size() > 0) {
            auto p = m_pred_map.begin();
            return std::make_pair(ids_to_values[p->second], ids_to_values[p->first]);
        } else {
            return std::make_pair(nullptr, nullptr);
        }
    }
};
#endif

namespace {

    // Return true if the value phi is used in any terminating instruction,
    // which requires special handling in the out-of-SSA transformation.
    bool used_in_terminator(llvm::Value* phi)
    {
        for (llvm::Value* user : phi->users()) {
            if (llvm::isa<llvm::Instruction>(user)) {
                llvm::Instruction *inst = llvm::cast<llvm::Instruction>(user);
                if (inst->isTerminator()) {
                    return true;
                }
            }
        }
        return false;
    }

    bool flows_into_terminator(llvm::Value *value, llvm::BasicBlock *bb, size_t limit = 10)
    {
        if (limit == 0) {
            // Safely return true if nested too deply.
            return true;
        }
        for (llvm::Value *user : value->users()) {
            if (llvm::isa<llvm::Instruction>(user)) {
                llvm::Instruction *inst = llvm::cast<llvm::Instruction>(user);
                if (inst->getParent() != bb) {
                    // Uses across basic blocks are already taken care of:
                    // 1. Normal values are materialized.
                    // 2. PHI nodes are handled by out-of-SSA transformation.
                    // Therefore, we can safely ignore this use.
                    return false;
                }
                if (inst->isTerminator()) {
                    return true;
                }
            }
            if (flows_into_terminator(user, bb, limit - 1)) {
                return true;
            }
        }
        return false;
    }
}

static bool has_non_const_index(llvm::InsertElementInst *inst)
{
    llvm::Value *index = inst->getOperand(2);
    return !llvm::isa<llvm::ConstantInt>(index);
}

// Get the AST region for a block.
template<typename BasePass>
llvm::sl::Region *SLWriterPass<BasePass>::get_ast_region(
    llvm::BasicBlock *bb)
{
    llvm::sl::Region        *r     = m_curr_ast_func->getRegion(bb);
    llvm::sl::RegionComplex *owner = r->getOwnerRegion();
    if (false && owner->getHead() == r) {
        // ugly special case: the AST element of a head is the AST node itself...

        if (!llvm::sl::is<llvm::sl::RegionNaturalLoop>(owner)) {
            // except for natural loops
            return owner;
        }
    }
    return r;
}

// Check if instructions in two basic blocks would end up in different scopes.
template<typename BasePass>
bool SLWriterPass<BasePass>::in_different_scopes(
    llvm::BasicBlock *bb1,
    llvm::BasicBlock *bb2)
{
#if 1
    return bb1 != bb2;
#else
    // currently buggy
    if (bb1 == bb2) {
        return false;
    }

    llvm::sl::Region *r1 = get_ast_region(bb1);
    llvm::sl::Region *r2 = get_ast_region(bb2);
    unsigned d1 = r1->get_depth();
    unsigned d2 = r2->get_depth();

    if (d1 == d2) {
        // same depth, but different regions, different scope
        return true;
    }
    if (d2 > d1) {
        do {
            r2 = r2->getOwnerRegion();
        } while (r2 != nullptr && r2->get_depth() > d1);
    } else {
        do {
            r1 = r1->getOwnerRegion();
        } while (r1 != nullptr && r1->get_depth() > d2);
    }

    if (r1 == r2) {
        return false;
    }

    // special case: if the regions are different, but both part of the same sequence,
    // then the scope is the same
    llvm::sl::RegionComplex *owner = r1->getOwnerRegion();
    if (llvm::sl::is<llvm::sl::RegionSequence>(owner) && owner == r2->getOwnerRegion()) {
        return false;
    }

    return true;
#endif
}

// Translate a block-region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_block(
    llvm::sl::Region const *region)
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
                if (name.startswith("hlsl.modf.") || name.startswith("hlsl.sincos.") ||
                        name.startswith("glsl.modf.") || name.startswith("glsl.sincos.")) {
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
            if (has_local_var(v)) {
                continue;
            }
            if (gen_insts_set.count(v) != 0) {
                continue;
            }
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
        } else if (llvm::isa<llvm::ArrayType>(value.getType()) &&
                !part_of_InsertValue_chain(value)) {
            // we need temporaries for every array typed return (except inner nodes of a
            // insertValue chain), because we do not have array typed literals in HLSL, only
            // compound expressions which can only be used in variable initializers
            gen_statement = true;
        } else if (llvm::isa<llvm::SelectInst>(value) &&
                llvm::isa<llvm::StructType>(value.getType())) {
            // conditional operator only supports results with numeric scalar, vector
            // or matrix types, so we need to generate an if-statement for it
            gen_statement = true;
        } else if (llvm::isa<llvm::InsertElementInst>(value) &&
                has_non_const_index(llvm::cast<llvm::InsertElementInst>(&value))) {
            // InsertElement with non-constant index needs a statement
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
                    case llvm::Intrinsic::experimental_noalias_scope_decl:
                                                          // llvm.experimental.noalias.scope.decl
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
                                if (name.startswith("hlsl.modf.") ||
                                        name.startswith("hlsl.sincos.") ||
                                        name.startswith("glsl.modf.") ||
                                        name.startswith("glsl.sincos.")) {
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
                if (dirty_base_pointers.count(base_pointer) != 0) {
                    gen_statement = true;
                }
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

                    if (!llvm::isa<llvm::Instruction>(user))
                        continue;

                    llvm::Instruction *user_inst = llvm::cast<llvm::Instruction>(user);

                    // is inst a live-out?
                    if (user_inst->getParent() != bb) {
                        gen_statement = true;
                        break;
                    }

                    // some constructs might require materialization of the instruction,
                    // for instance GLSL does not support Xor on non boolean types
                    if (Base::must_be_materialized(user_inst)) {
                        gen_statement = true;
                        break;
                    }
                }
            }

            if (!gen_statement) {
                bool has_phi_operands = false;
                if (llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(&value)) {
                    for (auto &operand : value.operands()) {
                        if (llvm::isa<llvm::PHINode>(&operand)) {
                            has_phi_operands = true;
                            break;
                        }
                    }
                }
                if (has_phi_operands && flows_into_terminator(&value, bb)) {
                    gen_statement = true;
                }
            }
        }

        if (gen_statement) {
            // skip any pointer casts
            llvm::Value *cur_val = &value;
            while (llvm::Instruction *cur_inst = llvm::dyn_cast<llvm::Instruction>(cur_val)) {
                if (!(cur_inst->isCast() && cur_inst->getType()->isPointerTy())) {
                    break;
                }
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

    llvm::SmallVector<Stmt *, 8> stmts;

#if !NEW_OUT_OF_SSA
    // check for phis and assign to their in-variables from the predecessor blocks to
    // the out-variables at the beginning of this block
    for (llvm::PHINode &phi : bb->phis()) {
        auto phi_in_out_vars = get_phi_vars(&phi);
        Expr *phi_in_expr = Base::create_reference(phi_in_out_vars.first);
        stmts.push_back(create_assign_stmt(phi_in_out_vars.second, phi_in_expr));
    }
#else
    // When a variable is used in a terminating instruction, we have to introduce a
    // variable for holding it's value across a control-flow edge, so that the old 
    // variable's value is available to the terminating instruction (for example, a 
    // branch).
    for (llvm::PHINode& phi : bb->phis()) {
        if (used_in_terminator(&phi)) {
            auto phi_in_out_vars = get_phi_vars(&phi, true);
            Expr* phi_in_expr = Base::create_reference(phi_in_out_vars.first);
            stmts.push_back(create_assign_stmt(phi_in_out_vars.second, phi_in_expr));
        }
    }
#endif // !NEW_OUT_OF_SSA

    for (llvm::Value *value : gen_insts) {
        // handle intrinsic functions
        if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(value)) {
            if (translate_intrinsic_call(stmts, call)) {
                continue;
            }
        }

        if (llvm::StoreInst *store = llvm::dyn_cast<llvm::StoreInst>(value)) {
            translate_store(stmts, store);
            continue;
        }

        if (llvm::SelectInst *sel = llvm::dyn_cast<llvm::SelectInst>(value)) {
            if (llvm::isa<llvm::StructType>(sel->getType())) {
                Def_variable *var_def = create_local_var(value, /*do_not_register=*/true);
                Stmt *stmt = translate_struct_select(sel, var_def);
                stmts.push_back(stmt);
                if (var_def != nullptr) {
                    m_local_var_map[value] = var_def;
                }
                continue;
            }
        }

        bool move_decl_to_prolog = false;
        for (auto user : value->users()) {
            // is inst a live-out?
            if (in_different_scopes(llvm::cast<llvm::Instruction>(user)->getParent(), bb)) {
                move_decl_to_prolog = true;
                break;
            }
        }

        Def_variable *var_def = create_local_var(
            value, /*do_not_register=*/ true, move_decl_to_prolog);

        if (llvm::InsertElementInst *insert = llvm::dyn_cast<llvm::InsertElementInst>(value)) {
            if (has_non_const_index(insert)) {
                // For dynamic indices, we cannot just create a constructor expression statement,
                // as in translate_expr_insertelement().
                // First initialize the new local variable with the input vector.
                Expr *input_vector = translate_expr(insert->getOperand(0));

                if (!move_decl_to_prolog) {
                    Declaration_variable *decl_var = var_def->get_declaration();
                    for (Init_declarator &init_decl : *decl_var) {
                        if (init_decl.get_name()->get_symbol() == var_def->get_symbol()) {
                            init_decl.set_initializer(input_vector);
                        }
                    }
                    // insert variable declaration here
                    stmts.push_back(Base::m_stmt_factory.create_declaration(decl_var));
                } else {
                    stmts.push_back(create_assign_stmt(var_def, input_vector));
                }

                // then assign the new value to the dynamic index in a second statement.
                Expr *elem_index = translate_expr(insert->getOperand(2));
                Expr *elem_access = Base::create_binary(
                    Expr_binary::OK_ARRAY_SUBSCRIPT, Base::create_reference(var_def), elem_index);
                Expr *new_value = translate_expr(insert->getOperand(1));
                stmts.push_back(create_assign_stmt(elem_access, new_value));

                // register now
                if (var_def != nullptr) {
                    m_local_var_map[value] = var_def;
                }
                continue;
            }
        }
        Expr *res = translate_expr(value);
        if (var_def != nullptr) {
            // don't initialize with itself
            if (!is_ref_to_def(res, var_def)) {
                if (!move_decl_to_prolog) {
                    // set variable initializer
                    Declaration_variable *decl_var = var_def->get_declaration();
                    for (Init_declarator &init_decl : *decl_var) {
                        if (init_decl.get_name()->get_symbol() == var_def->get_symbol()) {
                            init_decl.set_initializer(res);
                        }
                    }

                    // insert variable declaration here
                    stmts.push_back(Base::m_stmt_factory.create_declaration(decl_var));
                } else {
                    stmts.push_back(create_assign_stmt(var_def, res));
                }
            }

            // register now
            m_local_var_map[value] = var_def;
        } else {
            // for void calls, var_def is nullptr
            MDL_ASSERT(is<Expr_call>(res));

            // ignore reference expressions
            if (!is<Expr_ref>(res)) {
                stmts.push_back(Base::m_stmt_factory.create_expression(res->get_location(), res));
            }
        }
    }

#if NEW_OUT_OF_SSA
    // Used as a marker to add a dependency on the "swap" temporary.
    llvm::Value  *swap_marker = reinterpret_cast<llvm::Value *>(static_cast<intptr_t>(1));
    Def_variable *swap_tmp_v  = nullptr;
#endif // NEW_OUT_OF_SSA

#if NEW_OUT_OF_SSA
    Rtg rtg;
#endif // NEW_OUT_OF_SSA

    // check for phis in successor blocks and assign to their in-variables at the end of this block
    for (llvm::BasicBlock *succ_bb : successors(bb->getTerminator())) {
        for (llvm::PHINode &phi : succ_bb->phis()) {
            bool used_in_branch = used_in_terminator(&phi);
            for (unsigned i = 0, n = phi.getNumIncomingValues(); i < n; ++i) {
                if (phi.getIncomingBlock(i) == bb) {
                    llvm::Value *incoming = phi.getIncomingValue(i);
#if NEW_OUT_OF_SSA
                    if (used_in_branch) {
                        // This is a variable used in a terminating instruction: copy it's value
                        // into the temporary that holds it's value for the successor block.
                        Def_variable *phi_in_var = get_phi_in_var(&phi, true);
                        Expr *res = translate_expr(incoming);
                        stmts.push_back(create_assign_stmt(phi_in_var, res));
                    } else {
                        // Variables not used in terminating instruction go into the register 
                        // transfer graph.
                        rtg.add_edge(incoming, &phi);
                    }
#else // !NEW_OUT_OF_SSA
                    Def_variable *phi_in_var = get_phi_in_var(&phi);
                    Expr *res = translate_expr(incoming);
                    stmts.push_back(create_assign_stmt(phi_in_var, res));
#endif // !NEW_OUT_OF_SSA
                }
            }
        }
    }
#if NEW_OUT_OF_SSA

        while (rtg.edge_count() > 0) {
#if DEBUG_NEW_OUT_OF_SSA
            rtg.dump();
#endif
            llvm::Value *to = rtg.pick_without_outgoing_edges();
            if (to != nullptr) {

#if DEBUG_NEW_OUT_OF_SSA
                printf("[select] ");
                print_ssa_value(to);
                printf("\n");
#endif
                llvm::Value  *from        = rtg.predecessor(to);
                Def_variable *phi_out_var = get_phi_out_var(llvm::cast<llvm::PHINode>(to));

                Expr *res;
                if (from == swap_marker) {
                    res = Base::create_reference(swap_tmp_v);
                } else {
                    res = translate_expr(from);
                }
#if DEBUG_NEW_OUT_OF_SSA
                printf("  [generate] ");
                if (Expr_ref* v = as<Expr_ref>(res)) {
                    printf("%s", v->get_definition()->get_symbol()->get_name());
                } else if (Expr_literal* l = as<Expr_literal>(res)) {
                    printf("<literal>");
                } else {
                    printf("<node>");
                }
                printf(" -> %s\n", phi_out_var->get_symbol()->get_name());
#endif
                bool generate = true;
                if (Expr_ref *ref = as<Expr_ref>(res)) {
                    if (ref->get_definition()->get_symbol() == phi_out_var->get_symbol()) {
                        generate = false;
                    }
                }
                if (generate) {
                    stmts.push_back(create_assign_stmt(phi_out_var, res));
                }
                rtg.remove_edge(from, to);
            } else {
                // Since we did not find a variable without an outgoing edge in, we choose an
                // arbitrary edge and introduce a temporary variable. We move the source of
                // the edge into the temporary and add an edge from the temporary to the
                // destination of the picked edge.

                auto picked_edge = rtg.pick_arbitrary();

                // Note: the pair elements are non-null, because the graph is not empty here.
                llvm::Value *from = picked_edge.first;
                llvm::Value *to   = picked_edge.second;
#if DEBUG_NEW_OUT_OF_SSA
                printf("[cycle detected]\n  [select cycle-breaking edge] ");
                print_ssa_value(from);
                printf(" -> ");
                print_ssa_value(to);
                printf("\n");
#endif
                // Create temporary variable to break cycle. We only to this once per block,
                // because we can reuse the variable: if there are multiple cycles, they will
                // be done one after the other.
                if (swap_tmp_v == nullptr) {
                    swap_tmp_v = create_local_var(
                        Base::get_unique_sym("swp", "swp"), from->getType());
                }
                Expr *swap_tmp = Base::create_reference(swap_tmp_v);

                // Assign from value to temporary. The `from` node is now free to use.
                Expr *from_expr = translate_expr(from);
                stmts.push_back(create_assign_stmt(swap_tmp, from_expr));

#if DEBUG_NEW_OUT_OF_SSA
                printf("  [generate] ");
                if (Expr_ref* v = as<Expr_ref>(from_expr)) {
                    printf("%s", v->get_definition()->get_symbol()->get_name());
                } else if (Expr_literal* l = as<Expr_literal>(from_expr)) {
                    printf("<literal>");
                } else {
                    printf("<node>");
                }
                printf(" -> swp\n");
#endif

#if DEBUG_NEW_OUT_OF_SSA
                printf("  [adding edge] swp -> ");
                print_ssa_value(to);
                printf("\n");
#endif
                // Remove original edge, as the from node is now free to use.
                rtg.remove_edge(from, to);
                // Add an edge from the temporary swap variable to the destination of the
                // cycle-breaking edge.
                rtg.add_edge(swap_marker, to);
            }
        }
#endif // NEW_OUT_OF_SSA

    size_t n_stmts = stmts.size();
    if (n_stmts > 1) {
        return Base::m_stmt_factory.create_compound(
            Base::zero_loc, Array_ref<Stmt *>(stmts.data(), n_stmts));
    } else if (n_stmts == 1) {
        return stmts[0];
    } else {
        // no code was generated; create an empty (expression) statement
        return Base::m_stmt_factory.create_expression(Base::zero_loc, /*expr=*/nullptr);
    }
}

// Translate a natural loop into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_natural(
    llvm::sl::RegionNaturalLoop const *region)
{
    llvm::sl::Region *body      = region->getHead();
    Stmt             *body_stmt = translate_region(body);
    Expr             *cond      = nullptr;

    // TODO: currently disabled because of variable declaration at assignment
    //       (see deriv_tests::test_math_emission_color_2 with enabled derivatives)
#if 0
    // check if we can transform it into a do-while loop
    if (Stmt_compound *c_body = as<Stmt_compound>(body_stmt)) {
        if (Stmt *last = c_body->back()) {
            if (Stmt_if *if_stmt = as<Stmt_if>(last)) {
                if (if_stmt->get_else_statement() == nullptr &&
                    is<Stmt_break>(if_stmt->get_then_statement()))
                {
                    // the last statement of our loop is a if(cond) break, turn the natural loop
                    // into a do { ... } while(! cond)

                    cond = if_stmt->get_condition();
                    Type *cond_type = cond->get_type();
                    cond = Base::create_unary(
                        Base::zero_loc,
                        Expr_unary::OK_LOGICAL_NOT,
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
        cond = Base::m_expr_factory.create_literal(
            Base::zero_loc,
            Base::m_value_factory.get_bool(true));
    }
    return Base::m_stmt_factory.create_do_while(Base::zero_loc, cond, body_stmt);
}

// Translate a if-then-region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_if_then(
    llvm::sl::RegionIfThen const *region)
{
    llvm::sl::Region *head      = region->getHead();
    Stmt             *head_stmt = translate_region(head);

    llvm::BranchInst *branch    = llvm::cast<llvm::BranchInst>(region->get_terminator_inst());
    Expr             *cond_expr = translate_expr(branch->getCondition());

    if (region->isNegated()) {
        cond_expr = Base::create_unary(
            cond_expr->get_location(), Expr_unary::OK_LOGICAL_NOT, cond_expr);
    }

    Stmt *then_stmt = translate_region(region->getThen());

    Stmt *if_stmt = Base::m_stmt_factory.create_if(
        cond_expr->get_location(), cond_expr, then_stmt, /*else_stmt=*/nullptr);

    return join_statements(head_stmt, if_stmt);
}

// Translate a if-then-else-region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_if_then_else(
    llvm::sl::RegionIfThenElse const *region)
{
    llvm::sl::Region *head      = region->getHead();
    Stmt             *head_stmt = translate_region(head);

    llvm::BranchInst *branch    = llvm::cast<llvm::BranchInst>(region->get_terminator_inst());
    Expr             *cond_expr = translate_expr(branch->getCondition());

    if (region->isNegated()) {
        MDL_ASSERT(!"if-then-else regions should not use negated");
        cond_expr = Base::create_unary(
            cond_expr->get_location(), Expr_unary::OK_LOGICAL_NOT, cond_expr);
    }

    Stmt *then_stmt = translate_region(region->getThen());
    Stmt *else_stmt = translate_region(region->getElse());

    Stmt *if_stmt = Base::m_stmt_factory.create_if(
        cond_expr->get_location(), cond_expr, then_stmt, else_stmt);

    return join_statements(head_stmt, if_stmt);
}

// Translate a break-region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_break(
    llvm::sl::RegionBreak const *region)
{
    return Base::m_stmt_factory.create_break(Base::zero_loc);
}

// Translate a continue-region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_continue(
    llvm::sl::RegionContinue const *region)
{
    return Base::m_stmt_factory.create_continue(Base::zero_loc);
}

// Translate a return-region into an AST.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_return(
    llvm::sl::RegionReturn const *region)
{
    llvm::sl::Region *head      = region->getHead();
    Stmt             *head_stmt = translate_region(head);

    llvm::ReturnInst *ret_inst = region->get_return_inst();
    llvm::Value      *ret_val  = ret_inst->getReturnValue();

    Expr *sl_ret_expr = nullptr;
    if (ret_val != nullptr) {
        if (!is<Type_void>(Base::convert_type(ret_val->getType()))) {
            sl_ret_expr = translate_expr(ret_val);
        }
    }

    Stmt *ret_stmt = nullptr;
    if (m_out_def != nullptr && sl_ret_expr != nullptr) {
        // return through an out parameter
        Stmt *expr_stmt = create_assign_stmt(m_out_def, sl_ret_expr);
        ret_stmt = Base::m_stmt_factory.create_return(
            Base::convert_location(ret_inst), /*expr=*/nullptr);
        ret_stmt = join_statements(expr_stmt, ret_stmt);
    } else {
        // normal return
        ret_stmt = Base::m_stmt_factory.create_return(
            Base::convert_location(ret_inst), sl_ret_expr);
    }

    return join_statements(head_stmt, ret_stmt);
}

// Return the base pointer of the given pointer after skipping all bitcasts and GEPs.
template<typename BasePass>
llvm::Value *SLWriterPass<BasePass>::get_base_pointer(llvm::Value *pointer)
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
template<typename BasePass>
llvm::Value *SLWriterPass<BasePass>::process_pointer_address_recurse(
    Type_walk_stack &stack,
    llvm::Value     *pointer,
    uint64_t        write_size,
    bool            allow_early_out)
{
    llvm::Value *base_pointer;

    // skip bitcasts
    llvm::Value *orig_pointer = pointer;
    pointer = pointer->stripPointerCasts();

    if (llvm::GEPOperator *gep = llvm::dyn_cast<llvm::GEPOperator>(pointer)) {
        // process pointer of GEP without early out, as the GEP will modify the pointer
        base_pointer = process_pointer_address_recurse(
            stack, gep->getPointerOperand(), write_size, /*allow_early_out=*/ false);
        if (base_pointer == nullptr) {
            return nullptr;
        }

        llvm::Type *cur_llvm_type = stack.back().field_type;
        unsigned num_indices = gep->getNumIndices();

        // handle non-zero first index
        // example:
        //    %.repack = getelementptr %struct.BSDF_eval_data, %struct.BSDF_eval_data* %sret_ptr,
        //               i64 0, i32 4, i32 0
        //    %5 = getelementptr inbounds float, float* %.repack, i64 1
        if (num_indices > 0) {
            if (llvm::ConstantInt *idx = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(1))) {
                if (!idx->isZero()) {
                    // We currently don't support bitcasts between GEPs, like
                    //   %.repack = getelementptr %struct.BSDF_eval_data,
                    //              %struct.BSDF_eva_data* %sret_ptr, i64 0, i32 4, i32 0
                    //   %5 = bitcast float* %.repack to i8*
                    //   %6 = getelementptr inbounds i8, i8* %5, i64 4
                    if (pointer != orig_pointer) {
                        MDL_ASSERT(!"Cannot move to next element when GEP was casted");
                        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                        return nullptr;
                    }

                    // Move to the idx'th next element
                    unsigned idx_imm = unsigned(idx->getZExtValue());
                    for (unsigned i = 0; i < idx_imm; ++i) {
                        if (!move_to_next_compound_elem(stack, write_size)) {
                            MDL_ASSERT(!"Failed to move to idx'th next element");
                            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                            return nullptr;
                        }
                    }
                }
            } else {
                MDL_ASSERT(!"First GEP index is not a constant");
                Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                return nullptr;
            }
        }

        for (unsigned i = 1; i < num_indices; ++i) {
            // check whether we can stop going deeper into the compound
            if (allow_early_out) {
                uint64_t cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);
                if (cur_type_size <= write_size) {
                    // we can only stop if all of the remaining indices are zero
                    bool all_zero = true;
                    for (unsigned j = i; j < num_indices; ++j) {
                        if (llvm::ConstantInt *idx =
                            llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(j + 1)))
                        {
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
                    if (all_zero) {
                        break;
                    }
                }
            }

            llvm::Value *idx_val = gep->getOperand(i + 1);
            if (llvm::ConstantInt *idx = llvm::dyn_cast<llvm::ConstantInt>(idx_val)) {
                unsigned idx_imm = unsigned(idx->getZExtValue());
                llvm::Type *field_type =
                    llvm::GetElementPtrInst::getTypeAtIndex(cur_llvm_type, idx_imm);
                stack.push_back(Type_walk_element(cur_llvm_type, i, idx_val, 0, field_type));
                cur_llvm_type = field_type;
            } else {
                if (!llvm::isa<llvm::ArrayType>(cur_llvm_type) &&
                    !llvm::isa<llvm::VectorType>(cur_llvm_type))
                {
                    stack.clear();
                    Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                    return nullptr;
                }
                llvm::Type *field_type =
                    llvm::GetElementPtrInst::getTypeAtIndex(cur_llvm_type, (uint64_t)0);
                stack.push_back(Type_walk_element(
                    cur_llvm_type, i, idx_val, 0, field_type));
                cur_llvm_type = field_type;
            }
        }
    } else {
        // should be alloca, pointer parameter or global
        base_pointer = pointer;

        llvm::Type *compound_type = pointer->getType()->getPointerElementType();
        stack.push_back(Type_walk_element(compound_type, 0, nullptr, 0, compound_type));
    }

    return base_pointer;
}

// Initialize the type walk stack for the given pointer and the size to be written
// and return the base pointer.
template<typename BasePass>
llvm::Value *SLWriterPass<BasePass>::process_pointer_address(
    Type_walk_stack &stack,
    llvm::Value *pointer,
    uint64_t write_size)
{
    llvm::Value *base_pointer = process_pointer_address_recurse(stack, pointer, write_size);
    if (base_pointer == nullptr) {
        return nullptr;
    }

    // move deeper into compound elements, if necessary
    move_into_compound_elem(stack, write_size);

    return base_pointer;
}

template<typename BasePass>
bool SLWriterPass<BasePass>::move_into_compound_elem(
    Type_walk_stack &stack,
    size_t          target_size)
{
    llvm::Type *cur_llvm_type = stack.back().field_type;
    uint64_t cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);

    // still too big? check whether we can go deeper
    // examples:
    //   - %10 = bitcast [16 x <4 x float>]* %0 to i32*
    //   - writing 16 bytes from bsdf_diffuse.y to bsdf_glossy.y, we need to go from
    //     bsdf_glossy down to bsdf_glossy.x when there are still 8 bytes left to write
    while (cur_type_size >= target_size) {
        // check for array, vector or struct
        if (!llvm::isa<llvm::ArrayType>(cur_llvm_type) &&
                !llvm::isa<llvm::FixedVectorType>(cur_llvm_type) &&
                !llvm::isa<llvm::StructType>(cur_llvm_type)) {
            return cur_type_size == target_size;
        }

        llvm::Type *first_type = cur_llvm_type->getContainedType(0);
        uint64_t first_type_size = m_cur_data_layout->getTypeStoreSize(first_type);
        if (cur_type_size == target_size && first_type_size < target_size) {
            // with target size already found, don't go deeper, if this isn't the right size anymore
            break;
        }
        stack.push_back(Type_walk_element(cur_llvm_type, ~0, nullptr, 0, first_type));
        cur_llvm_type = first_type;
        cur_type_size = first_type_size;
    }

    return true;
}

// Get the definition for a compound type field.
template<typename BasePass>
typename SLWriterPass<BasePass>::Definition *SLWriterPass<BasePass>::get_field_definition(
    Type   *type,
    Symbol *sym)
{
    if (Scope *scope = Base::m_def_tab.get_type_scope(type)) {
        return scope->find_definition_in_scope(sym);
    }
    return nullptr;
}

// Create an lvalue expression for the compound element given by the type walk stack.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_compound_elem_expr(
    Type_walk_stack &stack,
    llvm::Value     *base_pointer)
{
    Expr *cur_expr = translate_expr(base_pointer);
    Type *cur_type = cur_expr->get_type()->skip_type_alias();

    // start after base element
    for (size_t i = 1, n = stack.size(); i < n; ++i) {
        typename Type::Kind type_kind = cur_type->get_kind();
        if (type_kind == Type::TK_ARRAY ||
            type_kind == Type::TK_VECTOR ||
            type_kind == Type::TK_MATRIX)
        {
            if (type_kind == Type::TK_VECTOR &&
                    (stack[i].field_index_val == nullptr ||
                        llvm::isa<llvm::ConstantInt>(stack[i].field_index_val))) {
                unsigned idx_imm;
                if (stack[i].field_index_val) {
                    llvm::ConstantInt *idx =
                        llvm::cast<llvm::ConstantInt>(stack[i].field_index_val);
                    idx_imm = unsigned(idx->getZExtValue()) + stack[i].field_index_offs;
                } else {
                    idx_imm = stack[i].field_index_offs;
                }
                cur_expr = create_vector_access(cur_expr, idx_imm);
            } else {
                Expr *array_index;
                if (stack[i].field_index_val) {
                    array_index = translate_expr(stack[i].field_index_val);
                    if (stack[i].field_index_offs != 0) {
                        Expr *offs = Base::m_expr_factory.create_literal(
                            Base::zero_loc,
                            Base::m_value_factory.get_int32(int32_t(stack[i].field_index_offs)));
                        array_index = Base::create_binary(
                            Expr_binary::OK_PLUS,
                            array_index,
                            offs);
                    }
                } else {
                    array_index = Base::m_expr_factory.create_literal(
                        Base::zero_loc,
                        Base::m_value_factory.get_int32(int32_t(stack[i].field_index_offs)));
                }
                cur_expr = Base::create_binary(
                    Expr_binary::OK_ARRAY_SUBSCRIPT, cur_expr, array_index);
            }

            cur_type = cast<Type_compound>(cur_type)->get_compound_type(0);
            cur_type = cur_type->skip_type_alias();
            cur_expr->set_type(cur_type);
            continue;
        }

        if (Type_struct *struct_type = as<Type_struct>(cur_type)) {
            unsigned idx_imm = stack[i].field_index_offs;
            if (stack[i].field_index_val) {
                llvm::ConstantInt *idx = llvm::cast<llvm::ConstantInt>(stack[i].field_index_val);
                idx_imm += unsigned(idx->getZExtValue());
            }
            typename Type_struct::Field *field = struct_type->get_field(idx_imm);
            Definition                  *f_def = get_field_definition(
                struct_type, field->get_symbol());

            Expr *field_ref = f_def != nullptr ?
                Base::create_reference(f_def) :
                Base::create_reference(
                    field->get_symbol(), field->get_type());
            cur_expr = Base::create_binary(
                Expr_binary::OK_SELECT, cur_expr, field_ref);

            cur_type = field->get_type()->skip_type_alias();
            cur_expr->set_type(cur_type);
            continue;
        }

        MDL_ASSERT(!"unexpected type kind");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    return cur_expr;
}

// Return true, if the index is valid for the given composite type.
template<typename BasePass>
bool SLWriterPass<BasePass>::is_valid_composite_index(
    llvm::Type *type,
    size_t     index)
{
    if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(type)) {
        return index < st->getNumElements();
    }
    if (llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        return index < vt->getNumElements();
    }
    if (llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(type)) {
        return index < at->getNumElements();
    }
    MDL_ASSERT(!"unexpected composite type");
    return false;
}

/// Go to the next element in the stack and into the element until element size is not
/// too big anymore.
template<typename BasePass>
bool SLWriterPass<BasePass>::move_to_next_compound_elem(Type_walk_stack &stack, size_t target_size)
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
            ++cur_elem->field_index_offs;
            cur_elem->field_type = llvm::GetElementPtrInst::getTypeAtIndex(
                cur_elem->llvm_type, index + 1);
            return move_into_compound_elem(stack, target_size);
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
template<typename BasePass>
template<unsigned N>
void SLWriterPass<BasePass>::create_zero_init(
    llvm::SmallVector<Stmt *, N> &stmts,
    Expr                         *lval_expr)
{
    Type *dest_type = lval_expr->get_type();
    if (Type_array *arr_type = as<Type_array>(dest_type)) {
        Type *elem_type = arr_type->get_element_type();
        for (size_t i = 0, n = arr_type->get_size(); i < n; ++i) {
            // as arrays of arrays are not allowed in MDL, we can use the "cast 0" approach
            Expr *res = Base::create_type_cast(
                elem_type,
                Base::m_expr_factory.create_literal(
                    Base::zero_loc,
                    Base::m_value_factory.get_int32(0)));

            Expr *array_index = Base::m_expr_factory.create_literal(
                Base::zero_loc,
                Base::m_value_factory.get_int32(int32_t(i)));
            Expr *array_elem = Base::create_binary(
                Expr_binary::OK_ARRAY_SUBSCRIPT, lval_expr, array_index);
            array_elem->set_type(elem_type);

            stmts.push_back(create_assign_stmt(array_elem, res));
        }
    } else {
        // implement as casting 0 to the requested type
        Expr *res = Base::create_type_cast(
            dest_type,
            Base::m_expr_factory.create_literal(
                Base::zero_loc,
                Base::m_value_factory.get_int32(0)));

        stmts.push_back(create_assign_stmt(lval_expr, res));
    }
}

// Translate a call into one or more statements, if it is a supported intrinsic call.
template<typename BasePass>
template <unsigned N>
bool SLWriterPass<BasePass>::translate_intrinsic_call(
    llvm::SmallVector<Stmt *, N> &stmts,
    llvm::CallInst               *call)
{
    llvm::Function *called_func = call->getCalledFunction();
    if (called_func == nullptr) {
        return false;
    }

    // we need to handle memset and memcpy intrinsics
    switch (called_func->getIntrinsicID()) {
    case llvm::Intrinsic::memset:
        {
            // expect the destination pointer to be a bitcast
            llvm::BitCastInst *bitcast = llvm::dyn_cast<llvm::BitCastInst>(call->getOperand(0));
            if (bitcast == nullptr) {
                return false;
            }

            // only handle as zero memory
            llvm::ConstantInt *fill_val = llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(1));
            if (fill_val == nullptr || !fill_val->isZero()) {
                return false;
            }

            // only allow constant size
            llvm::ConstantInt *size_val = llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2));
            if (size_val == nullptr) {
                return false;
            }
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
            if (base_pointer == nullptr) {
                return false;
            }

            uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
            while (write_size > 0) {
                Expr *lval_expr = create_compound_elem_expr(stack, base_pointer);
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
                if (write_size <= 0) {
                    break;
                }

                // go to next element
                if (!move_to_next_compound_elem(stack, write_size)) {
                    return false;
                }
            }
            return true;
        }
    case llvm::Intrinsic::memcpy:
        {
            // expect the destination pointer to be a bitcast
            llvm::BitCastInst *bitcast_dst = llvm::dyn_cast<llvm::BitCastInst>(call->getOperand(0));
            if (bitcast_dst == nullptr) {
                return false;
            }

            // expect the source pointer to be a bitcast (may be a bitcast instruction or value
            // (in case of a constant))
            llvm::Operator *bitcast_src = llvm::dyn_cast<llvm::Operator>(call->getOperand(1));
            if (bitcast_src->getOpcode() != llvm::Instruction::BitCast) {
                return false;
            }

            // only allow constant size
            llvm::ConstantInt *size_val = llvm::dyn_cast<llvm::ConstantInt>(call->getOperand(2));
            if (size_val == nullptr) {
                return false;
            }
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
            if (base_pointer_dest == nullptr || base_pointer_src == nullptr) {
                return false;
            }

            uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
            while (write_size > 0) {
                Expr *lval_expr = create_compound_elem_expr(stack_dest, base_pointer_dest);
                Expr *rval_expr = create_compound_elem_expr(stack_src, base_pointer_src);
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
                if (write_size <= 0) {
                    break;
                }

                // go to next element in both stacks
                if (!move_to_next_compound_elem(stack_dest, write_size)) {
                    return false;
                }
                if (!move_to_next_compound_elem(stack_src, write_size)) {
                    return false;
                }
            }
            return true;
        }
    default:
        return false;
    }
}

// Translate a struct select into an if statement writing to the given variable.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::translate_struct_select(
    llvm::SelectInst *select,
    Def_variable     *dst_var)
{
    Expr *cond       = translate_expr(select->getCondition());
    Expr *true_expr  = translate_expr(select->getTrueValue());
    Expr *false_expr = translate_expr(select->getFalseValue());

    Location loc = Base::convert_location(select);
    Stmt *res = Base::m_stmt_factory.create_if(
        loc,
        cond,
        create_assign_stmt(dst_var, true_expr),
        create_assign_stmt(dst_var, false_expr));
    return res;
}

// Bitcast the given SL value to the given SL type and return an AST expression for it.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::bitcast_to(
    Expr *val,
    Type *dest_type)
{
    Type *src_type = val->get_type()->skip_type_alias();
    dest_type = dest_type->skip_type_alias();

    // first convert bool to int, if necessary
    if (is<Type_bool>(src_type) && is<Type_float>(dest_type)) {
        val = Base::create_type_cast(Base::m_tc.int_type, val);
        src_type = Base::m_tc.int_type;
    }

    if (is<Type_int>(src_type) && is<Type_float>(dest_type)) {
        return Base::create_int2float_bitcast(Base::m_tc.float_type, val);
    }

    if (is<Type_float>(src_type) && is<Type_int>(dest_type)) {
        return Base::create_float2int_bitcast(Base::m_tc.int_type, val);
    }

    return nullptr;
}

// Bitcast the given LLVM value to the given LLVM type and return an AST expression for it.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::bitcast_to(
    llvm::Value *val,
    llvm::Type  *dest_type)
{
    llvm::Type *src_type = val->getType();

    if ((src_type->isIntegerTy() && dest_type->isFloatingPointTy()) ||
            (src_type->isFloatingPointTy() && dest_type->isIntegerTy())) {
        Expr *sl_val = translate_expr(val);
        Type *sl_dest_type = Base::convert_type(dest_type);
        return bitcast_to(sl_val, sl_dest_type);
    }

    // check whether src and dest type are mapped to the same SL type
    if (Base::convert_type(src_type) == Base::convert_type(dest_type)) {
        // directly return the value without any cast
        return translate_expr(val);
    }

    return nullptr;
}

// Translate an LLVM store instruction into one or more AST statements.
template<typename BasePass>
template <unsigned N>
void SLWriterPass<BasePass>::translate_store(
    llvm::SmallVector<Stmt *, N> &stmts,
    llvm::StoreInst              *inst)
{
    llvm::Value *value   = inst->getValueOperand();
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
        // no bitcasts, so we can do a direct assignment
        Expr *lvalue = translate_lval_expression(pointer);
        stmts.push_back(create_assign_stmt(lvalue, translate_expr(value)));
        return;
    }

    Type_walk_stack stack;
    llvm::Value *base_pointer = process_pointer_address(stack, pointer, target_size);
    if (base_pointer == nullptr) {
        return;
    }

    // handle cases like:
    //   - bitcast float* %0 to i32*
    //   - bitcast [2 x float]* %weights.i.i.i to i32*
    //   - bitcast [16 x <4 x float>]* %0 to i32*
    if (m_cur_data_layout->getTypeStoreSize(stack.back().field_type) == target_size) {
        llvm::Type *src_type = stack.back().field_type;

        if (Expr *rvalue = bitcast_to(value, src_type)) {
            Expr *lvalue = create_compound_elem_expr(stack, base_pointer);
            stmts.push_back(create_assign_stmt(lvalue, rvalue));
            return;
        }
    }

    Expr  *expr = nullptr;
    Type  *expr_elem_type = nullptr;

    if (llvm::isa<llvm::VectorType>(value->getType())) {
        expr = translate_expr(value);
        if (Type_vector *vt = as<Type_vector>(expr->get_type())) {
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
        Expr *lval = create_compound_elem_expr(stack, base_pointer);
        if (Type_vector *lval_vt = as<Type_vector>(lval->get_type())) {
            for (size_t i = 0, n = lval_vt->get_size(); i < n; ++i) {
                Expr *lhs = create_vector_access(lval, i);
                Expr *rhs;
                if (expr == nullptr) {
                    // zero initialization
                    rhs = Base::m_expr_factory.create_literal(
                        Base::convert_location(inst),
                        Base::m_value_factory.get_zero_initializer(lhs->get_type()));
                } else {
                    rhs = create_vector_access(expr, cur_vector_index++);
                    expr = translate_expr(value);   // create new AST for the expression
                }
                stmts.push_back(create_assign_stmt(lhs, rhs));
            }
        } else {
            Expr *rhs;
            if (expr == nullptr) {
                // zero initialization
                rhs = Base::m_expr_factory.create_literal(
                    Base::convert_location(inst),
                    Base::m_value_factory.get_zero_initializer(lval->get_type()));
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
        if (target_size <= 0) {
            break;
        }

        // go to next element
        if (!move_to_next_compound_elem(stack, target_size)) {
            MDL_ASSERT(!"moving to next element failed");
            return;
        }
    }
}

// Translate an LLVM ConstantInt value to an AST value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant_int(
    llvm::ConstantInt *ci)
{
    unsigned int bit_width = ci->getBitWidth();
    if (bit_width > 1 && bit_width < 16) {
        // allow comparison of "trunc i32 %Y to i2" with "i2 -1"
        // TODO: sign
        return Base::m_value_factory.get_int16(
            int16_t(ci->getSExtValue()) & ((1 << bit_width) - 1));
    }

    if (bit_width > 16 && bit_width < 32) {
        // allow comparison of "trunc i32 %Y to i2" with "i2 -1"
        // TODO: sign
        return Base::m_value_factory.get_int32(
            int32_t(ci->getSExtValue()) & ((1 << bit_width) - 1));
    }

    switch (bit_width) {
    case 1:
        return Base::m_value_factory.get_bool(!ci->isZero());
    case 16:
        // TODO: sign
        return Base::m_value_factory.get_int16(int16_t(ci->getSExtValue()));
    case 32:
        // TODO: sign
        return Base::m_value_factory.get_int32(int32_t(ci->getSExtValue()));
    case 64:  // TODO: always treat as 32-bit, maybe not a good idea
        return Base::m_value_factory.get_int32(int32_t(ci->getZExtValue()));
    }
    MDL_ASSERT(!"unexpected LLVM integer type");
    return Base::m_value_factory.get_bad();
}

// Translate an LLVM ConstantFP value to an AST Expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant_fp(
    llvm::ConstantFP *cf)
{
    llvm::APFloat const &apfloat = cf->getValueAPF();

    switch (cf->getType()->getTypeID()) {
    case llvm::Type::HalfTyID:
        return Base::m_value_factory.get_half(apfloat.convertToFloat());
    case llvm::Type::FloatTyID:
        return Base::m_value_factory.get_float(apfloat.convertToFloat());
    case llvm::Type::DoubleTyID:
        if (Base::has_double_type()) {
            return Base::m_value_factory.get_double(apfloat.convertToDouble());
        } else {
            return Base::m_value_factory.get_float(float(apfloat.convertToDouble()));
        }
    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        MDL_ASSERT(!"unexpected float literal type");
        // fallthrough
    default:
        break;
    }
    MDL_ASSERT(!"invalid float literal type");
    return Base::m_value_factory.get_bad();
}

// Translate an LLVM ConstantDataVector value to an AST Value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant_data_vector(
    llvm::ConstantDataVector *cv)
{
    size_t num_elems = size_t(cv->getNumElements());
    Type_vector *sl_type;
    Small_VLA<Value_scalar *, 8> values(Base::m_alloc, num_elems);

    switch (cv->getElementType()->getTypeID()) {
    case llvm::Type::IntegerTyID:
        {
            llvm::IntegerType *int_type = llvm::cast<llvm::IntegerType>(cv->getElementType());
            switch (int_type->getBitWidth()) {
            case 1:
            case 8:  // TODO: maybe not bool but really 8-bit
                sl_type = Base::m_tc.get_vector(Base::m_tc.bool_type, num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = Base::m_value_factory.get_bool(cv->getElementAsInteger(i) != 0);
                }
                break;
            case 16:  // always treat as 32-bit
            case 32:
                // TODO: sign
                sl_type = Base::m_tc.get_vector(Base::m_tc.int_type, num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = Base::m_value_factory.get_int32(
                        int32_t(cv->getElementAsAPInt(i).getSExtValue()));
                }
                break;
            case 64:  // always cast to signed 32-bit
                sl_type = Base::m_tc.get_vector(Base::m_tc.int_type, num_elems);
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = Base::m_value_factory.get_int32(
                        int32_t(cv->getElementAsAPInt(i).getZExtValue()));
                }
                break;
            default:
                MDL_ASSERT(!"invalid integer vector literal type");
                return Base::m_value_factory.get_bad();
            }
            break;
        }

    case llvm::Type::HalfTyID:
        sl_type = Base::m_tc.get_vector(Base::m_tc.half_type, num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = Base::m_value_factory.get_half(cv->getElementAsFloat(i));
        }
        break;
    case llvm::Type::FloatTyID:
        sl_type = Base::m_tc.get_vector(Base::m_tc.float_type, num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = Base::m_value_factory.get_float(cv->getElementAsFloat(i));
        }
        break;

    case llvm::Type::DoubleTyID:
        if (Base::has_double_type()) {
            sl_type = Base::m_tc.get_vector(Base::m_tc.double_type, num_elems);
            for (size_t i = 0; i < num_elems; ++i) {
                values[i] = Base::m_value_factory.get_double(cv->getElementAsDouble(i));
            }
        } else {
            sl_type = Base::m_tc.get_vector(Base::m_tc.float_type, num_elems);
            for (size_t i = 0; i < num_elems; ++i) {
                values[i] = Base::m_value_factory.get_float(float(cv->getElementAsDouble(i)));
            }
        }
        break;

    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        // fallthrough
    default:
        MDL_ASSERT(!"invalid vector literal type");
        return Base::m_value_factory.get_bad();
    }

    return Base::m_value_factory.get_vector(sl_type, values);
}

// Translate an LLVM ConstantDataArray value to an AST compound initializer.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_constant_data_array(
    llvm::ConstantDataArray *cv)
{
    size_t num_elems = size_t(cv->getNumElements());
    Type_array *sl_type;
    Small_VLA<Value *, 8> values(Base::m_alloc, num_elems);

    switch (cv->getElementType()->getTypeID()) {
    case llvm::Type::IntegerTyID:
        {
            llvm::IntegerType *int_type = llvm::cast<llvm::IntegerType>(cv->getElementType());
            switch (int_type->getBitWidth()) {
            case 1:
            case 8:  // TODO: maybe not bool but really 8-bit
                sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.bool_type, num_elems));
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = Base::m_value_factory.get_bool(cv->getElementAsInteger(i) != 0);
                }
                break;
            case 16:  // always treat as 32-bit
            case 32:
                // TODO: sign
                sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.int_type, num_elems));
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = Base::m_value_factory.get_int32(
                            int32_t(cv->getElementAsAPInt(i).getSExtValue()));
                }
                break;
            case 64:  // always cast to signed 32-bit
                sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.int_type, num_elems));
                for (size_t i = 0; i < num_elems; ++i) {
                    values[i] = Base::m_value_factory.get_int32(
                            int32_t(cv->getElementAsAPInt(i).getZExtValue()));
                }
                break;
            default:
                MDL_ASSERT(!"invalid integer vector literal type");
                Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                return Base::m_expr_factory.create_invalid(Base::zero_loc);
            }
            break;
        }

    case llvm::Type::HalfTyID:
        sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.half_type, num_elems));
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = Base::m_value_factory.get_half(cv->getElementAsFloat(i));
        }
        break;
    case llvm::Type::FloatTyID:
        sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.float_type, num_elems));
        for (size_t i = 0; i < num_elems; ++i) {
            values[i] = Base::m_value_factory.get_float(cv->getElementAsFloat(i));
        }
        break;

    case llvm::Type::DoubleTyID:
        if (Base::has_double_type()) {
            sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.double_type, num_elems));
            for (size_t i = 0; i < num_elems; ++i) {
                values[i] = Base::m_value_factory.get_double(cv->getElementAsDouble(i));
            }
        } else {
            sl_type = cast<Type_array>(Base::m_tc.get_array(Base::m_tc.float_type, num_elems));
            for (size_t i = 0; i < num_elems; ++i) {
                values[i] = Base::m_value_factory.get_float(float(cv->getElementAsDouble(i)));
            }
        }
        break;

    case llvm::Type::X86_FP80TyID:
    case llvm::Type::FP128TyID:
    case llvm::Type::PPC_FP128TyID:
        // fallthrough
    default:
        MDL_ASSERT(!"invalid vector literal type");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    Value *arr_value = Base::m_value_factory.get_array(sl_type, values);
    return Base::m_expr_factory.create_literal(Base::zero_loc, arr_value);
}

// Translate an LLVM ConstantStruct value to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_constant_struct_expr(
    llvm::ConstantStruct *cv,
    bool                 is_global)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Small_VLA<Expr *, 8> agg_elems(Base::m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        agg_elems[i] = translate_constant_expr(elem, is_global);
    }

    Type *res_type = Base::convert_type(cv->getType());
    return create_constructor_call(res_type, agg_elems, Base::zero_loc, is_global);
}

// Translate an LLVM ConstantVector value to an AST Value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant_vector(
    llvm::ConstantVector *cv)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Small_VLA<Value_scalar *, 8> values(Base::m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        values[i] = cast<Value_scalar>(translate_constant(elem));
    }

    return Base::m_value_factory.get_vector(
        cast<Type_vector>(Base::convert_type(cv->getType())), values);
}

// Translate an LLVM ConstantArray value to an AST compound expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_constant_array(
    llvm::ConstantArray *cv,
    bool                is_global)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Small_VLA<Expr *, 8> values(Base::m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        values[i] = translate_constant_expr(elem, is_global);
    }

    Type_array *arr_type = cast<Type_array>(Base::convert_type(cv->getType()));
    return Base::create_initializer(Base::zero_loc, arr_type, values);
}

// Translate an LLVM ConstantAggregateZero value to an AST compound expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_constant_array(
    llvm::ConstantAggregateZero *cv,
    bool                        is_global)
{
    llvm::ArrayType *at = llvm::cast<llvm::ArrayType>(cv->getType());
    size_t num_elems = size_t(at->getArrayNumElements());
    Small_VLA<Expr *, 8> agg_elems(Base::m_alloc, num_elems);

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getElementValue(i);
        agg_elems[i] = translate_constant_expr(elem, is_global);
    }

    Type_array *arr_type = cast<Type_array>(Base::convert_type(cv->getType()));
    return Base::create_initializer(Base::zero_loc, arr_type, agg_elems);
}

// Translate an LLVM ConstantArray value to an AST matrix value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant_matrix(
    llvm::ConstantArray *cv)
{
    size_t num_elems = size_t(cv->getNumOperands());
    Value_vector *values[4];

    Type_matrix *mt = cast<Type_matrix>(Base::convert_type(cv->getType()));

    for (size_t i = 0; i < num_elems; ++i) {
        llvm::Constant *elem = cv->getOperand(i);
        values[i] = cast<Value_vector>(translate_constant(elem));
    }
    return Base::m_value_factory.get_matrix(mt, Array_ref<Value_vector *>(values, num_elems));
}

// Translate an LLVM ConstantAggregateZero value to an AST Value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant_aggregate_zero(
    llvm::ConstantAggregateZero *cv)
{
    Type *type = Base::convert_type(cv->getType());
    return Base::m_value_factory.get_zero_initializer(type);
}

// Translate an LLVM Constant value to an AST Value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Value *SLWriterPass<BasePass>::translate_constant(
    llvm::Constant *c)
{
    if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(c)) {
        return translate_constant_vector(cv);
    }
    if (llvm::isa<llvm::UndefValue>(c)) {
        Type *type = Base::convert_type(c->getType());

#if 0
        // use special undef values for debugging
        if (is<Type_float>(type)) {
            return Base::m_value_factory.get_float(-107374176.0f);
        } else if(is<Type_double>(type)) {
            return Base::m_value_factory.get_double(-9.25596313493e+61);
        } else if (is<Type_int>(type)) {
            return Base::m_value_factory.get_int32(0xcccccccc);
        } else if (is<Type_uint>(type)) {
            return Base::m_value_factory.get_uint32(0xcccccccc);
        }
#endif

        return Base::m_value_factory.get_zero_initializer(type);
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
        if (Base::is_matrix_type(at)) {
            return translate_constant_matrix(cv);
        }
    }
    MDL_ASSERT(!"Unsupported Constant kind");
    return Base::m_value_factory.get_bad();
}

// Translate an LLVM Constant value to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_constant_expr(
    llvm::Constant *c,
    bool           is_global)
{
    if (llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(c)) {
        if (gv->isConstant() && gv->hasInitializer()) {
            MDL_ASSERT(!gv->isExternallyInitialized());
            c = gv->getInitializer();

            Definition *c_def = create_global_const(c);
            return Base::create_reference(c_def);
        }
    }

    if (llvm::ConstantStruct *cv = llvm::dyn_cast<llvm::ConstantStruct>(c)) {
        return translate_constant_struct_expr(cv, is_global);
    }

    if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(c->getType())) {
        if (llvm::ConstantAggregateZero *cv = llvm::dyn_cast<llvm::ConstantAggregateZero>(c)) {
            size_t num_elems = size_t(cv->getNumElements());
            Small_VLA<Expr *, 8> agg_elems(Base::m_alloc, num_elems);

            for (size_t i = 0; i < num_elems; ++i) {
                llvm::Constant *elem = cv->getElementValue(i);
                agg_elems[i] = translate_constant_expr(elem, is_global);
            }

            Type *res_type = Base::convert_type(st);
            return create_constructor_call(res_type, agg_elems, Base::zero_loc, is_global);
        }

        if (llvm::UndefValue *undef = llvm::dyn_cast<llvm::UndefValue>(c)) {
            size_t num_elems = size_t(undef->getNumElements());
            Small_VLA<Expr *, 8> agg_elems(Base::m_alloc, num_elems);

            for (size_t i = 0; i < num_elems; ++i) {
                llvm::Constant *elem = undef->getElementValue(i);
                agg_elems[i] = translate_constant_expr(elem, is_global);
            }

            Type *res_type = Base::convert_type(st);
            return create_constructor_call(res_type, agg_elems, Base::zero_loc, is_global);
        }
    }

    if (llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(c->getType())) {
        if (!Base::is_matrix_type(at)) {
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

    return Base::m_expr_factory.create_literal(Base::zero_loc, translate_constant(c));
}

// Translate an LLVM value to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr(
    llvm::Value *value)
{
    // check whether a local variable was generated for this instruction
    auto it = m_local_var_map.find(value);
    if (it != m_local_var_map.end()) {
        return Base::create_reference(it->second);
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

        Expr *result = nullptr;

        switch (inst->getOpcode()) {
        case llvm::Instruction::Alloca:
            {
                Def_variable *var_def = create_local_var(inst);
                result = Base::create_reference(var_def);
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
#if 0
                MDL_ASSERT(gep->hasAllZeroIndices());

                // treat as pointer cast, skip it
                return translate_expr(gep->getOperand(0));
#else
                // reuse lval expression code, which will create the necessary selects/array access
                return translate_lval_expression(gep);
#endif
            }


        case llvm::Instruction::PHI:
            {
                llvm::PHINode *phi = llvm::cast<llvm::PHINode>(inst);
                MDL_ASSERT(!"unexpected PHI node, a local variable should have been registered");

                Def_variable *phi_out_var = get_phi_out_var(phi);
                return Base::create_reference(phi_out_var);
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

        case llvm::Instruction::FNeg:
            {
                Expr *operand = translate_expr(inst->getOperand(0));
                result = Base::m_expr_factory.create_unary(
                    Base::convert_location(inst), Expr_unary::OK_NEGATIVE, operand);
                break;
            }

        case llvm::Instruction::Freeze:
            {
                // we ignore the freeze so far (which is completely valid)
                result = translate_expr(inst->getOperand(0));
                break;
            }

        case llvm::Instruction::CallBr:
        case llvm::Instruction::Fence:
        case llvm::Instruction::AtomicCmpXchg:
        case llvm::Instruction::AtomicRMW:
        case llvm::Instruction::CleanupPad:
        case llvm::Instruction::CatchPad:
        case llvm::Instruction::VAArg:
        case llvm::Instruction::LandingPad:
            MDL_ASSERT(!"unexpected LLVM instruction");
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            result = Base::m_expr_factory.create_invalid(Base::convert_location(inst));
            break;

        default:
            break;
        }

        if (result != nullptr) {
            return result;
        }
    }

    if (llvm::Constant *ci = llvm::dyn_cast<llvm::Constant>(value)) {
        // check, if the constant can be expressed as a SL constant expression, otherwise
        // create a SL constant declaration
        if (llvm::ArrayType *a_type = llvm::dyn_cast<llvm::ArrayType>(ci->getType())) {
            if (!Base::is_matrix_type(a_type)) {
                Def_variable *var_def = create_local_const(ci);
                return Base::create_reference(var_def);
            }
        }
        return translate_constant_expr(ci, /*is_global=*/ false);
    }
    MDL_ASSERT(!"unexpected LLVM value");
    Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
    return Base::m_expr_factory.create_invalid(Base::zero_loc);
}

// Translate a binary LLVM instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_bin(
    llvm::Instruction *inst)
{
    Expr *left  = translate_expr(inst->getOperand(0));
    Expr *right = translate_expr(inst->getOperand(1));

    typename Expr_binary::Operator sl_op;

    bool need_unsigned_operands = false;
    bool need_backcast          = false;

    switch (inst->getOpcode()) {
    // "Standard binary operators"
    case llvm::Instruction::Add:
    case llvm::Instruction::FAdd:
        sl_op = Expr_binary::OK_PLUS;
        break;
    case llvm::Instruction::Sub:
    case llvm::Instruction::FSub:
        sl_op = Expr_binary::OK_MINUS;
        break;
    case llvm::Instruction::Mul:
    case llvm::Instruction::FMul:
        sl_op = Expr_binary::OK_MULTIPLY;
        break;
    case llvm::Instruction::UDiv:
    case llvm::Instruction::SDiv:
    case llvm::Instruction::FDiv:
        sl_op = Expr_binary::OK_DIVIDE;
        break;
    case llvm::Instruction::URem:
    case llvm::Instruction::SRem:
    case llvm::Instruction::FRem:
        sl_op = Expr_binary::OK_MODULO;
        break;

    // "Logical operators (integer operands)"
    case llvm::Instruction::Shl:
        sl_op = Expr_binary::OK_SHIFT_LEFT;
        break;
    case llvm::Instruction::LShr:  // unsigned
        need_unsigned_operands = true;
        need_backcast = true;
        sl_op = Expr_binary::OK_SHIFT_RIGHT;
        break;
    case llvm::Instruction::AShr:  // signed
        sl_op = Expr_binary::OK_SHIFT_RIGHT;
        break;
    case llvm::Instruction::And:
        if (is<Type_bool>(left->get_type()->skip_type_alias()) &&
            is<Type_bool>(right->get_type()->skip_type_alias()))
        {
            // map bitwise AND on boolean values to logical and
            sl_op = Expr_binary::OK_LOGICAL_AND;
        } else {
            sl_op = Expr_binary::OK_BITWISE_AND;
        }
        break;
    case llvm::Instruction::Or:
        if (is<Type_bool>(left->get_type()->skip_type_alias()) &&
            is<Type_bool>(right->get_type()->skip_type_alias()))
        {
            // map bitwise OR on boolean values to logical OR
            sl_op = Expr_binary::OK_LOGICAL_OR;
        } else {
            sl_op = Expr_binary::OK_BITWISE_OR;
        }
        break;
    case llvm::Instruction::Xor:
        sl_op = Expr_binary::OK_BITWISE_XOR;
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
                sl_op = Expr_binary::OK_EQUAL;
                break;
            case llvm::CmpInst::FCMP_ONE:
            case llvm::CmpInst::FCMP_UNE:
            case llvm::CmpInst::ICMP_NE:
                sl_op = Expr_binary::OK_NOT_EQUAL;
                break;

            case llvm::CmpInst::FCMP_OGT:
            case llvm::CmpInst::FCMP_UGT:
            case llvm::CmpInst::ICMP_SGT:
                sl_op = Expr_binary::OK_GREATER;
                break;
            case llvm::CmpInst::ICMP_UGT:
                need_unsigned_operands = true;
                sl_op = Expr_binary::OK_GREATER;
                break;

            case llvm::CmpInst::FCMP_OGE:
            case llvm::CmpInst::FCMP_UGE:
            case llvm::CmpInst::ICMP_SGE:
                sl_op = Expr_binary::OK_GREATER_OR_EQUAL;
                break;
            case llvm::CmpInst::ICMP_UGE:
                need_unsigned_operands = true;
                sl_op = Expr_binary::OK_GREATER_OR_EQUAL;
                break;

            case llvm::CmpInst::FCMP_OLT:
            case llvm::CmpInst::FCMP_ULT:
            case llvm::CmpInst::ICMP_SLT:
                sl_op = Expr_binary::OK_LESS;
                break;
            case llvm::CmpInst::ICMP_ULT:
                need_unsigned_operands = true;
                sl_op = Expr_binary::OK_LESS;
                break;

            case llvm::CmpInst::FCMP_OLE:
            case llvm::CmpInst::FCMP_ULE:
            case llvm::CmpInst::ICMP_SLE:
                sl_op = Expr_binary::OK_LESS_OR_EQUAL;
                break;
            case llvm::CmpInst::ICMP_ULE:
                need_unsigned_operands = true;
                sl_op = Expr_binary::OK_LESS_OR_EQUAL;
                break;

            case llvm::CmpInst::FCMP_ORD:
            case llvm::CmpInst::FCMP_UNO:
                {
                    // FIXME: use specific isnan() functions here
                    Type *bool_type = Base::m_tc.bool_type;
                    Expr *isnan_ref = Base::create_reference(
                        Base::get_sym("isnan"), left->get_type());

                    Value *left_val = nullptr;
                    if (Expr_literal *left_lit = as<Expr_literal>(left)) {
                        left_val = left_lit->get_value();
                    }

                    Value *right_val = nullptr;
                    if (Expr_literal *right_lit = as<Expr_literal>(right)) {
                        right_val = right_lit->get_value();
                    }

                    Expr *res = nullptr;
                    if (left_val == nullptr) {
                        // compute left expression
                        Expr *left_isnan =
                            Base::m_expr_factory.create_call(isnan_ref, { left });
                        left_isnan->set_type(bool_type);

                        if (right_val == nullptr) {
                            // compute right expression
                            Expr *right_isnan =
                                Base::m_expr_factory.create_call(isnan_ref, { right });
                            right_isnan->set_type(bool_type);

                            res = Base::create_binary(
                                Expr_binary::OK_LOGICAL_OR,
                                left_isnan,
                                right_isnan);
                            res->set_type(bool_type);
                        } else {
                            // right expression is a constant
                            if (right_val->is_nan()) {
                                // we know, the result is true
                                res = Base::m_expr_factory.create_literal(
                                    Base::convert_location(inst),
                                    Base::m_value_factory.get_bool(true));
                            } else {
                                // the right argument does not matter
                                res = left_isnan;
                            }
                        }
                        return left_isnan;
                    } else if (right_val == nullptr) {
                        // compute right expression, left is a constant
                        Expr *right_isnan =
                            Base::m_expr_factory.create_call(isnan_ref, { right });
                        right_isnan->set_type(bool_type);

                        if (left_val->is_nan()) {
                            // we know, the result is true
                            res = Base::m_expr_factory.create_literal(
                                Base::convert_location(inst),
                                Base::m_value_factory.get_bool(true));
                        } else {
                            // the right argument does not matter
                            res = right_isnan;
                        }
                    } else {
                        // both expressions are constants
                        res = Base::m_expr_factory.create_literal(
                            Base::convert_location(inst),
                            Base::m_value_factory.get_bool(
                                left_val->is_nan() || right_val->is_nan()));
                    }

                    // negate the result if we ask for ordered
                    if (cmp->getPredicate() == llvm::CmpInst::FCMP_ORD) {
                        // == !is_nan(left) && !is_nan(right)
                        res = Base::create_unary(
                            Base::convert_location(inst), Expr_unary::OK_LOGICAL_NOT, res);
                        res->set_type(bool_type);
                    }
                    return res;
                }
            default:
                MDL_ASSERT(!"unexpected LLVM comparison predicate");
                Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                return Base::m_expr_factory.create_invalid(Base::convert_location(inst));
            }
            break;
        }

    default:
        MDL_ASSERT(!"unexpected LLVM binary instruction");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::convert_location(inst));
    }

    // LLVM does not have an unary minus, but SL has
    bool convert_to_unary_minus = false;
    if (sl_op == Expr_binary::OK_MINUS) {
        if (Expr_literal *l = as<Expr_literal>(left)) {
            Value *lv = l->get_value();

            if (lv->is_zero()) {
                if (Value_fp *f = as<Value_fp>(lv)) {
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

    Type *res_type = Base::convert_type(inst->getType());
    Expr *res      = nullptr;
    if (convert_to_unary_minus) {
        res = Base::create_unary(
            Base::convert_location(inst), Expr_unary::OK_NEGATIVE, right);
    } else {
        if (need_unsigned_operands) {
            Type *l_tp   = left->get_type()->skip_type_alias();
            Type *u_l_tp = Base::m_tc.to_unsigned_type(l_tp);

            if (u_l_tp != nullptr && u_l_tp != l_tp) {
                left = Base::create_type_cast(u_l_tp, left);
            }

            Type *r_tp = right->get_type()->skip_type_alias();
            Type *u_r_tp = Base::m_tc.to_unsigned_type(r_tp);

            if (u_r_tp != nullptr && u_r_tp != r_tp) {
                right = Base::create_type_cast(u_r_tp, right);
            }
        }
        res = Base::create_binary(sl_op, left, right);
        if (need_backcast) {
            res = Base::create_type_cast(res_type, res);
        }
    }
    res->set_type(res_type);

    return res;
}

// Find the first function parameter of the given type in the current function.
template<typename BasePass>
typename SLWriterPass<BasePass>::Definition *SLWriterPass<BasePass>::find_parameter_of_type(
    llvm::Type *t)
{
    for (llvm::Argument &arg_it : m_curr_func->args()) {
        llvm::Type *arg_llvm_type = arg_it.getType();

        if (arg_llvm_type == t) {
            return m_local_var_map[&arg_it];
        }
    }
    return nullptr;
}

// Translate an LLVM select instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_select(
    llvm::SelectInst *select)
{
    Expr *cond       = translate_expr(select->getCondition());
    Expr *true_expr  = translate_expr(select->getTrueValue());
    Expr *false_expr = translate_expr(select->getFalseValue());
    Expr *res        = Base::m_expr_factory.create_conditional(cond, true_expr, false_expr);

    res->set_location(Base::convert_location(select));
    return res;
}

// Translate an LLVM cast instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_call(
    llvm::CallInst *call,
    Definition     *dst_var)
{
    llvm::Function *func = call->getCalledFunction();
    if (func == nullptr) {
        MDL_ASSERT(func && "indirection function calls not supported");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::convert_location(call));
    }

    Def_function *func_def      = get_definition(func);
    bool         has_out_return = false;

    if (func_def != nullptr) {
        // handle converted out parameter
        Type_function *func_type = func_def->get_type();
        Type          *ret_type  = func_type->get_return_type();

        if (is<Type_void>(ret_type) && !call->getType()->isVoidTy()) {
            typename Type_function::Parameter *param = func_type->get_parameter(0);
            if (param->get_modifier() == Type_function::Parameter::PM_OUT) {
                has_out_return = true;
            }
        }
    }

    // prepare arguments:
    // 1. count all zero-sized array arguments
    unsigned num_void_args = 0;
    for (llvm::Value *arg : call->arg_operands()) {
        llvm::Type *arg_llvm_type = arg->getType();
        if (llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(arg_llvm_type)) {
            arg_llvm_type = pt->getPointerElementType();
        }

        Type *arg_type = Base::convert_type(arg_llvm_type);
        if (is<Type_void>(arg_type)) {
            ++num_void_args;
        }
    }

    size_t n_args = call->getNumArgOperands() - num_void_args;
    if (has_out_return) {
        ++n_args;
    }

    Small_VLA<Expr *, 8> args(Base::m_alloc, n_args);
    size_t i = 0;

    // 2. handle out parameter
    Def_variable *result_tmp = nullptr;
    if (has_out_return) {
        if (dst_var != nullptr) {
            args[i++] = Base::create_reference(dst_var);
            dst_var = nullptr;
        } else {
            result_tmp = create_local_var(Base::get_unique_sym("tmp", "tmp"), call->getType());
            args[i++] = Base::create_reference(result_tmp);
        }
    }

    // 3. convert the LLVM arguments
    for (llvm::Value *arg : call->arg_operands()) {
        llvm::Type *arg_llvm_type = arg->getType();
        if (llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(arg_llvm_type)) {
            arg_llvm_type = pt->getPointerElementType();
        }

        Type *arg_type = Base::convert_type(arg_llvm_type);
        if (is<Type_void>(arg_type)) {
            continue;  // skip void typed parameters
        }

        Expr *arg_expr = nullptr;
        if (llvm::isa<llvm::UndefValue>(arg)) {
            // the called function does not use this argument
            llvm::Type *t = arg->getType();

            // we call a function with an argument, try to find
            // a matching parameter and replace it
            // In our case, this typically "reverts" the LLVM optimization
            if (Definition *param_def = find_parameter_of_type(t)) {
                arg_expr = Base::create_reference(param_def);
            }
        }

        if (arg_expr == nullptr) {
            arg_expr = translate_expr(arg);
        }
        args[i++] = arg_expr;
    }

    Expr *expr_call = nullptr;

    if (func_def != nullptr) {
        DG_node<Definition> *call_node = m_dg.get_node(func_def);

        if (m_curr_node != nullptr) {
            m_dg.add_edge(m_curr_node, call_node);
        }

        Type_function *func_type = func_def->get_type();

        Symbol   *func_sym = func_def->get_symbol();
        Type     *ret_type = func_type->get_return_type();
        Expr_ref *callee   = Base::create_reference(func_sym, func_type);

        expr_call = Base::m_expr_factory.create_call(callee, args);
        expr_call->set_type(ret_type);
        expr_call->set_location(Base::convert_location(call));

    } else {
        // call to an unknown entity, should be a runtime function
        expr_call = Base::create_runtime_call(Base::convert_location(call), func, args);
    }

    Expr *res = expr_call;
    if (result_tmp != nullptr) {
        Expr *t = Base::create_reference(result_tmp);
        res = Base::create_binary(
            Expr_binary::OK_SEQUENCE,
            expr_call,
            t);
        res->set_type(t->get_type());
        res->set_location(expr_call->get_location());
    }

    if (dst_var != nullptr) {
        return create_assign_expr(dst_var, res);
    }
    return res;
}

// Translate an LLVM cast instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_cast(
    llvm::CastInst *inst)
{
    Expr *expr = translate_expr(inst->getOperand(0));

    llvm::Type *src_type = inst->getSrcTy();
    llvm::Type *dest_type = inst->getDestTy();

    switch (inst->getOpcode()) {
    case llvm::Instruction::ZExt:
        {
            if (inst->isIntegerCast()) {
                unsigned src_bits  = src_type->getIntegerBitWidth();
                unsigned dest_bits = dest_type->getIntegerBitWidth();
                if (src_bits == 32 && dest_bits == 64) {
                    // FIXME: i32 -> i64: ignore is not true in general
                    return expr;
                }

                // i* -> i*
                // Note: HLSL can implicitly convert from bool to integer, but it doesn't
                //    work when resolving overloads (for example for asfloat())
                Type *sl_type = Base::convert_type(inst->getType());
                return Base::create_type_cast(sl_type, expr);
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
                    Location loc = Base::convert_location(inst);

                    Type  *sl_type = Base::convert_type(inst->getType());
                    Value *m1      = Base::m_value_factory.get_int32(-1);
                    Value *z       = Base::m_value_factory.get_int32(0);
                    Expr  *t       = Base::m_expr_factory.create_literal(loc, m1);
                    Expr  *f       = Base::m_expr_factory.create_literal(loc, z);
                    Expr  *res     = Base::m_expr_factory.create_conditional(expr, t, f);
                    res->set_type(sl_type);
                    return res;
                }

                if (src_bits == 32 && dest_bits == 64) {
                    // FIXME: i32 -> i64: ignore is not true in general
                    return expr;
                }
            }
            MDL_ASSERT(!"unsupported LLVM SExt cast instruction");
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            return Base::m_expr_factory.create_invalid(Base::convert_location(inst));
        }

    case llvm::Instruction::Trunc:
        {
            Type *sl_type = Base::convert_type(inst->getType());
            if (expr->get_type() == sl_type) {
                return expr;
            }

            Expr *casted_expr = Base::create_type_cast(sl_type, expr);
            if (llvm::IntegerType *dest_int_type = llvm::dyn_cast<llvm::IntegerType>(dest_type)) {
                int trunc_mask = (1 << dest_int_type->getBitWidth()) - 1;
                Expr *trunk_mask_expr = Base::m_expr_factory.create_literal(
                    Base::zero_loc, Base::m_value_factory.get_int32(trunc_mask));
                return Base::create_binary(
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
            Type *sl_type = Base::convert_type(inst->getType());
            return Base::create_type_cast(sl_type, expr);
        }

    case llvm::Instruction::BitCast:
        {
            Type *sl_type = Base::convert_type(inst->getType());
            if (Expr *res = bitcast_to(expr, sl_type)) {
                return res;
            }

            llvm::FixedVectorType *src_vt  = llvm::dyn_cast<llvm::FixedVectorType>(src_type);
            llvm::FixedVectorType *dest_vt = llvm::dyn_cast<llvm::FixedVectorType>(dest_type);
            if (src_vt && dest_vt && src_vt->getNumElements() == dest_vt->getNumElements()) {
                llvm::Type *src_vet  = src_vt->getElementType();
                llvm::Type *dest_vet = dest_vt->getElementType();

                if (src_vet->isFloatingPointTy() && dest_vet->isIntegerTy()) {
                    Type *vt = Base::m_tc.get_vector(
                        Base::m_tc.int_type, src_vt->getNumElements());

                    return Base::create_float2int_bitcast(vt, expr);
                }

                if (src_vet->isIntegerTy() && dest_vet->isFloatingPointTy()) {
                    Type *vt = Base::m_tc.get_vector(
                        Base::m_tc.float_type, src_vt->getNumElements());

                    return Base::create_int2float_bitcast(vt, expr);
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
                if (src_elem_st != nullptr && dest_elem_st != nullptr &&
                    src_elem_st->hasName() && dest_elem_st->hasName() &&
                    (
                        (
                            src_elem_st->getName() == "State_core" &&
                            dest_elem_st->getName() == "class.State"
                        ) || (
                            src_elem_st->getName() == "class.State" &&
                            dest_elem_st->getName() == "State_core"
                        )
                    )) {
                    return expr;
                }
            }

            MDL_ASSERT(!"unexpected LLVM BitCast instruction");
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            return Base::m_expr_factory.create_invalid(Base::convert_location(inst));
        }

    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::AddrSpaceCast:
    default:
        MDL_ASSERT(!"unexpected LLVM cast instruction");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::convert_location(inst));
    }
}

// Translate an LLVM load instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_load(
    llvm::LoadInst *inst)
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
    if (base_pointer == nullptr) {
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    Type    *res_type      = Base::convert_type(inst->getType());
    Type    *res_elem_type = nullptr;
    int64_t res_elem_size  = 0;
    Type    *conv_to_type  = res_type;
    if (Type_vector *vt = as<Type_vector>(res_type)) {
        res_elem_type = vt->get_element_type();

        llvm::Type *llvm_elem_type;
        if (llvm::FixedVectorType *fvt = llvm::dyn_cast<llvm::FixedVectorType>(inst->getType())) {
            llvm_elem_type = fvt->getElementType();
        } else if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(inst->getType())) {
            // assumed to be a vector, so all element types are the same
            llvm_elem_type = st->getElementType(0);
        } else {
            MDL_ASSERT(!"Unexpected LLVM type for an HLSL vector");
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            return Base::m_expr_factory.create_invalid(Base::zero_loc);
        }

        res_elem_size = m_cur_data_layout->getTypeStoreSize(llvm_elem_type);
        conv_to_type = res_elem_type;
    }

    typename vector<Expr *>::Type expr_parts(Base::m_alloc);
    uint64_t cur_offs = 0;  // TODO: is it valid to assume correct alignment here?
    while (target_size > 0) {
        if (res_elem_size != 0 && !move_into_compound_elem(stack, res_elem_size)) {
            MDL_ASSERT(!"couldn't access element of right size");
            break;
        }
        Expr *cur_expr = create_compound_elem_expr(stack, base_pointer);
        if (cur_expr->get_type() != conv_to_type) {
            if (Expr *converted_expr = bitcast_to(cur_expr, conv_to_type)) {
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
        Type *part_type = expr_parts.back()->get_type();
        if (res_elem_type && part_type != res_elem_type) {
            if (Type_vector *part_vt = as<Type_vector>(part_type)) {
                expr_parts.back() = create_vector_access(expr_parts.back(), 0);
                for (size_t i = 1, n = part_vt->get_size(); i < n; ++i) {
                    Expr *sub_part = create_compound_elem_expr(stack, base_pointer);
                    expr_parts.push_back(create_vector_access(sub_part, unsigned(i)));
                }
            } else {
                MDL_ASSERT(!"unexpected part type");
            }
        }

        // done?
        if (target_size <= 0) {
            break;
        }

        // go to next element
        if (!move_to_next_compound_elem(stack, target_size)) {
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            return Base::m_expr_factory.create_invalid(Base::zero_loc);
        }
    }

    // atomic element read, just return it
    if (expr_parts.size() == 1) {
        return expr_parts[0];
    }

    // we need to construct the result from multiple parts
    MDL_ASSERT(expr_parts.size() > 0);
    return create_constructor_call(res_type, expr_parts, Base::convert_location(inst));
}

// Translate an LLVM store instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_store(
    llvm::StoreInst *inst)
{
    llvm::Value *pointer = inst->getPointerOperand();
    Expr        *lvalue  = translate_lval_expression(pointer);

    llvm::Value *value   = inst->getValueOperand();
    Expr        *expr    = translate_expr(value);

    return Base::create_binary(Expr_binary::OK_ASSIGN, lvalue, expr);
}

// Translate an LLVM pointer value to an AST lvalue expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_lval_expression(
    llvm::Value *pointer)
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
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            return Base::m_expr_factory.create_invalid(Base::convert_location(gep));
        }

        // walk the base expression using select operations according to the indices of the gep
        llvm::Type *cur_llvm_type = gep->getPointerOperandType()->getPointerElementType();
        Expr *cur_expr = translate_lval_expression(gep->getPointerOperand());
        Type *cur_type = cur_expr->get_type()->skip_type_alias();

        for (unsigned i = 1, num_indices = gep->getNumIndices(); i < num_indices; ++i) {
            // check whether a bitcast wants us to stop going deeper into the compound
            if (target_size != 0) {
                uint64_t cur_type_size = m_cur_data_layout->getTypeStoreSize(cur_llvm_type);
                if (cur_type_size <= target_size) {
                    break;
                }
            }

            if (is<Type_array>(cur_type) ||
                is<Type_vector>(cur_type) ||
                is<Type_matrix>(cur_type))
            {
                llvm::Value *gep_index = gep->getOperand(i + 1);
                if (is<Type_vector>(cur_type) &&
                        llvm::isa<llvm::ConstantInt>(gep_index)) {
                    llvm::ConstantInt *idx = llvm::cast<llvm::ConstantInt>(gep_index);
                    cur_expr = create_vector_access(cur_expr, unsigned(idx->getZExtValue()));
                } else {
                    Expr *array_index = translate_expr(gep_index);
                    cur_expr = Base::create_binary(
                        Expr_binary::OK_ARRAY_SUBSCRIPT, cur_expr, array_index);
                }

                cur_type = cast<Type_compound>(cur_type)->get_compound_type(0);
                cur_type = cur_type->skip_type_alias();
                cur_expr->set_type(cur_type);
                cur_llvm_type = llvm::GetElementPtrInst::getTypeAtIndex(
                    cur_llvm_type, (uint64_t) 0);

                continue;
            }

            if (Type_struct *struct_type = as<Type_struct>(cur_type)) {
                llvm::ConstantInt *idx = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(i + 1));
                if (idx == nullptr) {
                    MDL_ASSERT(!"invalid field index for a struct type");
                    Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                    return Base::m_expr_factory.create_invalid(Base::convert_location(gep));
                }
                unsigned idx_val = unsigned(idx->getZExtValue());
                typename Type_struct::Field *field = struct_type->get_field(idx_val);
                Definition                  *f_def = get_field_definition(
                    struct_type, field->get_symbol());

                Expr *field_ref = f_def != nullptr ?
                    Base::create_reference(f_def) :
                    Base::create_reference(field->get_symbol(), field->get_type());
                cur_expr = Base::create_binary(
                    Expr_binary::OK_SELECT, cur_expr, field_ref);

                cur_type = field->get_type()->skip_type_alias();
                cur_expr->set_type(cur_type);
                cur_llvm_type = llvm::GetElementPtrInst::getTypeAtIndex(
                    cur_llvm_type, (uint64_t) idx_val);
                continue;
            }

            MDL_ASSERT(!"Unexpected element type for GEP");
        }
        return cur_expr;
    }

    // should be alloca or pointer parameter
    auto it = m_local_var_map.find(pointer);
    if (it != m_local_var_map.end()) {
        return Base::create_reference(it->second);
    }
    MDL_ASSERT(!"unexpected unmapped alloca or pointer parameter");
    Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
    return Base::m_expr_factory.create_invalid(Base::zero_loc);
}

// Returns the given expression as a call expression, if it is a call to a vector constructor.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr_call *SLWriterPass<BasePass>::as_vector_constructor_call(
    Expr *expr)
{
    Expr_call *call = as<Expr_call>(expr);
    if (call == nullptr) {
        return nullptr;
    }
    Expr_ref *callee_ref = as<Expr_ref>(call->get_callee());
    if (callee_ref == nullptr) {
        return nullptr;
    }

    Type *res_type = call->get_type();
    if (res_type == nullptr || !is<Type_vector>(res_type->skip_type_alias())) {
        return nullptr;
    }

    Definition *def = callee_ref->get_definition();
    if (def == nullptr) {
        return nullptr;
    }

    Def_function *call_def = as<Def_function>(def);
    if (call_def == nullptr) {
        return nullptr;
    }

    if (call_def->get_semantics() != Def_function::DS_ELEM_CONSTRUCTOR) {
        return nullptr;
    }

    return call;
}

// Translate an LLVM ShuffleVector instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_shufflevector(
    llvm::ShuffleVectorInst *inst)
{
    Expr *v1_expr = translate_expr(inst->getOperand(0));

    // is this shuffle a swizzle?
    if (llvm::isa<llvm::UndefValue>(inst->getOperand(1))) {
        if (Expr_call *constr_call = as_vector_constructor_call(v1_expr)) {
            Type_vector *call_vt = cast<Type_vector>(constr_call->get_type());
            size_t len = inst->getShuffleMask().size();
            Type_vector *res_type = Base::m_tc.get_vector(
                call_vt->get_element_type(), len);

            Small_VLA<Expr *, 4> vec_elems(Base::m_alloc, len);
            unsigned cur_elem_idx = 0;
            for (int index : inst->getShuffleMask()) {
                vec_elems[cur_elem_idx] = constr_call->get_argument(index);
                ++cur_elem_idx;
            }

            return create_constructor_call(res_type, vec_elems, Base::convert_location(inst));
        }

        string shuffle_name(Base::m_alloc);
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

            if (char const *index_name = Base::get_vector_index_str(uint64_t(index))) {
                shuffle_name.append(index_name);
            } else {
                MDL_ASSERT(!"invalid shuffle mask");
                Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                return Base::m_expr_factory.create_invalid(Base::convert_location(inst));
            }
        }

        Type *res_type  = Base::convert_type(inst->getType());
        Expr *index_ref = Base::create_reference(
            Base::get_sym(shuffle_name.c_str()), res_type);
        Expr *res       = Base::create_binary(
            Expr_binary::OK_SELECT, v1_expr, index_ref);
        res->set_type(res_type);
        res->set_location(Base::convert_location(inst));

        return res;
    }

    // no, use constructor for translation
    // collect elements from both LLVM values via the shuffle matrix
    Expr *v2_expr = translate_expr(inst->getOperand(1));

    uint64_t num_elems = llvm::cast<llvm::FixedVectorType>(
        inst->getType())->getNumElements();
    int v1_size = llvm::cast<llvm::FixedVectorType>(
        inst->getOperand(0)->getType())->getNumElements();
    Small_VLA<Expr *, 4> vec_elems(Base::m_alloc, num_elems);
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

        if (index >= v1_size) {
            vec_elems[cur_elem_idx] = create_vector_access(v2_expr, unsigned(index - v1_size));
        } else {
            vec_elems[cur_elem_idx] = create_vector_access(v1_expr, unsigned(index));
        }
        ++cur_elem_idx;
    }

    return create_constructor_call(
        Base::convert_type(inst->getType()), vec_elems, Base::convert_location(inst));
}

// Create a select or array_subscript expression on an element of a vector.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_vector_access(
    Expr     *vec,
    unsigned index)
{
    Type_vector *vt = as<Type_vector>(vec->get_type());
    if (vt == nullptr) {
        MDL_ASSERT(!"create_vector_access on non-vector expression");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    if (Expr_literal *literal_expr = as<Expr_literal>(vec)) {
        if (Value_vector *val = as<Value_vector>(literal_expr->get_value())) {
            Value *extracted_value = val->extract(Base::m_value_factory, index);
            return Base::m_expr_factory.create_literal(Base::zero_loc, extracted_value);
        }
    }

    Expr *res;

    if (Symbol *index_sym = get_vector_index_sym(index)) {
        Expr *index_ref = Base::m_expr_factory.create_reference(Base::get_type_name(index_sym));

        index_ref->set_type(vt->get_element_type());

        res = Base::create_binary(Expr_binary::OK_SELECT, vec, index_ref);
    } else {
        Expr *index_expr = Base::m_expr_factory.create_literal(
            Base::zero_loc,
            Base::m_value_factory.get_int32(index));
        res = Base::create_binary(Expr_binary::OK_ARRAY_SUBSCRIPT, vec, index_expr);
    }

    res->set_type(vt->get_element_type());
    return res;
}

// Translate an LLVM InsertElement instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_insertelement(
    llvm::InsertElementInst *inst)
{
    // collect all values for the vector by following the InsertElement instruction chain
    uint64_t num_elems = llvm::cast<llvm::FixedVectorType>(inst->getType())->getNumElements();
    uint64_t remaining_elems = num_elems;
    Small_VLA<Expr *, 4> vec_elems(Base::m_alloc, num_elems);
    memset(vec_elems.data(), 0, vec_elems.size() * sizeof(Expr *));

    llvm::InsertElementInst *cur_insert = inst;
    llvm::Value *cur_value = cur_insert;
    while (!has_local_var(cur_value)) {
        llvm::Value *index = cur_insert->getOperand(2);
        if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
            uint64_t index_val = ci->getZExtValue();
            if (index_val >= num_elems) {
                MDL_ASSERT(!"invalid InsertElement instruction");
                Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                return Base::m_expr_factory.create_invalid(Base::zero_loc);
            }
            if (vec_elems[index_val] == nullptr) {
                vec_elems[index_val] = translate_expr(cur_insert->getOperand(1));

                // all vector element initializers found?
                if (--remaining_elems == 0) {
                    break;
                }
            }
        } else {
            MDL_ASSERT(!"InsertElement with non-constant index not supported");
            Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
            return Base::m_expr_factory.create_invalid(Base::zero_loc);
        }

        cur_value = cur_insert->getOperand(0);
        cur_insert = llvm::dyn_cast<llvm::InsertElementInst>(cur_value);
        if (cur_insert == nullptr) {
            break;
        }
    }

    // not all elements found? -> try extracting them from the current remaining value
    if (remaining_elems) {
        if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(cur_value)) {
            for (uint64_t i = 0; i < num_elems; ++i) {
                if (vec_elems[i] != nullptr) {
                    continue;
                }
                vec_elems[i] = translate_expr(cv->getOperand(unsigned(i)));
            }
            remaining_elems = 0;
        } else if (llvm::ShuffleVectorInst *sv = llvm::dyn_cast<llvm::ShuffleVectorInst>(cur_value))
        {
            // if only remaining elements is the first and comes from a shuffle vector inst
            // with non-undef first element
            if (remaining_elems == 1 &&
                vec_elems[0] == nullptr &&
                sv->getShuffleMask()[0] != llvm::UndefMaskElem) {
                Expr *base_expr = translate_expr(sv->getOperand(0));
                vec_elems[0] = create_vector_access(base_expr, unsigned(sv->getShuffleMask()[0]));
                remaining_elems = 0;
            }
        }
    }

    // if special cases failed, access elements directly
    if (remaining_elems) {
        Expr *base_expr = translate_expr(cur_value);
        for (uint64_t i = 0; i < num_elems; ++i) {
            if (vec_elems[i] != nullptr) {
                continue;
            }

            vec_elems[i] = create_vector_access(base_expr, unsigned(i));
        }
    }

    return create_constructor_call(
        Base::convert_type(inst->getType()), vec_elems, Base::convert_location(inst));
}

/// Helper class to process insertvalue instructions of nested types.
template<typename WriterPass>
class InsertValueObject
{
    typedef typename WriterPass::Location  Location;
    typedef typename WriterPass::Expr      Expr;
public:
    /// Constructor.
    InsertValueObject(
        Arena_builder  *arena_builder,
        llvm::Type     *llvm_type,
        Location const &loc)
    : m_expr(nullptr)
    , m_children(*arena_builder->get_arena(), get_num_children_for_type(llvm_type))
    , m_loc(loc)
    {
        if (m_children.size()) {
            // Add children if get_num_children_for_type() said, we need them
            for (size_t i = 0, n = m_children.size(); i < n; ++i) {
                llvm::Type *child_type = llvm::GetElementPtrInst::getTypeAtIndex(
                    llvm_type, unsigned(i));
                m_children[i] =
                    arena_builder->create<InsertValueObject>(arena_builder, child_type, loc);
            }
        }
    }

    /// Get the child at the given index.
    ///
    /// \returns nullptr, if the index is invalid
    InsertValueObject *get_child(unsigned index)
    {
        MDL_ASSERT(index < m_children.size());
        if (index >= m_children.size()) {
            return nullptr;
        }
        return m_children[index];
    }

    /// Returns true, if the object is already fully set by later InsertValue instructions.
    bool has_expr()
    {
        return m_expr != nullptr;
    }

    /// Set the expression for this object.
    void set_expr(Expr *expr)
    {
        m_expr = expr;
    }

    /// Translate the object into an expression.
    ///
    /// \param alloc      an allocator used for temporary arrays
    /// \param writer     the SLWriterPass to create AST
    /// \param base_expr  an expression to use, when the expression for this object has not been set
    Expr *translate(IAllocator *alloc, WriterPass *writer, Expr *base_expr)
    {
        if (m_expr != nullptr) {
            return m_expr;
        }

        if (m_children.size() == 0) {
            return base_expr;
        }

        Small_VLA<Expr *, 4> agg_elems(alloc, m_children.size());
        for (size_t i = 0, n = m_children.size(); i < n; ++i) {
            Expr *child_base_expr = writer->create_compound_access(base_expr, unsigned(i));

            agg_elems[i] = m_children[i]->translate(alloc, writer, child_base_expr);
        }

        return writer->create_constructor_call(base_expr->get_type(), agg_elems, m_loc);
    }

private:
    /// Returns the number of children objects which should be created for the given type.
    static size_t get_num_children_for_type(llvm::Type *llvm_type)
    {
        if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(llvm_type)) {
            return st->getNumElements();
        } else if (llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(llvm_type)) {
            return at->getNumElements();
        } else {
            return 0;
        }
    }

private:
    Expr                           *m_expr;
    Arena_VLA<InsertValueObject *> m_children;
    Location                       m_loc;
};

// Translate an LLVM InsertValue instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_insertvalue(
    llvm::InsertValueInst *inst)
{
    // collect all values for the struct or array by following the InsertValue instruction chain
    llvm::Type *comp_type = inst->getType();

    Memory_arena arena(Base::m_alloc);
    Arena_builder builder(arena);
    InsertValueObject<Self> root(&builder, comp_type, Base::convert_location(inst));

    llvm::InsertValueInst *cur_insert = inst;
    llvm::Value           *cur_value  = cur_insert;
    while (!has_local_var(cur_value)) {
        InsertValueObject<Self> *cur_obj = &root;
        llvm::ArrayRef<unsigned> indices = cur_insert->getIndices();
        for (unsigned i = 0, n = indices.size(); i < n && !cur_obj->has_expr(); ++i) {
            unsigned cur_index = indices[i];
            cur_obj = cur_obj->get_child(cur_index);
        }

        // only overwrite the value, if it has not been set, yet (here or in a parent level)
        if (!cur_obj->has_expr()) {
            Expr *cur_expr = translate_expr(cur_insert->getInsertedValueOperand());
            cur_obj->set_expr(cur_expr);
        }

        cur_value = cur_insert->getAggregateOperand();
        cur_insert = llvm::dyn_cast<llvm::InsertValueInst>(cur_value);
        if (cur_insert == nullptr) {
            break;
        }
    }

    // translate collected values into an expression
    Expr *base_expr = translate_expr(cur_value);
    Expr *res = root.translate(Base::m_alloc, this, base_expr);
    return res;
}

// Translate an LLVM ExtractElement instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_extractelement(
    llvm::ExtractElementInst *extract)
{
    Expr        *expr  = translate_expr(extract->getVectorOperand());
    llvm::Value *index = extract->getIndexOperand();

    Type *res_type = Base::convert_type(extract->getType());
    Expr *res;

    if (Symbol *index_sym = get_vector_index_sym(index)) {
        Expr *index_ref = Base::create_reference(index_sym, res_type);
        res = Base::create_binary(
            Expr_binary::OK_SELECT, expr, index_ref);
    } else {
        Expr *index_expr = translate_expr(index);
        res = Base::create_binary(
            Expr_binary::OK_ARRAY_SUBSCRIPT, expr, index_expr);
    }

    res->set_type(res_type);
    res->set_location(Base::convert_location(extract));

    return res;
}

// Translate an LLVM ExtractValue instruction to an AST expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::translate_expr_extractvalue(
    llvm::ExtractValueInst *extract)
{
    Expr *res = translate_expr(extract->getAggregateOperand());
    Type *cur_type = res->get_type()->skip_type_alias();

    for (unsigned i : extract->getIndices()) {
        if (is<Type_array>(cur_type) ||
            is<Type_matrix>(cur_type))
        {
            Expr *index_expr = Base::m_expr_factory.create_literal(
                Base::zero_loc,
                Base::m_value_factory.get_int32(i));
            res = Base::create_binary(
                Expr_binary::OK_ARRAY_SUBSCRIPT, res, index_expr);
        } else if (is<Type_vector>(cur_type)) {
            // due to type mapping, this could also be a vector
            res = create_vector_access(res, i);
        } else {
            Type_struct *s_type = cast<Type_struct>(cur_type);

            if (typename Type_struct::Field *field = s_type->get_field(i)) {
                Definition *f_def = get_field_definition(s_type, field->get_symbol());

                Expr *index_ref = f_def != nullptr ?
                    Base::create_reference(f_def) :
                    Base::create_reference(field->get_symbol(), field->get_type());
                res = Base::create_binary(
                    Expr_binary::OK_SELECT, res, index_ref);
            } else {
                MDL_ASSERT(!"ExtractValue index too high");
                Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
                res = Base::m_expr_factory.create_invalid(Base::zero_loc);
            }
        }

        cur_type = get_compound_sub_type(cast<Type_compound>(cur_type), i);
        cur_type = cur_type->skip_type_alias();
        res->set_type(cur_type);
    }

    res->set_location(Base::convert_location(extract));

    return res;
}

// Get the type of the i-th subelement.
template<typename BasePass>
typename SLWriterPass<BasePass>::Type *SLWriterPass<BasePass>::get_compound_sub_type(
    Type_compound *comp_type, unsigned i)
{
    if (Type_vector *v_tp = as<Type_vector>(comp_type)) {
        return v_tp->get_element_type();
    } else if (Type_matrix *m_tp = as<Type_matrix>(comp_type)) {
        return m_tp->get_element_type();
    } else if (Type_array *a_tp = as<Type_array>(comp_type)) {
        return a_tp->get_element_type();
    } else {
        Type_struct *s_tp = cast<Type_struct>(comp_type);
        return s_tp->get_compound_type(i);
    }
}

// Add a parameter to the given function and the current definition table
// and return its definition.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_param *SLWriterPass<BasePass>::add_func_parameter(
    Declaration_function *func,
    char const           *param_name,
    Type                 *param_type)
{
    // Note: HLSL does not support array specifiers on types while GLSL does.
    // So far, we always create the "C-Syntax", by adding the array specifiers
    // to the parameter name
    Declaration_param *decl_param = Base::m_decl_factory.create_param(
        Base::get_type_name(inner_element_type(param_type)));
    Symbol *param_sym = Base::get_unique_sym(param_name, param_name);
    Name *param_sl_name = Base::get_name(Base::zero_loc, param_sym);
    decl_param->set_name(param_sl_name);
    Base::add_array_specifiers(decl_param, param_type);
    func->add_param(decl_param);

    Def_param *param_elem_def = Base::m_def_tab.enter_parameter_definition(
        param_sym, param_type, &param_sl_name->get_location());
    param_elem_def->set_declaration(decl_param);
    param_sl_name->set_definition(param_elem_def);

    return param_elem_def;
}

// Get the AST definition of a generated struct constructor.
// This generates the struct constructor, if it does not exist, yet.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_function *SLWriterPass<BasePass>::get_struct_constructor(
    Type_struct *type)
{
    auto it = m_struct_constructor_map.find(type);
    if (it != m_struct_constructor_map.end()) {
        return it->second;
    }

    // declare the constructor function
    string constr_name("constr_", Base::m_alloc);
    constr_name += type->get_sym()->get_name();

    // generate name in current scope to avoid name clashes
    Symbol *func_sym = Base::get_unique_sym(constr_name.c_str(), "constr");

    typename Definition_table::Scope_transition transition(
        Base::m_def_tab, Base::m_def_tab.get_global_scope());

    Declaration_function *decl_func = Base::m_decl_factory.create_function(
        Base::get_type_name(type),
        Base::get_name(Base::zero_loc, constr_name.c_str()));

    // build the function type
    typename vector<typename Type_function::Parameter>::Type params(Base::m_alloc);
    for (size_t i = 0, n = type->get_field_count(); i < n; ++i) {
        typename Type_struct::Field *field = type->get_field(i);
        Type *field_type = field->get_type();
        if (Type_array *at = as<Type_array>(field_type)) {
            // add one parameter per array element
            Type *elem_type = at->get_element_type();
            for (size_t j = 0, num_elems = at->get_size(); j < num_elems; ++j) {
                params.push_back(typename Type_function::Parameter(
                    elem_type, Type_function::Parameter::PM_IN));
            }
        } else {
            params.push_back(typename Type_function::Parameter(
                field->get_type(), Type_function::Parameter::PM_IN));
        }
    }

    // create the function definition
    Type_function *func_type = Base::m_tc.get_function(type, params);
    Def_function  *func_def  = Base::m_def_tab.enter_function_definition(
        func_sym, func_type, Def_function::DS_ELEM_CONSTRUCTOR, &Base::zero_loc);

    func_def->set_declaration(decl_func);

    // create the body
    {
        typename Definition_table::Scope_enter enter(Base::m_def_tab, func_def);

        Stmt_compound *block = Base::m_stmt_factory.create_compound(Base::zero_loc);

        // declare the struct "res" variable
        Declaration_variable *decl_var  = Base::m_decl_factory.create_variable(
            Base::get_type_name(type));
        Init_declarator      *init_decl = Base::m_decl_factory.create_init_declarator(
            Base::zero_loc);
        Symbol               *var_sym   = Base::get_unique_sym("res", "res");
        Name                 *var_name  = Base::get_name(Base::zero_loc, var_sym);
        init_decl->set_name(var_name);
        decl_var->add_init(init_decl);

        Def_variable *var_def = Base::m_def_tab.enter_variable_definition(
            var_sym, type, &var_name->get_location());
        var_def->set_declaration(decl_var);
        var_name->set_definition(var_def);

        block->add_stmt(Base::m_stmt_factory.create_declaration(decl_var));

        // add a parameter and an assignment to our "res" variable per struct field
        for (size_t i = 0, n = type->get_field_count(); i < n; ++i) {
            typename Type_struct::Field *field = type->get_field(i);

            if (Type_array *at = as<Type_array>(field->get_type())) {
                // add one parameter per array element and fill the array in "res" with them
                Type *elem_type = at->get_element_type();
                char const *field_name = field->get_symbol()->get_name();
                for (size_t j = 0, num_elems = at->get_size(); j < num_elems; ++j) {
                    Def_param *param = add_func_parameter(decl_func, field_name, elem_type);

                    Expr *lvalue_expr = create_field_access(
                        Base::create_reference(var_def),
                        unsigned(i));
                    lvalue_expr = create_array_access(lvalue_expr, unsigned(j));
                    Stmt *assign_stmt = create_assign_stmt(
                        lvalue_expr,
                        Base::create_reference(param));
                    block->add_stmt(assign_stmt);
                }
                continue;
            }

            Def_param *param =
                add_func_parameter(decl_func, field->get_symbol()->get_name(), field->get_type());

            Expr *lvalue_expr = create_field_access(
                Base::create_reference(var_def),
                unsigned(i));
            Stmt *assign_stmt = create_assign_stmt(
                lvalue_expr,
                Base::create_reference(param));
            block->add_stmt(assign_stmt);
        }

        // return the struct
        block->add_stmt(Base::m_stmt_factory.create_return(
            Base::zero_loc,
            Base::create_reference(var_def)));

        decl_func->set_body(block);
        Base::m_unit->add_decl(decl_func);
    }

    m_struct_constructor_map[type] = func_def;
    return func_def;
}

// Get an AST symbol for a given vector index, if possible.
// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
template<typename BasePass>
typename SLWriterPass<BasePass>::Symbol *SLWriterPass<BasePass>::get_vector_index_sym(
    uint64_t index)
{
    if (char const *index_name = Base::get_vector_index_str(index)) {
        return Base::get_sym(index_name);
    }
    return nullptr;
}

// Get the AST symbol for a given LLVM vector index, if possible.
// Translates constants 0 - 3 to x, y, z, w. Otherwise returns nullptr.
template<typename BasePass>
typename SLWriterPass<BasePass>::Symbol *SLWriterPass<BasePass>::get_vector_index_sym(
    llvm::Value *index)
{
    if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
        return get_vector_index_sym(ci->getZExtValue());
    }
    return nullptr;
}

// Create a select expression on a field of a struct.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_field_access(
    Expr     *struct_expr,
    unsigned field_index)
{
    Type_struct *type = as<Type_struct>(struct_expr->get_type());
    if (type == nullptr) {
        MDL_ASSERT(!"create_field_access on non-struct expression");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    // fold a field access on a constructor call to the corresponding constructor argument
    if (Expr_call *call = as<Expr_call>(struct_expr)) {
        if (Expr_ref *callee_ref = as<Expr_ref>(call->get_callee())) {
            if (Def_function *callee_def =
                    as<Def_function>(callee_ref->get_definition())) {
                if (callee_def->get_semantics() == Def_function::DS_ELEM_CONSTRUCTOR) {
                    // if the struct contains any array types, they were flattened,
                    // so we need to get the right argument index
                    size_t arg_idx = 0;
                    for (size_t i = 0, n = type->get_compound_size(); i < n; ++i) {
                        Type *field_type = type->get_compound_type(i);
                        if (Type_array *at = as<Type_array>(field_type)) {
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

    typename Type_struct::Field *field = type->get_field(field_index);
    if (field == nullptr) {
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    Definition *f_def = get_field_definition(type, field->get_symbol());

    Expr *field_ref = f_def != nullptr ?
        Base::create_reference(f_def) :
        Base::create_reference(field->get_symbol(), field->get_type());

    Expr *res = Base::create_binary(
        Expr_binary::OK_SELECT, struct_expr, field_ref);
    res->set_type(field->get_type());
    return res;
}

// Create an array_subscript expression on an element of an array.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_array_access(
    Expr     *array,
    unsigned index)
{
    Type_array *type = as<Type_array>(array->get_type());
    if (type == nullptr) {
        MDL_ASSERT(!"create_array_access on non-array expression");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    if (Expr_literal *literal_expr = as<Expr_literal>(array)) {
        if (Value_array *val = as<Value_array>(literal_expr->get_value())) {
            Value *extracted_value = val->extract(Base::m_value_factory, index);
            return Base::m_expr_factory.create_literal(Base::zero_loc, extracted_value);
        }
    }

    if (Expr_compound *comp_expr = as<Expr_compound>(array)) {
        return comp_expr->get_element(index);
    }

    Expr *index_expr = Base::m_expr_factory.create_literal(
        Base::zero_loc,
        Base::m_value_factory.get_int32(int32_t(index)));
    Expr *res = Base::create_binary(
        Expr_binary::OK_ARRAY_SUBSCRIPT, array, index_expr);
    res->set_type(type->get_element_type());
    return res;
}

// Create an array_subscript expression on a matrix.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_matrix_access(
    Expr     *matrix,
    unsigned index)
{
    Type_matrix *type = as<Type_matrix>(matrix->get_type());
    if (type == nullptr) {
        MDL_ASSERT(!"create_matrix_access on non-matrix expression");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }

    if (Expr_literal *literal_expr = as<Expr_literal>(matrix)) {
        if (Value_matrix *val = as<Value_matrix>(literal_expr->get_value())) {
            Value *extracted_value = val->extract(Base::m_value_factory, index);
            return Base::m_expr_factory.create_literal(Base::zero_loc, extracted_value);
        }
    }

    Expr *index_expr = Base::m_expr_factory.create_literal(
        Base::zero_loc,
        Base::m_value_factory.get_int32(int32_t(index)));
    Expr *res = Base::create_binary(
        Expr_binary::OK_ARRAY_SUBSCRIPT, matrix, index_expr);
    res->set_type(type->get_element_type());
    return res;
}

// Create a select expression on a field of a struct.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_compound_access(
    Expr     *comp_expr,
    unsigned comp_index)
{
    Type *comp_type = comp_expr->get_type()->skip_type_alias();
    switch (comp_type->get_kind()) {
    case Type::TK_STRUCT:
        return create_field_access(comp_expr, comp_index);

    case Type::TK_ARRAY:
        return create_array_access(comp_expr, comp_index);

    case Type::TK_VECTOR:
        return create_vector_access(comp_expr, comp_index);

    case Type::TK_MATRIX:
        return create_matrix_access(comp_expr, comp_index);

    default:
        MDL_ASSERT(!"create_compound_access on invalid expression");
        Base::error(mi::mdl::INTERNAL_JIT_UNSUPPORTED_EXPR);
        return Base::m_expr_factory.create_invalid(Base::zero_loc);
    }
}

// Create an assign statement, assigning an expression to an lvalue expression.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::create_assign_stmt(
    Expr *lvalue,
    Expr *expr)
{
    Expr *assign = Base::create_binary(
        Expr_binary::OK_ASSIGN, lvalue, expr);
    assign->set_location(lvalue->get_location());
    assign->set_type(expr->get_type());
    return Base::m_stmt_factory.create_expression(
        lvalue->get_location(), assign);
}

// Create an assign expression, assigning an expression to a variable.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_assign_expr(
    Definition *var_def,
    Expr       *expr)
{
    Expr *lvalue = Base::create_reference(var_def);
    Expr *assign = Base::create_binary(
        Expr_binary::OK_ASSIGN, lvalue, expr);
    assign->set_location(expr->get_location());
    assign->set_type(expr->get_type());
    return assign;
}

// Create an assign statement, assigning an expression to a variable.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::create_assign_stmt(
    Definition *var_def,
    Expr       *expr)
{
    Expr *assign = create_assign_expr(var_def, expr);
    return Base::m_stmt_factory.create_expression(assign->get_location(), assign);
}

// Generates a constructor call for the given AST type.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_constructor_call(
    Type                    *res_type,
    Array_ref<Expr *> const &args,
    Location const          &loc,
    bool                    is_global)
{
    if (Base::compound_allowed(is_global)) {
        Expr *res = Base::m_expr_factory.create_compound(loc, args);
        res->set_type(res_type);
        return res;
    }
    return create_constructor_call(res_type, args, loc);
}

// Generates a constructor call for the given AST type.
template<typename BasePass>
typename SLWriterPass<BasePass>::Expr *SLWriterPass<BasePass>::create_constructor_call(
    Type                    *type,
    Array_ref<Expr *> const &args,
    Location const          &loc)
{
    Expr *ref;

    type = type->skip_type_alias();

    if (Type_struct *res_type = as<Type_struct>(type)) {
        Def_function *constr_def = get_struct_constructor(res_type);
        ref = Base::create_reference(constr_def);
    } else if (Type_array *arr_type = as<Type_array>(type)) {
        // this should only appear as part of a complex insertvalue chain
        // and should be broken apart again by a constructor call using this as an argument
        return Base::create_initializer(loc, arr_type, args);
    } else {
        Def_function *def = Base::lookup_constructor(type, args);
        if (def != nullptr) {
            ref = Base::create_reference(def);
        } else {
            Type_name *constr_name = Base::get_type_name(type);
            ref = Base::create_reference(constr_name, type);
        }
    }

    // for struct types, check whether we have any array arguments
    bool has_array_args = false;
    size_t num_args = 0;
    if (is<Type_struct>(type)) {
        for (Expr *expr : args) {
            if (Type_array *at = as<Type_array>(expr->get_type())) {
                has_array_args = true;
                num_args += at->get_size();
            } else {
                ++num_args;
            }
        }
    }

    Expr *res;
    if (has_array_args) {
        // arrays are split into one parameter per array element
        Small_VLA<Expr *, 8> split_args(Base::m_alloc, num_args);
        size_t cur_idx = 0;
        for (Expr *expr : args) {
            if (Type_array *at = as<Type_array>(expr->get_type())) {
                for (size_t i = 0, n = at->get_size(); i < n; ++i) {
                    split_args[cur_idx++] = create_array_access(expr, unsigned(i));
                }
            } else {
                split_args[cur_idx++] = expr;
            }
        }
        res = Base::m_expr_factory.create_call(ref, split_args);
    } else {
        res = Base::m_expr_factory.create_call(ref, args);
    }

    res->set_type(type);
    res->set_location(loc);
    return res;
}

// Generates a new local variable for an AST symbol and an LLVM type.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_variable *SLWriterPass<BasePass>::create_local_var(
    Symbol     *var_sym,
    llvm::Type *type,
    bool       add_decl_to_prolog)
{
    // Note: HLSL does not support array specifiers on types while GLSL does.
    // So far, we always create the "C-Syntax", by adding the array specifiers
    // to the variable name
    Type      *var_type      = Base::convert_type(type);
    Type_name *var_type_name = Base::get_type_name(inner_element_type(var_type));

    Declaration_variable *decl_var = Base::m_decl_factory.create_variable(var_type_name);

    Init_declarator *init_decl = Base::m_decl_factory.create_init_declarator(Base::zero_loc);
    Name *var_name = Base::get_name(Base::zero_loc, var_sym);
    init_decl->set_name(var_name);
    Base::add_array_specifiers(init_decl, var_type);
    decl_var->add_init(init_decl);

    Def_variable *var_def = Base::m_def_tab.enter_variable_definition(
        var_sym, var_type, &var_name->get_location());
    var_def->set_declaration(decl_var);
    var_name->set_definition(var_def);

    if (add_decl_to_prolog) {
        // so far, add all declarations to the function scope
        m_cur_start_block->add_stmt(Base::m_stmt_factory.create_declaration(decl_var));
    }

    return var_def;
}

// Generates a new local variable for an LLVM value and use this variable as the value's
// result in further generated AST code.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_variable *SLWriterPass<BasePass>::create_local_var(
    llvm::Value *value,
    bool        do_not_register,
    bool        add_decl_to_prolog)
{
    llvm::Type *type = value->getType();
    if (llvm::isa<llvm::AllocaInst>(value)) {
        // alloc instructions return a pointer to the allocated space
        llvm::PointerType *p_type = llvm::cast<llvm::PointerType>(type);
        type = p_type->getElementType();
    }

    if (type->isVoidTy()) {
        return nullptr;
    }

    auto it = m_local_var_map.find(value);
    if (it != m_local_var_map.end()) {
        // this should be a variable, otherwise something did go really wrong
        return cast<Def_variable>(it->second);
    }

    Symbol *var_sym = Base::get_unique_sym(value->getName(), "tmp");

    Def_variable *var_def = create_local_var(var_sym, type, add_decl_to_prolog);

    if (!do_not_register) {
        m_local_var_map[value] = var_def;
    }
    return var_def;
}

// Generates a new local const variable to hold an LLVM value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_variable *SLWriterPass<BasePass>::create_local_const(
    llvm::Constant *cv)
{
    auto it = m_local_var_map.find(cv);
    if (it != m_local_var_map.end()) {
        return as<Def_variable>(it->second);
    }

    llvm::Type *type      = cv->getType();
    Symbol     *cnst_sym  = Base::get_unique_sym(cv->getName(), "cnst");
    Type       *cnst_type = Base::convert_type(type);

    cnst_type = Base::m_tc.get_alias(cnst_type, Type::MK_CONST);

    // Note: HLSL does not support array specifiers on types while GLSL does.
    // So far, we always create the "C-Syntax", by adding the array specifiers
    // to the constant name
    Type_name *cnst_type_name = Base::get_type_name(inner_element_type(cnst_type));

    Declaration_variable *decl_cnst = Base::m_decl_factory.create_variable(cnst_type_name);

    Init_declarator *init_decl = Base::m_decl_factory.create_init_declarator(Base::zero_loc);
    Name *var_name = Base::get_name(Base::zero_loc, cnst_sym);
    init_decl->set_name(var_name);
    Base::add_array_specifiers(init_decl, cnst_type);
    decl_cnst->add_init(init_decl);

    init_decl->set_initializer(translate_constant_expr(cv, /*is_global=*/ false));

    Def_variable *cnst_def = Base::m_def_tab.enter_variable_definition(
        cnst_sym, cnst_type, &var_name->get_location());
    cnst_def->set_declaration(decl_cnst);
    var_name->set_definition(cnst_def);

    // so far, add all declarations to the function scope
    m_cur_start_block->add_stmt(Base::m_stmt_factory.create_declaration(decl_cnst));

    m_local_var_map[cv] = cnst_def;
    return cnst_def;
}

// Generates a new global static const variable to hold an LLVM value.
template<typename BasePass>
typename SLWriterPass<BasePass>::Definition *SLWriterPass<BasePass>::create_global_const(
    llvm::Constant *cv)
{
    auto it = m_global_const_map.find(cv);
    if (it != m_global_const_map.end()) {
        return it->second;
    }

    Expr       *c_expr   = translate_constant_expr(cv, /*is_global=*/ true);
    Definition *cnst_def = Base::create_global_const(cv->getName(), c_expr);

    m_global_const_map[cv] = cnst_def;
    return cnst_def;
}

// Get or create the in- and out-variables for an LLVM PHI node.
template<typename BasePass>
std::pair<
    typename SLWriterPass<BasePass>::Def_variable *,
    typename SLWriterPass<BasePass>::Def_variable *>
SLWriterPass<BasePass>::get_phi_vars(
    llvm::PHINode *phi,
    bool enter_in_var
    )
{
    auto it = m_phi_var_in_out_map.find(phi);
    if (it != m_phi_var_in_out_map.end()) {
        return it->second;
    }

#if NEW_OUT_OF_SSA
    // TODO: use debug info to generate a better name
    Symbol *in_sym = Base::get_unique_sym("phi_in", "phi_in_");
    Symbol *out_sym = nullptr;

    llvm::StringRef name = phi->getName();
    if (name.size() > 0) {
        out_sym = Base::get_unique_sym(name, "phi_");
    } else {
        out_sym = Base::get_unique_sym("phi", "phi_");
    }

    // TODO: arrays?
    llvm::Type   *type        = phi->getType();
    Def_variable *phi_in_def  = create_local_var(in_sym, type, enter_in_var);
    Def_variable *phi_out_def = create_local_var(out_sym, type, true);
#else
    // TODO: use debug info to generate a better name
    Symbol *in_sym  = Base::get_unique_sym("phi_in", "phi_in_");
    Symbol *out_sym = Base::get_unique_sym("phi_out", "phi_out_");

    bool do_enter = true;

    // TODO: arrays?
    llvm::Type   *type        = phi->getType();
    Def_variable *phi_in_def  = create_local_var(in_sym, type);
    Def_variable *phi_out_def = create_local_var(out_sym, type, do_enter);
#endif
    auto res = std::make_pair(phi_in_def, phi_out_def);
    m_phi_var_in_out_map[phi] = res;

    m_local_var_map[phi] = phi_out_def;

    return res;
}

// Get the definition of the in-variable of a PHI node.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_variable *SLWriterPass<BasePass>::get_phi_in_var(
    llvm::PHINode *phi,
    bool enter_in_var)
{
    return get_phi_vars(phi, enter_in_var).first;
}

// Get or create the symbol of the out-variable of a PHI node, where the in-variable
// will be written to at the start of a block.
template<typename BasePass>
typename SLWriterPass<BasePass>::Def_variable *SLWriterPass<BasePass>::get_phi_out_var(
    llvm::PHINode *phi)
{
    return get_phi_vars(phi).second;
}

// Check if the given AST statement is empty.
template<typename BasePass>
bool SLWriterPass<BasePass>::is_empty_statment(Stmt *stmt) {
    if (stmt == nullptr) {
        return true;
    }
    if (Stmt_expr *expr_stmt = as<Stmt_expr>(stmt)) {
        return expr_stmt->get_expression() == nullptr;
    }
    return false;
}

// Joins two statements into a compound statement.
// If one or both statements already are compound statements, they will be merged.
// If one statement is "empty" (nullptr), the other will be returned.
template<typename BasePass>
typename SLWriterPass<BasePass>::Stmt *SLWriterPass<BasePass>::join_statements(
    Stmt *head,
    Stmt *tail)
{
    if (is_empty_statment(head)) {
        return tail;
    }
    if (is_empty_statment(tail)) {
        return head;
    }

    Stmt_compound *block;
    if (head->get_kind() == Stmt::SK_COMPOUND) {
        block = cast<Stmt_compound>(head);
    } else {
        block = Base::m_stmt_factory.create_compound(Base::zero_loc);
        block->add_stmt(head);
    }
    if (tail->get_kind() == Stmt::SK_COMPOUND) {
        Stmt_compound *tail_list = cast<Stmt_compound>(tail);
        while (!tail_list->empty()) {
            Stmt *cur_stmt = tail_list->front();
            tail_list->pop_front();
            block->add_stmt(cur_stmt);
        }
    } else {
        block->add_stmt(tail);
    }
    return block;
}

template<typename BasePass>
char SLWriterPass<BasePass>::ID = 0;

// Creates a HLSL writer pass.
llvm::Pass *createHLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    Generated_code_source                                &code,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    mi::mdl::Options_impl const                          &options,
    mi::mdl::Messages_impl                               &messages,
    bool                                                 enable_debug,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
    mi::mdl::Function_remap const                        &func_remaps,
    bool                                                 enable_opt_remarks,
    bool                                                 enable_noinline_support)
{
    SLWriterPass<hlsl::HLSLWriterBasePass> *pass = new SLWriterPass<hlsl::HLSLWriterBasePass>(
        alloc,
        type_mapper,
        code,
        num_texture_spaces,
        num_texture_results,
        options,
        messages,
        enable_debug,
        exp_func_list,
        func_remaps,
        df_handle_slot_mode,
        enable_opt_remarks);

    pass->set_noinline_mode(
        enable_noinline_support ?
            hlsl::IPrinter::ATTR_NOINLINE_WRAP :
            hlsl::IPrinter::ATTR_NOINLINE_IGNORE);

    return pass;
}

// Creates a GLSL writer pass.
llvm::Pass *createGLSLWriterPass(
    mi::mdl::IAllocator                                  *alloc,
    Type_mapper const                                    &type_mapper,
    Generated_code_source                                &code,
    unsigned                                             num_texture_spaces,
    unsigned                                             num_texture_results,
    mi::mdl::Options_impl const                          &options,
    mi::mdl::Messages_impl                               &messages,
    bool                                                 enable_debug,
    mi::mdl::Df_handle_slot_mode                         df_handle_slot_mode,
    mi::mdl::LLVM_code_generator::Exported_function_list &exp_func_list,
    mi::mdl::Function_remap const                        &func_remaps)
{
    return new SLWriterPass<glsl::GLSLWriterBasePass>(
        alloc,
        type_mapper,
        code,
        num_texture_spaces,
        num_texture_results,
        options,
        messages,
        enable_debug,
        exp_func_list,
        func_remaps,
        df_handle_slot_mode,
        /*enable_opt_remarks=*/false);
}

}  // sl
}  // mdl
}  // mi
