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
#include "generator_jit_sl_writer.h"
#include "generator_jit_streams.h"
#include "generator_jit_sl_passes.h"

namespace mi {
namespace mdl {

// Compile the given module into HLSL or GLSL code.
void LLVM_code_generator::sl_compile(
    llvm::Module          *mod,
    Target_language       target,
    Options_impl const    &options,
    Generated_code_source &code)
{
    std::unique_ptr<llvm::Module> loaded_module;

    char const *load_env  = nullptr;
    char const *store_env = nullptr;

    switch (target) {
    case ICode_generator::TL_HLSL:
        load_env  = "MI_MDL_HLSL_LOAD_MODULE";
        store_env = "MI_MDL_HLSL_SAVE_MODULE";
        break;
    case ICode_generator::TL_GLSL:
        load_env  = "MI_MDL_GLSL_LOAD_MODULE";
        store_env = "MI_MDL_GLSL_SAVE_MODULE";
        break;
    default:
        MDL_ASSERT(!"unsupported target language in sl_compile()");
        return;
    }

    if (char const *load_module_name = getenv(load_env)) {
        llvm::SMDiagnostic err;
        loaded_module = std::move(llvm::parseIRFile(load_module_name, err, m_llvm_context));

        if (!loaded_module) {
            err.print("MDL", llvm::dbgs());
        } else {
            mod = loaded_module.get();

            llvm::dbgs() << "\nsl_compile: Loaded module from \"" << load_module_name << "\"!\n";
        }
    }

    if (char const *save_module_name = getenv(store_env)) {
        std::error_code ec;
        llvm::raw_fd_ostream file(save_module_name, ec);

        if (!ec) {
            llvm::legacy::PassManager mpm;
            mpm.add(llvm::createPrintModulePass(file));
            mpm.run(*mod);

            llvm::dbgs() <<
                "\nsl_compile: Saved input module to \"" << save_module_name << "\".\n";
        } else {
            llvm::dbgs() <<
                "\nsl_compile: Cannot save input to \"" << save_module_name << "\".\n";
        }
    }

    {
        llvm::legacy::PassManager mpm;

        if (target == ICode_generator::TL_GLSL) {
            mpm.add(llvm::sl::createRemoveBoolVectorOPsPass(*this));
        }

        mpm.add(llvm::createCFGSimplificationPass(     // must be executed before CNS
            llvm::SimplifyCFGOptions().avoidPointerPHIs(true)));
        mpm.add(llvm::sl::createControlledNodeSplittingPass());  // resolve irreducible CF
        mpm.add(llvm::createCFGSimplificationPass(     // eliminate dead blocks created by CNS
            llvm::SimplifyCFGOptions().avoidPointerPHIs(true)));
        mpm.add(llvm::sl::createUnswitchPass());       // get rid of all switch instructions
        mpm.add(llvm::createLoopSimplifyCFGPass());    // ensure all exit blocks are dominated by
                                                       // the loop header
        mpm.add(llvm::sl::createLoopExitEnumerationPass());  // ensure all loops have <= 1 exits
        mpm.add(llvm::sl::createUnswitchPass());       // get rid of all switch instructions
                                                       // introduced by the loop exit enumeration
        mpm.add(llvm::sl::createRemovePointerPHIsPass());
        mpm.add(llvm::sl::createHandlePointerSelectsPass());
        mpm.add(llvm::sl::createASTComputePass(m_type_mapper));
        switch (target) {
        case ICode_generator::TL_HLSL:
            mpm.add(sl::createHLSLWriterPass(
                get_allocator(),
                m_type_mapper,
                code,
                m_num_texture_spaces,
                m_num_texture_results,
                options,
                m_messages,
                m_enable_full_debug,
                m_link_libbsdf_df_handle_slot_mode,
                m_exported_func_list));
            break;
        case ICode_generator::TL_GLSL:
            mpm.add(sl::createGLSLWriterPass(
                get_allocator(),
                m_type_mapper,
                code,
                m_num_texture_spaces,
                m_num_texture_results,
                options,
                m_messages,
                m_enable_full_debug,
                m_link_libbsdf_df_handle_slot_mode,
                m_exported_func_list));
            break;
        default:
            MDL_ASSERT(!"unsupported target language in sl_compile()");
            return;
        }
        mpm.run(*mod);
    }
}

// Get the SL function suffix for the texture type in the first parameter of the given
// function definition.
char const *LLVM_code_generator::get_sl_tex_type_func_suffix(
    IDefinition const *tex_func_def)
{
    if (tex_func_def->get_semantics() == IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID) {
        // we have only one tex::isvalid() function in the SL runtime
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

// Get the intrinsic LLVM function for a MDL function for SL code.
llvm::Function *LLVM_code_generator::get_sl_intrinsic_function(
    IDefinition const *def,
    bool              return_derivs)
{
    char const *module_name = NULL;
    enum Module_enum {
        ME_OTHER,
        ME_DF,
        ME_SCENE,
        ME_TEX,
    } module_id = ME_OTHER;

    bool can_return_derivs = false;
    IDefinition::Semantics sema = def->get_semantics();
    if (is_tex_semantics(sema)) {
        module_name = "tex";
        module_id = ME_TEX;
    } else {
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            module_name = "df";
            module_id = ME_DF;
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
            module_id = ME_SCENE;
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
            module_id = ME_SCENE;
            break;

        default:
            break;
        }
    }

    // no special handling required?
    if (module_name == NULL) {
        return NULL;
    }

    Function_instance inst(get_allocator(), def, return_derivs);
    llvm::Function *func = get_function(inst);
    if (func != NULL) {
        return func;
    }

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
            get_sl_intrinsic_function(def, /*return_derivs=*/ false);
        llvm::SmallVector<llvm::Value *, 8> args;
        for (llvm::Function::arg_iterator ai = ctx.get_first_parameter(), ae = func->arg_end();
                ai != ae; ++ai) {
            args.push_back(ai);
        }

        llvm::Value *res = ctx->CreateCall(runtime_func, args);
        res = ctx.get_dual(res);
        ctx.create_return(res);
    } else {
        // return the external texture runtime function of the renderer
        LLVM_context_data *ctx_data = get_or_create_context_data(
            owner, inst, module_name, /*is_prototype=*/ true);
        func = ctx_data->get_function();

        char const *func_name = def->get_symbol()->get_name();

        if (module_id == ME_TEX) {
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
                        cast<IType_function>(def->get_type())->get_parameter(
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
            func->setName("tex_" + llvm::Twine(func_name) + get_sl_tex_type_func_suffix(def));
        } else if (module_id == ME_SCENE &&
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
                    runtime_func = &m_sl_funcs.m_scene_data_lookup_int;
                    runtime_func_name = "scene_data_lookup_int";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
                    runtime_func = &m_sl_funcs.m_scene_data_lookup_int2;
                    runtime_func_name = "scene_data_lookup_int2";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
                    runtime_func = &m_sl_funcs.m_scene_data_lookup_int3;
                    runtime_func_name = "scene_data_lookup_int3";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
                    runtime_func = &m_sl_funcs.m_scene_data_lookup_int4;
                    runtime_func_name = "scene_data_lookup_int4";
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
                    if (return_derivs) {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_deriv_float;
                        runtime_func_name = "scene_data_lookup_deriv_float";
                    } else {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_float;
                        runtime_func_name = "scene_data_lookup_float";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
                    if (return_derivs) {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_deriv_float2;
                        runtime_func_name = "scene_data_lookup_deriv_float2";
                    } else {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_float2;
                        runtime_func_name = "scene_data_lookup_float2";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
                    if (return_derivs) {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_deriv_float3;
                        runtime_func_name = "scene_data_lookup_deriv_float3";
                    } else {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_float3;
                        runtime_func_name = "scene_data_lookup_float3";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
                    if (return_derivs) {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_deriv_float4;
                        runtime_func_name = "scene_data_lookup_deriv_float4";
                    } else {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_float4;
                        runtime_func_name = "scene_data_lookup_float4";
                    }
                    break;
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
                case mi::mdl::IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
                    if (return_derivs) {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_deriv_color;
                        runtime_func_name = "scene_data_lookup_deriv_color";
                    } else {
                        runtime_func = &m_sl_funcs.m_scene_data_lookup_color;
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

                // let LLVM treat the function as a scalar to avoid duplicate calls
                (*runtime_func)->setDoesNotThrow();
                (*runtime_func)->setDoesNotAccessMemory();
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
        } else {
            func->setName(llvm::Twine(module_name) + "_" + func_name);
        }
    }

    // let LLVM treat the function as a scalar to avoid duplicate calls
    func->setDoesNotThrow();
    func->setDoesNotAccessMemory();

    return func;
}

}  // mdl
}  // mi
