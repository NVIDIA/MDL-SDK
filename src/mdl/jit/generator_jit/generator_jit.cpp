/******************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifdef PRINT_TIMINGS
#include <chrono>
#endif

#include <mi/base/handle.h>

#include <llvm/IR/Module.h>

#include "mdl/compiler/compilercore/compilercore_errors.h"
#include "mdl/compiler/compilercore/compilercore_mdl.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "mdl/compiler/compilercore/compilercore_hash.h"
#include "mdl/codegenerators/generator_dag/generator_dag_tools.h"

#include "generator_jit.h"
#include "generator_jit_code_printer.h"
#include "generator_jit_generated_code.h"
#include "generator_jit_llvm.h"
#include "generator_jit_opt_pass_gate.h"
#include "generator_jit_libbsdf_data.h"


namespace mi {
namespace mdl {

// Allow impl_cast on Link_unit
template<>
Link_unit_jit const *impl_cast(ILink_unit const *unit) {
    return static_cast<Link_unit_jit const *>(unit);
}

// Creates a JIT code generator.
Code_generator_jit *Code_generator_jit::create_code_generator(
    IAllocator *alloc,
    MDL        *mdl)
{
    Allocator_builder builder(alloc);

    return builder.create<Code_generator_jit>(alloc, mdl, mdl->get_jitted_code());
}

/// Create the default JIT code generator options.
static void fill_default_cg_options(
    Options_impl &options)
{
    options.add_option(
        MDL_JIT_OPTION_OPT_LEVEL,
        "2",
        "The optimization level of the JIT code generator");
    options.add_option(
        MDL_JIT_OPTION_FAST_MATH,
        "true",
        "Enables unsafe math optimizations of the JIT code generator");
    options.add_option(
        MDL_JIT_OPTION_INLINE_AGGRESSIVELY,
        "false",
        "Instructs the JIT code generator to aggressively inline functions");
    options.add_option(
        MDL_JIT_OPTION_EVAL_DAG_TERNARY_STRICTLY,
        "true",
        "Enable strict evaluation of the ternary operator on the DAG");
    options.add_option(
        MDL_JIT_OPTION_DISABLE_EXCEPTIONS,
        "false",
        "Disable exception handling in the generated code");
    options.add_option(
        MDL_JIT_OPTION_ENABLE_RO_SEGMENT,
        "false",
        "Enable the creation of a read-only data segment");
    options.add_option(
        MDL_JIT_OPTION_WRITE_BITCODE,
        "false",
        "Generate LLVM bitcode instead of LLVM IR code");
    options.add_option(
        MDL_JIT_OPTION_LINK_LIBDEVICE,
        "true",
        "Link libdevice into PTX module");
    options.add_option(
        MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE,
        "none",
        "Defines the libbsdf version to link into the output");
    options.add_option(
        MDL_JIT_OPTION_USE_BITANGENT,
        "false",
        "Use bitangent instead of tangent_u, tangent_v in the generated MDL core state");
    options.add_option(
        MDL_JIT_OPTION_REMAP_FUNCTIONS,
        "",
        "Function remap list in the form src1=dst1, src2=dst2, ...");
    options.add_option(
        MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE,
        "false",
        "Include the uniform state in the generated MDL core state");
    options.add_option(
        MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE,
        "vtable",
        "The mode for texture lookup functions on GPU (vtable, direct_call or optix_cp)");
    options.add_option(
        MDL_JIT_OPTION_MAP_STRINGS_TO_IDS,
        "false",
        "Map string constants to identifiers");
    options.add_option(
        MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES,
        "false",
        "The generated code should use texture lookup functions with derivative parameters "
        "for the texture coordinates");
    options.add_option(
        MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU,
        "true",
        "Use built-in resource handler on CPU");
    options.add_option(
        MDL_JIT_OPTION_USE_RENDERER_ADAPT_MICROFACET_ROUGHNESS,
        "false",
        "Use a renderer provided function to adapt microfacet roughness");
    options.add_option(
        MDL_JIT_OPTION_USE_RENDERER_ADAPT_NORMAL,
        "false",
        "Use a renderer provided function to adapt normals");
    options.add_option(
        MDL_JIT_OPTION_ENABLE_AUXILIARY,
        "false",
        "Enable code generation for auxiliary functions on DFs");
    options.add_option(
        MDL_JIT_OPTION_ENABLE_PDF,
        "true",
        "Enable code generation for PDF functions");
    options.add_option(
        MDL_JIT_OPTION_VISIBLE_FUNCTIONS,
        "",
        "Comma-separated list of names of functions which will be visible in the generated code "
        "(empty string means no special restriction).");

    // GLSL/HLSL specific options
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_NORMAL_MODE,
        "field",
        "The implementation mode of state::normal()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_NORMAL_NAME,
        "normal",
        "The name of the entity representing state::normal()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_GEOMETRY_NORMAL_MODE,
        "field",
        "The implementation mode of state::geometry_normal()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_GEOMETRY_NORMAL_NAME,
        "geometry_normal",
        "The name of the entity representingstate::geometry_normal()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_POSITION_MODE,
        "field",
        "The implementation mode of state::position()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_POSITION_NAME,
        "position",
        "The name of the entity representing state::position()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_MOTION_MODE,
        "zero",
        "The implementation mode of state::motion()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_MOTION_NAME,
        "motion",
        "The name of the entity representing state::motion()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_SPACE_MAX_MODE,
        "zero",
        "The implementation mode of state::texture_space_max()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_SPACE_MAX_NAME,
        "texture_space_max",
        "The name of the entity representing state::texture_space_max()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_ANIMATION_TIME_MODE,
        "zero",
        "The implementation mode of state::animation_time()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_ANIMATION_TIME_NAME,
        "animation_time",
        "The name of the entity representing state::animation_time()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_COORDINATE_MODE,
        "zero",
        "The implementation mode of state::texture_coordinate()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_COORDINATE_NAME,
        "texture_coordinate",
        "The name of the entity representing state::texture_coordinate()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_TANGENT_U_MODE,
        "zero",
        "The implementation mode of state::texture_tangent_u()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_TANGENT_U_NAME,
        "texture_tangent_u",
        "The name of the entity representing state::texture_tangent_u()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_TANGENT_V_MODE,
        "zero",
        "The implementation mode of state::texture_tangent_v()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TEXTURE_TANGENT_V_NAME,
        "texture_tangent_v",
        "The name of the entity representing state::texture_tangent_v()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_GEOMETRY_TANGENT_U_MODE,
        "zero",
        "The implementation mode of state::geometry_tangent_u()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_GEOMETRY_TANGENT_U_NAME,
        "geometry_tangent_u",
        "The name of the entity representing state::geometry_tangent_u()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_GEOMETRY_TANGENT_V_MODE,
        "zero",
        "The implementation mode of state::geometry_tangent_v()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_GEOMETRY_TANGENT_V_NAME,
        "geometry_tangent_v",
        "The name of the entity representing state::geometry_tangent_v()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TRANSFORM_W2O_MODE,
        "zero",
        "The implementation mode of the uniform world-to-object transform");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TRANSFORM_W2O_NAME,
        "w2o_transform",
        "The name of the entity representing the uniform world-to-object transform");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TRANSFORM_O2W_MODE,
        "zero",
        "The implementation mode of the uniform object-to-world transform");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_TRANSFORM_O2W_NAME,
        "o2w_transform",
        "The name of the entity representing the uniform object-to-world transform");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_OBJECT_ID_MODE,
        "zero",
        "The implementation mode of state::object_id()");
    options.add_option(
        MDL_JIT_OPTION_SL_STATE_OBJECT_ID_NAME,
        "object_id",
        "The name of the entity representing state::object_id()");
    options.add_option(
        MDL_JIT_OPTION_SL_USE_RESOURCE_DATA,
        "false",
        "GLSL/HLSL: Pass an extra user defined resource data struct to resource callbacks");
    options.add_option(
        MDL_JIT_OPTION_SL_CORE_STATE_API_NAME,
        "",
        "GLSL/HLSL: Name of the API Core State struct");
    options.add_option(
        MDL_JIT_OPTION_SL_ENV_STATE_API_NAME,
        "",
        "GLSL/HLSL: Name of the API Environment State struct");

    // GLSL specific options
    options.add_option(
        MDL_JIT_OPTION_GLSL_VERSION,
        "4.50",
        "GLSL: The target language version");

    options.add_option(
        MDL_JIT_OPTION_GLSL_PROFILE,
        "core",
        "GLSL: The target language profile");

    options.add_option(
        MDL_JIT_OPTION_GLSL_ENABLED_EXTENSIONS,
        "",
        "GLSL: Enabled GLSL extensions");

    options.add_option(
        MDL_JIT_OPTION_GLSL_REQUIRED_EXTENSIONS,
        "",
        "GLSL: Required GLSL extensions");

    options.add_option(
        MDL_JIT_OPTION_GLSL_MAX_CONST_DATA,
        "1024",
        "GLSL: Maximum allowed constant data in shader");

    options.add_option(
        MDL_JIT_OPTION_GLSL_PLACE_UNIFORMS_INTO_SSBO,
        "false",
        "GLSL: Use SSBO for uniform initializers");

    options.add_option(
        MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_NAME,
        "",
        "GLSL: If non-empty, the name of the SSBO buffer for uniform initializers");

    options.add_option(
        MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_BINDING,
        "",
        "GLSL: If non-empty, the binding index of the SSBO buffer for uniform initializers");

    options.add_option(
        MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_SET,
        "",
        "GLSL: If non-empty, the set index of the SSBO buffer for uniform initializers");

    options.add_binary_option(
        MDL_JIT_BINOPTION_LLVM_STATE_MODULE,
        "Use this user-specified LLVM implementation for the MDL state module");
    options.add_binary_option(
        MDL_JIT_BINOPTION_LLVM_RENDERER_MODULE,
        "Link and optimize this user-specified LLVM renderer module with the generated code");

    options.add_option(
        MDL_JIT_WARN_SPECTRUM_CONVERSION,
        "false",
        "Warn if a spectrum color is converted into an RGB value");
}

// Constructor.
Code_generator_thread_context::Code_generator_thread_context(
    IAllocator         *alloc,
    Options_impl const *options)
: Base(alloc)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_options(alloc, *options)
{
}

/// Access code generator messages of last operation.
Messages_impl const &Code_generator_thread_context::access_messages() const
{
    return m_msg_list;
}

// Access code generator messages of last operation.
Messages_impl &Code_generator_thread_context::access_messages()
{
    return m_msg_list;
}

// Access code generator options for the invocation.
Options_impl const &Code_generator_thread_context::access_options() const
{
    return m_options;
}

// Access code generator options for the invocation.
Options_impl &Code_generator_thread_context::access_options()
{
    return m_options;
}

// Constructor.
Code_generator_jit::Code_generator_jit(
    IAllocator  *alloc,
    MDL         *mdl,
    Jitted_code *jitted_code)
: Base(alloc, mdl)
, m_builder(alloc)
, m_jitted_code(mi::base::make_handle_dup(jitted_code))
{
    fill_default_cg_options(m_options);
}

// Get the name of the target language.
char const *Code_generator_jit::get_target_language() const
{
    return "executable";
}

// Creates a new thread context.
Code_generator_thread_context *Code_generator_jit::create_thread_context()
{
    return m_builder.create<Code_generator_thread_context>(m_builder.get_allocator(), &m_options);
}

// Acquires a const interface.
mi::base::IInterface const *Code_generator_jit::get_interface(
    mi::base::Uuid const &interface_id) const
{
    if (interface_id == IPrinter_interface::IID()) {
        return m_builder.create<JIT_code_printer>(m_builder.get_allocator());
    }
    return Base::get_interface(interface_id);
}

// Compile a whole module into LLVM-IR.
Generated_code_source *Code_generator_jit::compile_module_to_llvm(
    IModule const      *module,
    IModule_cache      *module_cache,
    Options_impl const &options)
{
    Module const *mod = impl_cast<Module>(module);
    mi::base::Handle<MDL> compiler(mod->get_compiler());

    Generated_code_source *result = m_builder.create<Generated_code_source>(
        m_builder.get_allocator(),
        IGenerated_code::CK_LLVM_IR);

    Messages_impl &msgs = result->access_messages();

    llvm::LLVMContext llvm_context;

    Generated_code_source::Source_res_manag res_manag(get_allocator(), NULL);

    LLVM_code_generator llvm_generator(
        m_jitted_code.get(),
        compiler.get(),
        module_cache,
        msgs,
        llvm_context,
        /*target_language=*/ICode_generator::TL_NATIVE,
        Type_mapper::TM_NATIVE_X86,
        /*sm_version=*/0,
        /*has_tex_handler=*/true,
        Type_mapper::SSM_FULL_SET,
        /*num_texture_spaces=*/4,
        /*num_texture_results=*/32,
        options,
        /*incremental=*/false,
        /*state_mapping=*/0,
        &res_manag,
        /*debug=*/false);

    // for now, mark exported functions as entries, so the module will not be empty
    llvm_generator.mark_exported_funcs_as_entries();

    std::unique_ptr<llvm::Module> llvm_module(llvm_generator.compile_module(module));

    if (llvm_module) {
        llvm_generator.llvm_ir_compile(llvm_module.get(), result->access_src_code());
    } else {
        size_t file_id = msgs.register_fname(module->get_filename());
        msgs.add_error_message(
            INTERNAL_COMPILER_ERROR,
            Generated_code_source::MESSAGE_CLASS,
            file_id,
            nullptr,
            "Compiling LLVM code failed");
    }

    result->set_render_state_usage(llvm_generator.get_render_state_usage());

    return result;
}

// Compile a whole module into PTX.
Generated_code_source *Code_generator_jit::compile_module_to_ptx(
    IModule const      *module,
    IModule_cache      *module_cache,
    Options_impl const &options)
{
    Module const *mod = impl_cast<Module>(module);
    mi::base::Handle<MDL> compiler(mod->get_compiler());

    Generated_code_source*result = m_builder.create<Generated_code_source>(
        m_builder.get_allocator(),
        IGenerated_code::CK_PTX);

    Messages_impl &msgs = result->access_messages();

    llvm::LLVMContext llvm_context;

    Generated_code_source::Source_res_manag res_manag(get_allocator(), NULL);

    unsigned sm_version = 20;
    LLVM_code_generator llvm_generator(
        m_jitted_code.get(),
        compiler.get(),
        module_cache,
        msgs,
        llvm_context,
        /*target_language=*/ICode_generator::TL_PTX,
        Type_mapper::TM_PTX,
        sm_version,
        /*has_tex_handler=*/false,
        Type_mapper::SSM_FULL_SET,
        /*num_texture_spaces=*/4,
        /*num_texture_results=*/32,
        options,
        /*incremental=*/false,
        /*state_mapping=*/0,
        &res_manag,
        /*debug=*/false);

    // for now, mark exported functions as entries, so the module will not be empty
    llvm_generator.mark_exported_funcs_as_entries();

    std::unique_ptr<llvm::Module> llvm_module(llvm_generator.compile_module(module));
    if (llvm_module) {
        // FIXME: pass the sm version here. However, this is currently used for debugging
        // only.
        llvm_generator.ptx_compile(llvm_module.get(), result->access_src_code());
    } else {
        size_t file_id = msgs.register_fname(module->get_filename());
        msgs.add_error_message(
            INTERNAL_COMPILER_ERROR,
            Generated_code_source::MESSAGE_CLASS,
            file_id,
            nullptr,
            "Compiling LLVM code failed");
    }

    result->set_render_state_usage(llvm_generator.get_render_state_usage());

    return result;
}

// Compile a whole MDL module into HLSL or GLSL.
Generated_code_source *Code_generator_jit::compile_module_to_sl(
    IModule const                    *imodule,
    IModule_cache                    *module_cache,
    ICode_generator::Target_language target,
    Options_impl const               &options)
{
    Module const *mod = impl_cast<Module>(imodule);
    mi::base::Handle<MDL> compiler(mod->get_compiler());

    Generated_code_source *result = m_builder.create<Generated_code_source>(
        m_builder.get_allocator(),
        target == ICode_generator::TL_HLSL ? IGenerated_code::CK_HLSL : IGenerated_code::CK_GLSL);

    Messages_impl &msgs = result->access_messages();

    llvm::LLVMContext llvm_context;
    SLOptPassGate opt_pass_gate;
    llvm_context.setOptPassGate(opt_pass_gate);

    Generated_code_source::Source_res_manag res_manag(get_allocator(), NULL);

    LLVM_code_generator llvm_generator(
        m_jitted_code.get(),
        compiler.get(),
        module_cache,
        msgs,
        llvm_context,
        target,
        target == ICode_generator::TL_HLSL ? Type_mapper::TM_HLSL : Type_mapper::TM_GLSL,
        /*sm_version=*/0,
        /*has_tex_handler=*/false,
        Type_mapper::SSM_FULL_SET,
        /*num_texture_spaces=*/4,
        /*num_texture_results=*/32,
        options,
        /*incremental=*/false,
        /*state_mapping=*/Type_mapper::SM_INCLUDE_UNIFORM_STATE,  // include uniform state for SL
        &res_manag,
        /*debug=*/false);

    // for now, mark exported functions as entries, so the module will not be empty
    llvm_generator.mark_exported_funcs_as_entries();

    // the HLSL backend expects the RO-data-segment to be used
    llvm_generator.enable_ro_data_segment();

    std::unique_ptr<llvm::Module> llvm_module(llvm_generator.compile_module(mod));
    if (llvm_module) {
        llvm_generator.sl_compile(llvm_module.get(), target, options, *result);
    } else {
        size_t file_id = msgs.register_fname(mod->get_filename());
        msgs.add_error_message(
            INTERNAL_COMPILER_ERROR,
            Generated_code_source::MESSAGE_CLASS,
            file_id,
            nullptr,
            "Compiling LLVM code failed");
    }

    result->set_render_state_usage(llvm_generator.get_render_state_usage());

    return result;
}

// Compile a whole module.
IGenerated_code_executable *Code_generator_jit::compile(
    IModule const                  *module,
    IModule_cache                  *module_cache,
    Target_language                target,
    ICode_generator_thread_context *ctx)
{
    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    switch (target) {
    case TL_NATIVE:
    case TL_LLVM_IR:
        return compile_module_to_llvm(module, module_cache, options);
    case TL_PTX:
        return compile_module_to_ptx(module, module_cache, options);
    case TL_HLSL:
    case TL_GLSL:
        return compile_module_to_sl(module, module_cache, target, options);
    }

    return nullptr;
}

// Compile a lambda function using the JIT into an environment (shader) of a scene.
IGenerated_code_lambda_function *Code_generator_jit::compile_into_environment(
    ILambda_function const         *ilambda,
    IModule_cache                  *module_cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx)
{
    return compile_into_generic_function(
        ilambda,
        module_cache,
        resolver,
        ctx,
        /*num_texture_spaces=*/ 0,
        /*num_texture_results=*/ 0,
        /*transformer=*/ NULL);
}

namespace {

/// Helper class to handle resources in const functions.
class Const_function_enumerator : public ILambda_resource_enumerator {
public:
    /// Constructor.
    ///
    /// \param attr    resource attribute requester interface
    /// \param lambda  currently processed lambda function
    Const_function_enumerator(
        ILambda_resource_attribute *attr,
        Lambda_function            &lambda)
    : m_attr(attr)
    , m_lambda(lambda)
    , m_tex_idx(0)
    , m_lp_idx(0)
    , m_bm_idx(0)
    {
    }

    /// Called for a texture resource.
    ///
    /// \param v  the texture resource or an invalid ref
    void texture(IValue const *v, Texture_usage tex_usage) MDL_FINAL
    {
        if (IValue_texture const *tex = as<IValue_texture>(v)) {
            bool valid = false;
            int width = 0, height = 0, depth = 0;
            int tag_value = tex->get_tag_value();
            IType_texture::Shape shape = tex->get_type()->get_shape();

            // FIXME: map the tag value using lambda's resource tag table
            m_attr->get_texture_attributes(tex, valid, width, height, depth);

            m_lambda.map_tex_resource(
                tex->get_kind(),
                tex->get_string_value(),
                tex->get_selector(),
                tex->get_gamma_mode(),
                tex->get_bsdf_data_kind(),
                shape,
                tag_value,
                m_tex_idx++,
                valid,
                width,
                height,
                depth);
        } else {
            m_lambda.map_tex_resource(
                v->get_kind(),
                /*res_url=*/NULL,
                /*res_sel=*/NULL,
                IValue_texture::gamma_default,
                IValue_texture::BDK_NONE,
                IType_texture::TS_2D,
                0,
                m_tex_idx++,
                /*valid=*/false,
                0,
                0,
                0);
        }
    }

    /// Called for a light profile resource.
    ///
    /// \param v  the light profile resource or an invalid ref
    void light_profile(IValue const *v) MDL_FINAL
    {
        if (IValue_resource const *r = as<IValue_resource>(v)) {
            bool valid = false;
            float power = 0.0f, maximum = 0.0f;
            int tag_value  = r->get_tag_value();

            // FIXME: map the tag value using lambda's resource tag table
            m_attr->get_light_profile_attributes(r, valid, power, maximum);

            m_lambda.map_lp_resource(
                r->get_kind(),
                r->get_string_value(),
                tag_value,
                m_lp_idx++,
                valid,
                power,
                maximum);
        } else {
            m_lambda.map_lp_resource(
                v->get_kind(),
                NULL,
                0,
                m_lp_idx++,
                /*valid=*/false,
                0.0f,
                0.0f);
        }
    }

    /// Called for a bsdf measurement resource.
    ///
    /// \param v  the bsdf measurement resource or an invalid_ref
    void bsdf_measurement(IValue const *v) MDL_FINAL
    {
        if (IValue_resource const *r = as<IValue_resource>(v)) {
            bool valid = false;
            int tag_value = r->get_tag_value();

            // FIXME: map the tag value using lambda's resource tag table
            m_attr->get_bsdf_measurement_attributes(r, valid);

            m_lambda.map_bm_resource(
                r->get_kind(),
                r->get_string_value(),
                tag_value,
                m_bm_idx++,
                valid);
        } else {
            m_lambda.map_bm_resource(v->get_kind(), NULL, 0, m_bm_idx++, /*valid=*/false);
        }
    }

private:
    /// The resource attribute requester.
    ILambda_resource_attribute *m_attr;

    /// The processed lambda function.
    Lambda_function            &m_lambda;

    /// Current texture index.
    size_t m_tex_idx;

    /// Current light profile index.
    size_t m_lp_idx;

    /// Current bsdf measurement index.
    size_t m_bm_idx;
};

} // anonymous

// Compile a lambda function using the JIT into a constant function.
IGenerated_code_lambda_function *Code_generator_jit::compile_into_const_function(
    ILambda_function const         *ilambda,
    IModule_cache                  *cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    ILambda_resource_attribute *attr,
    Float4_struct const            world_to_object[4],
    Float4_struct const            object_to_world[4],
    int                            object_id)
{
    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return NULL;
    }

    DAG_node const *body = lambda->get_body();
    if (body == NULL || lambda->get_root_expr_count() != 0) {
        // not a simple lambda
        return NULL;
    }

    if (lambda->may_use_varying_state(resolver, body)) {
        // currently we do not support any state access in const functions
        return NULL;
    }

    if (lambda->get_parameter_count() != 0) {
        // FIXME: Add support for class-compilation for const functions
        //    (const functions are not available via Neuray API, only material converter uses them)
        MDL_ASSERT(!"Class-compilation is not supported for const functions, yet");
        return NULL;
    }

    // FIXME: ugly, but ok for now: request all resource meta data through the attr interface
    // a better solution would be to do this outside this compile call
    Const_function_enumerator enumerator(attr, *const_cast<Lambda_function *>(lambda));
    lambda->enumerate_resources(*resolver, enumerator, body);

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    mi::base::Handle<Generated_code_lambda_function> code(
        builder.create<Generated_code_lambda_function>(m_jitted_code.get()));

    Generated_code_lambda_function::Lambda_res_manag res_manag(*code, /*resource_map=*/NULL);

    mi::base::Handle<MDL> compiler(lambda->get_compiler());

    // const function are evaluated on the CPU only
    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        cache,
        code->access_messages(),
        code->get_llvm_context(),
        ICode_generator::TL_NATIVE,
        Type_mapper::TM_NATIVE_X86,
        /*sm_version=*/0,
        /*has_tex_handler=*/options.get_bool_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU),
        Type_mapper::SSM_NO_STATE,
        /*num_texture_spaces=*/0,
        /*num_texture_results=*/0,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag,
        /*enable_debug=*/false);

    if (llvm::Function *func = code_gen.compile_const_lambda(
            *lambda, resolver, attr, world_to_object, object_to_world, object_id))
    {
        // remember the function name, because the function is not valid anymore after jit_compile
        string func_name(func->getName().begin(), func->getName().end(), alloc);

        MDL_JIT_module_key module_key = code_gen.jit_compile(func->getParent());
        code->set_llvm_module(module_key);
        code_gen.fill_function_info(code.get());

        // gen the entry point
        void *entry_point = code_gen.get_entry_point(module_key, func_name.c_str());
        code->add_entry_point(entry_point);

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling const function failed");
    }

    code->retain();
    return code.get();
}

// Compile a lambda switch function having several roots using the JIT into a
// function computing one of the root expressions.
IGenerated_code_lambda_function *Code_generator_jit::compile_into_switch_function(
    ILambda_function const         *ilambda,
    IModule_cache                  *cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results)
{
    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return NULL;
    }

    if (lambda->get_root_expr_count() < 1) {
        // there must be at least one root
        return NULL;
    }

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    // automatically activate deactivate the option if the state is set
    bool uses_ustate = lambda->is_uniform_state_set();
    options.set_option(
        MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, uses_ustate ? "false" : "true");

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    Generated_code_lambda_function *code =
        builder.create<Generated_code_lambda_function>(m_jitted_code.get());

    Resource_attr_map map(lambda->get_resource_attribute_map());
    Generated_code_lambda_function::Lambda_res_manag res_manag(
        *code, &map);

    mi::base::Handle<MDL> compiler(lambda->get_compiler());

    // switch functions are used in the core and for displacement, only in the later case
    // a texture handler is available
    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        cache,
        code->access_messages(),
        code->get_llvm_context(),
        ICode_generator::TL_NATIVE,
        Type_mapper::TM_NATIVE_X86,
        /*sm_version=*/0,
        /*has_tex_handler=*/lambda->get_execution_context() != ILambda_function::LEC_CORE,
        Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    // Enable the read-only data segment
    code_gen.enable_ro_data_segment();

    code_gen.set_resource_tag_map(&lambda->get_resource_tag_map());

    llvm::Function *func = code_gen.compile_switch_lambda(
        /*incremental=*/false, *lambda, resolver, /*next_arg_block_index=*/0);
    if (func != NULL) {
        // remember the function name, because the function is not valid anymore after jit_compile
        string func_name(func->getName().begin(), func->getName().end(), alloc);

        MDL_JIT_module_key module_key = code_gen.jit_compile(func->getParent());
        code->set_llvm_module(module_key);
        code_gen.fill_function_info(code);

        size_t data_size = 0;
        char const *data = reinterpret_cast<char const *>(code_gen.get_ro_segment(data_size));

        code->set_ro_segment(data, data_size);

        // gen the entry point
        void *entry_point = code_gen.get_entry_point(module_key, func_name.c_str());
        code->add_entry_point(entry_point);

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }

        // copy the string constant table.
        for (size_t i = 0, n = code_gen.get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(code_gen.get_string_constant(i), i);
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling switch function failed");
    }
    return code;
}

// Compile a lambda switch function having several roots using the JIT into a
// function computing one of the root expressions for execution on the GPU.
IGenerated_code_executable *Code_generator_jit::compile_into_switch_function_for_gpu(
    ICode_cache                    *code_cache,
    ILambda_function const         *ilambda,
    IModule_cache                  *module_cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results,
    unsigned                       sm_version)
{
    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return NULL;
    }

    if (lambda->get_root_expr_count() < 1) {
        // there must be at least one root
        return NULL;
    }

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    // automatically activate deactivate the option if the state is set
    bool uses_ustate = lambda->is_uniform_state_set();
    options.set_option(
        MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, uses_ustate ? "false" : "true");

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    Generated_code_source *code = builder.create<Generated_code_source>(
        alloc, IGenerated_code_executable::CK_PTX);

    unsigned char cache_key[16];

    if (code_cache != NULL) {
        MD5_hasher hasher;

        DAG_hash const *hash = lambda->get_hash();

        // set the generators name
        hasher.update("JIT");
        hasher.update(code->get_kind());

        hasher.update(lambda->get_name());
        hasher.update(hash->data(), hash->size());

        hasher.update(num_texture_spaces);
        hasher.update(num_texture_results);
        hasher.update(sm_version);

        // Beware: the selected options change the generated code, hence we must include them into
        // the key
        hasher.update(lambda->get_execution_context() == ILambda_function::LEC_ENVIRONMENT ?
            Type_mapper::SSM_ENVIRONMENT : Type_mapper::SSM_CORE);
        hasher.update(options.get_string_option(MDL_CG_OPTION_INTERNAL_SPACE));
        hasher.update(options.get_bool_option(MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT));
        hasher.update(options.get_float_option(MDL_CG_OPTION_METERS_PER_SCENE_UNIT));
        hasher.update(options.get_float_option(MDL_CG_OPTION_WAVELENGTH_MIN));
        hasher.update(options.get_float_option(MDL_CG_OPTION_WAVELENGTH_MAX));
        hasher.update(options.get_int_option(MDL_JIT_OPTION_OPT_LEVEL));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_FAST_MATH));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_INLINE_AGGRESSIVELY));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_EVAL_DAG_TERNARY_STRICTLY));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_LINK_LIBDEVICE));
        hasher.update(options.get_string_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS));

        hasher.final(cache_key);

        ICode_cache::Entry const *entry = code_cache->lookup(cache_key);

        if (entry != NULL) {
            // found a hit
            fill_code_from_cache(*ctx, code, entry);
            return code;
        }
    }

    Generated_code_source::Source_res_manag res_manag(alloc, &lambda->get_resource_attribute_map());

    llvm::LLVMContext llvm_context;
    mi::base::Handle<MDL> compiler(lambda->get_compiler());

    // GPU switch functions are used in the core only
    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        module_cache,
        code->access_messages(),
        llvm_context,
        ICode_generator::TL_PTX,
        Type_mapper::TM_PTX,
        sm_version,
        /*has_tex_handler=*/false,
        Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    // Enable the read-only data segment
    code_gen.enable_ro_data_segment();

    code_gen.set_resource_tag_map(&lambda->get_resource_tag_map());

    llvm::Function *func = code_gen.compile_switch_lambda(
        /*incremental=*/false, *lambda, resolver, /*next_arg_block_index=*/0);
    if (func != NULL) {
        llvm::Module *module = func->getParent();

        code_gen.ptx_compile(module, code->access_src_code());
        code_gen.fill_function_info(code);

        // it's now save to drop this module
        code_gen.drop_llvm_module(module);

        size_t data_size = 0;
        unsigned char const *data = code_gen.get_ro_segment(data_size);

        if (data_size > 0) {
            code->add_data_segment("RO", data, data_size);
        }

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }

        // copy the string constant table.
        for (size_t i = 0, n = code_gen.get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(code_gen.get_string_constant(i), i);
        }

        if (code_cache != NULL) {
            enter_code_into_cache(code, code_cache, cache_key);
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling GPU switch function failed");
    }
    return code;
}

// Compile a lambda function into a generic function using the JIT.
IGenerated_code_lambda_function *Code_generator_jit::compile_into_generic_function(
    ILambda_function const         *ilambda,
    IModule_cache                  *cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results,
    ILambda_call_transformer       *transformer)
{
    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return NULL;
    }

    if (lambda->get_body() == NULL || lambda->get_root_expr_count() != 0) {
        // not a simple lambda
        return NULL;
    }

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    // automatically activate deactivate the option if the state is set
    options.set_option(
        MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, lambda->is_uniform_state_set() ? "false" : "true");

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    mi::base::Handle<Generated_code_lambda_function> code(
        builder.create<Generated_code_lambda_function>(m_jitted_code.get()));

    Generated_code_lambda_function::Lambda_res_manag res_manag(*code, /*resource_map=*/NULL);

    // make sure, all registered resources are also known to the Lambda_res_manag
    res_manag.import_from_resource_attribute_map(&lambda->get_resource_attribute_map());

    mi::base::Handle<MDL> compiler(lambda->get_compiler());

    // generic functions are for CPU only
    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        cache,
        code->access_messages(),
        code->get_llvm_context(),
        ICode_generator::TL_NATIVE,
        Type_mapper::TM_NATIVE_X86,
        /*sm_version=*/0,
        /*has_tex_handler=*/options.get_bool_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU),
        lambda->get_execution_context() == ILambda_function::LEC_ENVIRONMENT ?
            Type_mapper::SSM_ENVIRONMENT : Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    // Enable the read-only data segment
    code_gen.enable_ro_data_segment();

    code_gen.set_resource_tag_map(&lambda->get_resource_tag_map());

    llvm::Function *func = code_gen.compile_lambda(
        /*incremental=*/false, *lambda, resolver, transformer, /*next_arg_block_index=*/0);
    if (func != NULL) {
        // remember the function name, because the function is not valid anymore after jit_compile
        string func_name(func->getName().begin(), func->getName().end(), alloc);

        MDL_JIT_module_key module_key = code_gen.jit_compile(func->getParent());
        code->set_llvm_module(module_key);
        code_gen.fill_function_info(code.get());

        size_t data_size = 0;
        char const *data = reinterpret_cast<char const *>(code_gen.get_ro_segment(data_size));

        code->set_ro_segment(data, data_size);

        // gen the entry point
        void *entry_point = code_gen.get_entry_point(module_key, func_name.c_str());
        code->add_entry_point(entry_point);

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }

        // copy the string constant table.
        for (size_t i = 0, n = code_gen.get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(code_gen.get_string_constant(i), i);
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(
            INTERNAL_JIT_BACKEND_ERROR,
            lambda->get_execution_context() == ILambda_function::LEC_ENVIRONMENT
                ? "Compiling environment function failed"
                : "Compiling generic function failed");
    }
    code->retain();
    return code.get();
}

// Compile a lambda function into a LLVM-IR using the JIT.
IGenerated_code_executable *Code_generator_jit::compile_into_llvm_ir(
    ILambda_function const         *ilambda,
    IModule_cache                  *cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results,
    bool                           enable_simd)
{
    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return NULL;
    }

    DAG_node const *body = lambda->get_body();
    if (body == NULL && lambda->get_root_expr_count() < 1) {
        // there must be at least one root or a body
        return NULL;
    }

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    // automatically activate deactivate the option if the state is set
    options.set_option(
        MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, lambda->is_uniform_state_set() ? "false" : "true");

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    Generated_code_source *code = builder.create<Generated_code_source>(
        alloc, IGenerated_code_executable::CK_LLVM_IR);

    Generated_code_source::Source_res_manag res_manag(alloc, &lambda->get_resource_attribute_map());

    llvm::LLVMContext llvm_context;
    mi::base::Handle<MDL> compiler(lambda->get_compiler());

    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        cache,
        code->access_messages(),
        llvm_context,
        ICode_generator::TL_NATIVE,
        enable_simd ? Type_mapper::TM_BIG_VECTORS : Type_mapper::TM_ALL_SCALAR,
        /*sm_version=*/0,
        /*has_tex_handler=*/options.get_bool_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU),
        Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    // enable name mangling
    code_gen.enable_name_mangling();

    // Enable the read-only data segment
    code_gen.enable_ro_data_segment();

    llvm::Function *func = NULL;
    if (body != NULL) {
        func = code_gen.compile_lambda(
            /*incremental=*/false,
            *lambda,
            resolver,
            /*transformer=*/NULL,
            /*next_arg_block_index=*/0);
    } else {
        func = code_gen.compile_switch_lambda(
            /*incremental=*/false,
            *lambda,
            resolver,
            /*next_arg_block_index=*/0);
    }
    if (func != NULL) {
        llvm::Module *module = func->getParent();

        if (options.get_bool_option(MDL_JIT_OPTION_WRITE_BITCODE)) {
            code_gen.llvm_bc_compile(module, code->access_src_code());
        } else {
            code_gen.llvm_ir_compile(module, code->access_src_code());
        }
        code_gen.fill_function_info(code);

        // it's now save to drop this module
        code_gen.drop_llvm_module(module);

        size_t data_size = 0;
        unsigned char const *data = code_gen.get_ro_segment(data_size);

        if (data_size > 0) {
            code->add_data_segment("RO", data, data_size);
        }

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }

        // copy the string constant table.
        for (size_t i = 0, n = code_gen.get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(code_gen.get_string_constant(i), i);
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling lambda function into LLVM IR failed");
    }
    return code;
}


// Compile a distribution function into native code using the JIT.
IGenerated_code_executable *Code_generator_jit::compile_distribution_function_cpu(
    IDistribution_function const   *idist_func,
    IModule_cache                  *cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results)
{
    Distribution_function const *dist_func = impl_cast<Distribution_function>(idist_func);

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());


    // always expect the uniform state to be part of the MDL SDK state structure
    options.set_option(MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, "true");

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    mi::base::Handle<Generated_code_lambda_function> code(
        builder.create<Generated_code_lambda_function>(m_jitted_code.get()));

    Generated_code_lambda_function::Lambda_res_manag res_manag(*code, /*resource_map=*/NULL);

    // make sure, all registered resources are also known to the Lambda_res_manag
    res_manag.import_from_resource_attribute_map(&dist_func->get_resource_attribute_map());

    mi::base::Handle<MDL> compiler(dist_func->get_compiler());

    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        cache,
        code->access_messages(),
        code->get_llvm_context(),
        TL_NATIVE,
        Type_mapper::TM_NATIVE_X86,
        /*sm_version=*/0,
        /*has_tex_handler=*/options.get_bool_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU),
        Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    LLVM_code_generator::Function_vector llvm_funcs(get_allocator());
    llvm::Module *module = code_gen.compile_distribution_function(
        /*incremental=*/false,
        *dist_func,
        resolver,
        llvm_funcs,
        /*next_arg_block_index=*/0,
        /*main_function_indices=*/NULL);

    if (module != NULL) {
        MDL_JIT_module_key module_key = code_gen.jit_compile(module);
        code->set_llvm_module(module_key);
        code_gen.fill_function_info(code.get());

        // add all generated functions (init, sample, evaluate, pdf) as entrypoints
        for (size_t i = 0, n = code->get_function_count(); i < n; ++i) {
            char const* func_name = code->get_function_name(i);
            code->add_entry_point(code_gen.get_entry_point(module_key, func_name));
        }

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling CPU DF function failed");
    }
    code->retain();
    return code.get();
}

// Compile a distribution function into a PTX, HLSL, or GLSL using the JIT.
IGenerated_code_executable *Code_generator_jit::compile_distribution_function_gpu(
    IDistribution_function const   *idist_func,
    IModule_cache                  *cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results,
    unsigned                       sm_version,
    Target_language                target,
    bool                           llvm_ir_output)
{
    Distribution_function const *dist_func = impl_cast<Distribution_function>(idist_func);

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    // always expect the uniform state to be part of the MDL SDK state structure
    options.set_option(MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, "true");

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    IGenerated_code_executable::Kind code_kind = IGenerated_code_executable::CK_EXECUTABLE;
    Type_mapper::Type_mapping_mode   tm_mode   = Type_mapper::TM_NATIVE_X86;

    switch (target) {
    case TL_PTX:
        code_kind = IGenerated_code_executable::CK_PTX;
        tm_mode   = Type_mapper::TM_PTX;
        break;
    case TL_HLSL:
        code_kind = IGenerated_code_executable::CK_HLSL;
        tm_mode   = Type_mapper::TM_HLSL;
        break;
    case TL_GLSL:
        code_kind = IGenerated_code_executable::CK_GLSL;
        tm_mode   = Type_mapper::TM_GLSL;
        break;
    case TL_NATIVE:
    case TL_LLVM_IR:
        MDL_ASSERT(!"Invalid compilation_mode for compile_distribution_function_gpu");
        return NULL;
    }

    if (llvm_ir_output) {
        // independent of the target language the code kind will be LLVM
        code_kind = IGenerated_code_executable::CK_LLVM_IR;
    }

    Generated_code_source *code = builder.create<Generated_code_source>(alloc, code_kind);

    Generated_code_source::Source_res_manag res_manag(
        alloc, &dist_func->get_resource_attribute_map());

    llvm::LLVMContext llvm_context;
    SLOptPassGate opt_pass_gate;
    if (target == TL_HLSL || target == TL_GLSL) {
        // disable several optimization if SL code generation is active
        llvm_context.setOptPassGate(opt_pass_gate);
    }

    mi::base::Handle<MDL> compiler(dist_func->get_compiler());

    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        cache,
        code->access_messages(),
        llvm_context,
        target,
        tm_mode,
        target == TL_PTX ? sm_version : 0,
        /*has_tex_handler=*/false,
        Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    // enable name mangling
    code_gen.enable_name_mangling();

    // Enable the read-only data segment
    code_gen.enable_ro_data_segment();

    LLVM_code_generator::Function_vector llvm_funcs(get_allocator());
    llvm::Module *mod = code_gen.compile_distribution_function(
        /*incremental=*/false,
        *dist_func,
        resolver,
        llvm_funcs,
        /*next_arg_block_index=*/0,
        /*main_function_indices=*/NULL);

    if (mod != NULL) {
        if (llvm_ir_output) {
            if (options.get_bool_option(MDL_JIT_OPTION_WRITE_BITCODE)) {
                code_gen.llvm_bc_compile(mod, code->access_src_code());
            } else {
                code_gen.llvm_ir_compile(mod, code->access_src_code());
            }
        } else {
            switch (target) {
            case TL_PTX:
                code_gen.ptx_compile(mod, code->access_src_code());
                break;
            case TL_HLSL:
            case TL_GLSL:
                code_gen.sl_compile(mod, target, options, *code);
                break;
            case TL_NATIVE:
            case TL_LLVM_IR:
                MDL_ASSERT(!"Invalid compilation_mode for compile_distribution_function_gpu");
                return NULL;
            }
        }
        code_gen.fill_function_info(code);

        // it's now save to drop this module
        code_gen.drop_llvm_module(mod);

        size_t data_size = 0;
        unsigned char const *data = code_gen.get_ro_segment(data_size);

        if (data_size > 0) {
            code->add_data_segment("RO", data, data_size);
        }

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            mi::base::Handle<Generated_code_value_layout> layout(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }

        // copy the string constant table.
        for (size_t i = 0, n = code_gen.get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(code_gen.get_string_constant(i), i);
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling GPU DF function failed");
    }
    return code;
}

// Fill a code object from a code cache entry.
void Code_generator_jit::fill_code_from_cache(
    ICode_generator_thread_context &ctx,
    Generated_code_source          *code,
    ICode_cache::Entry const       *entry)
{
    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    code->access_src_code() = string(entry->code, entry->code_size, alloc);

    if (entry->const_seg_size > 0) {
        code->add_data_segment("RO", (unsigned char *)entry->const_seg, entry->const_seg_size);
    }

    // only add a captured arguments layout, if it's non-empty
    if (entry->arg_layout_size != 0) {
        Options_impl &options = impl_cast<Options_impl>(ctx.access_options());

        mi::base::Handle<Generated_code_value_layout> layout(
            builder.create<Generated_code_value_layout>(
                alloc,
                entry->arg_layout,
                entry->arg_layout_size,
                options.get_bool_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS)));
        code->add_captured_arguments_layout(layout.get());
    }

    code->set_render_state_usage(entry->render_state_usage);

    // copy the string table if any
    for (size_t i = 0; i < entry->mapped_string_size; ++i) {
        code->add_mapped_string(entry->mapped_strings[i], i);
    }

    // copy function infos
    for (size_t i = 0; i < entry->func_info_size; ++i) {
        ICode_cache::Entry::Func_info const &info = entry->func_infos[i];
        size_t index = code->add_function_info(
            info.name, info.dist_kind, info.func_kind, info.arg_block_index, info.state_usage);

        for (int j = 0 ; j < int(IGenerated_code_executable::PL_NUM_LANGUAGES); ++j) {
            char const *prototype = info.prototypes[j];
            if (*prototype) {
                code->set_function_prototype(
                    index, IGenerated_code_executable::Prototype_language(j), prototype);
            }
        }
    }
}

// Enter a code object into the code cache.
void Code_generator_jit::enter_code_into_cache(
    Generated_code_source *code,
    ICode_cache           *code_cache,
    unsigned char const   cache_key[16])
{
    string const &code_str = code->access_src_code();

    size_t n_strings = code->get_string_constant_count();
    Small_VLA<char const *, 8> mapped_strings(get_allocator(), n_strings);
    for (size_t i = 0; i < n_strings; ++i) {
        mapped_strings[i] = code->get_string_constant(i);
    }

    size_t n_total_handles = 0;
    for (size_t i = 0, n = code->get_function_count(); i < n; ++i) {
        n_total_handles += code->get_function_df_handle_count(i);
    }

    // for simplicity, allocate at least one element
    Small_VLA<char const *, 8> handle_list(
        get_allocator(), n_total_handles > 0 ? n_total_handles : 1);
    size_t next_handle_index = 0;

    char const *empty_str = "";
    Small_VLA<ICode_cache::Entry::Func_info, 8> func_infos(
        get_allocator(), code->get_function_count());
    for (size_t i = 0, n = code->get_function_count(); i < n; ++i) {
        ICode_cache::Entry::Func_info &info = func_infos[i];
        info.name = code->get_function_name(i);
        info.dist_kind = code->get_distribution_kind(i);
        info.func_kind = code->get_function_kind(i);
        info.arg_block_index = code->get_function_arg_block_layout_index(i);

        for (int j = 0 ; j < int(IGenerated_code_executable::PL_NUM_LANGUAGES); ++j) {
            char const *prototype = code->get_function_prototype(
                i, IGenerated_code_executable::Prototype_language(j));
            info.prototypes[j] = prototype != NULL ? prototype : empty_str;
        }
        info.num_df_handles = code->get_function_df_handle_count(i);
        info.df_handles = &handle_list[next_handle_index];
        next_handle_index += info.num_df_handles;
        for (size_t j = 0; j < info.num_df_handles; ++j) {
            info.df_handles[j] = code->get_function_df_handle(i, j);
        }
    }

    size_t data_size = 0;
    char const *data = nullptr;

    if (IGenerated_code_executable::Segment const *ro_segment = code->get_data_segment(0)) {
        data      = (char const *)ro_segment->data;
        data_size = ro_segment->size;

        MDL_ASSERT(code->get_data_segment_count() == 1 && "can only cache one data segment");
    }

    char const *layout_data = nullptr;
    size_t layout_data_size = 0;
    if (code->get_captured_argument_layouts_count() > 0) {
        mi::base::Handle<IGenerated_code_value_layout const> i_layout(
            code->get_captured_arguments_layout(0));
        Generated_code_value_layout const *layout =
            impl_cast<Generated_code_value_layout>(i_layout.get());
        layout_data = layout->get_layout_data(layout_data_size);
    }

    ICode_cache::Entry entry(
        code_str.c_str(), code_str.size(),
        data, data_size,
        layout_data, layout_data_size,
        mapped_strings.data(), mapped_strings.size(),
        code->get_state_usage(),
        func_infos.data(), func_infos.size());

    code_cache->enter(cache_key, entry);
}

// Compile a lambda function into PTX or HLSL using the JIT.
IGenerated_code_executable *Code_generator_jit::compile_into_source(
    ICode_cache                    *code_cache,
    ILambda_function const         *ilambda,
    IModule_cache                  *module_cache,
    ICall_name_resolver const      *resolver,
    ICode_generator_thread_context *ctx,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results,
    unsigned                       sm_version,
    Target_language                target,
    bool                           llvm_ir_output)
{
    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return NULL;
    }

    DAG_node const *body = lambda->get_body();
    if (body == NULL && lambda->get_root_expr_count() < 1) {
        // there must be at least one root or a body
        return NULL;
    }

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    IGenerated_code_executable::Kind code_kind = IGenerated_code_executable::CK_EXECUTABLE;
    Type_mapper::Type_mapping_mode   tm_mode   = Type_mapper::TM_NATIVE_X86;

    switch (target) {
    case TL_PTX:
        code_kind = IGenerated_code_executable::CK_PTX;
        tm_mode = Type_mapper::TM_PTX;
        break;
    case TL_HLSL:
        code_kind = IGenerated_code_executable::CK_HLSL;
        tm_mode = Type_mapper::TM_HLSL;
        break;
    case TL_GLSL:
        code_kind = IGenerated_code_executable::CK_GLSL;
        tm_mode = Type_mapper::TM_GLSL;
        break;
    case TL_NATIVE:
    case TL_LLVM_IR:
        MDL_ASSERT(!"Invalid compilation_mode for compile_distribution_function_gpu");
        return NULL;
    }

    if (llvm_ir_output) {
        // independent of the target language the code kind will be LLVM
        code_kind = IGenerated_code_executable::CK_LLVM_IR;
    }

    Generated_code_source *code = builder.create<Generated_code_source>(alloc, code_kind);

    unsigned char cache_key[16];

    if (code_cache != NULL) {
        MD5_hasher hasher;

        DAG_hash const *hash = lambda->get_hash();

        // set the generators name
        hasher.update("JIT");
        hasher.update(code->get_kind());

        hasher.update(lambda->get_name());
        hasher.update(hash->data(), hash->size());
        if (target == TL_PTX) {
            hasher.update(sm_version);
        }
        hasher.update(llvm_ir_output);

        // Beware: the selected options change the generated code, hence we must include them into
        // the key
        hasher.update(lambda->get_execution_context() == ILambda_function::LEC_ENVIRONMENT ?
            Type_mapper::SSM_ENVIRONMENT : Type_mapper::SSM_CORE);
        hasher.update(num_texture_spaces);
        hasher.update(num_texture_results);
        hasher.update(options.get_string_option(MDL_CG_OPTION_INTERNAL_SPACE));
        hasher.update(options.get_bool_option(MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT));
        hasher.update(options.get_float_option(MDL_CG_OPTION_METERS_PER_SCENE_UNIT));
        hasher.update(options.get_float_option(MDL_CG_OPTION_WAVELENGTH_MIN));
        hasher.update(options.get_float_option(MDL_CG_OPTION_WAVELENGTH_MAX));
        hasher.update(options.get_int_option(MDL_JIT_OPTION_OPT_LEVEL));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_FAST_MATH));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_INLINE_AGGRESSIVELY));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_EVAL_DAG_TERNARY_STRICTLY));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_LINK_LIBDEVICE));
        hasher.update(options.get_string_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE));
        hasher.update(options.get_bool_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS));

        if (code_kind == IGenerated_code_executable::CK_GLSL ||
            code_kind == IGenerated_code_executable::CK_HLSL)
        {
            hasher.update(options.get_bool_option(MDL_JIT_OPTION_SL_USE_RESOURCE_DATA));
        }

        hasher.final(cache_key);

        ICode_cache::Entry const *entry = code_cache->lookup(cache_key);

        if (entry != NULL) {
            // found a hit
            fill_code_from_cache(*ctx, code, entry);
            return code;
        }
    }

    // automatically activate deactivate the option if the state is set
    options.set_option(
        MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, lambda->is_uniform_state_set() ? "false" : "true");

    Generated_code_source::Source_res_manag res_manag(alloc, &lambda->get_resource_attribute_map());

    llvm::LLVMContext llvm_context;
    SLOptPassGate opt_pass_gate;
    if (target == TL_HLSL || target == TL_GLSL) {
        // disable several optimization if SL code generation is active
        llvm_context.setOptPassGate(opt_pass_gate);
    }

    mi::base::Handle<MDL> compiler(lambda->get_compiler());

    LLVM_code_generator code_gen(
        m_jitted_code.get(),
        compiler.get(),
        module_cache,
        code->access_messages(),
        llvm_context,
        target,
        tm_mode,
        target == TL_PTX ? sm_version : 0,
        /*has_tex_handler=*/false,
        lambda->get_execution_context() == ILambda_function::LEC_ENVIRONMENT ?
            Type_mapper::SSM_ENVIRONMENT : Type_mapper::SSM_CORE,
        num_texture_spaces,
        num_texture_results,
        options,
        /*incremental=*/false,
        get_state_mapping(options),
        &res_manag, /*enable_debug=*/false);

    // enable name mangling
    code_gen.enable_name_mangling();

    // Enable the read-only data segment
    code_gen.enable_ro_data_segment();

    llvm::Function *func = NULL;
    if (body != NULL) {
        func = code_gen.compile_lambda(
            /*incremental=*/false,
            *lambda,
            resolver,
            /*transformer=*/NULL,
            /*next_arg_block_index=*/0);
    } else {
        func = code_gen.compile_switch_lambda(
            /*incremental=*/false,
            *lambda,
            resolver,
            /*next_arg_block_index=*/0);
    }
    if (func != NULL) {
        llvm::Module *mod = func->getParent();

        if (llvm_ir_output) {
            if (options.get_bool_option(MDL_JIT_OPTION_WRITE_BITCODE)) {
                code_gen.llvm_bc_compile(mod, code->access_src_code());
            } else {
                code_gen.llvm_ir_compile(mod, code->access_src_code());
            }
        } else {
            switch (target) {
            case TL_PTX:
                code_gen.ptx_compile(mod, code->access_src_code());
                break;
            case TL_HLSL:
            case TL_GLSL:
                code_gen.sl_compile(mod, target, options, *code);
                break;
            default:
                MDL_ASSERT(!"unexpected target language");
            }
        }
        code_gen.fill_function_info(code);

        // it's now save to drop this module
        code_gen.drop_llvm_module(mod);

        // copy the read-only segment
        size_t data_size = 0;
        unsigned char const *data = code_gen.get_ro_segment(data_size);
        if (data_size > 0) {
            code->add_data_segment("RO", data, data_size);
        }

        // copy the render state usage
        code->set_render_state_usage(code_gen.get_render_state_usage());

        // create the argument block layout if any arguments are captured
        mi::base::Handle<Generated_code_value_layout> layout;
        if (code_gen.get_captured_arguments_llvm_type() != NULL) {
            layout = mi::base::make_handle(
                builder.create<Generated_code_value_layout>(alloc, &code_gen));
            code->add_captured_arguments_layout(layout.get());
        }

        // copy the string constant table.
        size_t n_strings = code_gen.get_string_constant_count();
        for (size_t i = 0; i < n_strings; ++i) {
            code->add_mapped_string(code_gen.get_string_constant(i), i);
        }

        if (code_cache != NULL) {
            enter_code_into_cache(code, code_cache, cache_key);
        }
    } else if (code->access_messages().get_error_message_count() == 0) {
        // on failure, ensure that the code contains an error message
        code_gen.error(INTERNAL_JIT_BACKEND_ERROR, "Compiling lambda function into source failed");
    }
    return code;
}


// Get the device library for PTX compilation for the given target architecture.
unsigned char const *Code_generator_jit::get_libdevice_for_gpu(
    size_t &size)
{
    unsigned min_ptx_version = 0;
    return LLVM_code_generator::get_libdevice(size, min_ptx_version);
}

// Get the resolution of the libbsdf multi-scattering lookup table data.
bool Code_generator_jit::get_libbsdf_multiscatter_data_resolution(
    IValue_texture::Bsdf_data_kind bsdf_data_kind,
    size_t                         &out_theta,
    size_t                         &out_roughness,
    size_t                         &out_ior) const
{
    return libbsdf_data::get_libbsdf_multiscatter_data_resolution(
        bsdf_data_kind, out_theta, out_roughness, out_ior);
}

// Get access to the libbsdf multi-scattering lookup table data.
unsigned char const *Code_generator_jit::get_libbsdf_multiscatter_data(
    IValue_texture::Bsdf_data_kind bsdf_data_kind,
    size_t                         &size) const
{
    return libbsdf_data::get_libbsdf_multiscatter_data(bsdf_data_kind, size);
}

// Create a link unit.
Link_unit_jit *Code_generator_jit::create_link_unit(
    ICode_generator_thread_context *ctx,
    Target_language                target,
    bool                           enable_simd,
    unsigned                       sm_version,
    unsigned                       num_texture_spaces,
    unsigned                       num_texture_results)
{
    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    // link units always expect the uniform state to be included in the MDL SDK state
    options.set_option(MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, "true");

    Type_mapper::Type_mapping_mode tm_mode;
    switch (target) {
    case TL_PTX:
        tm_mode = Type_mapper::TM_PTX;
        break;

    case TL_LLVM_IR:
        tm_mode = enable_simd ? Type_mapper::TM_BIG_VECTORS : Type_mapper::TM_ALL_SCALAR;
        break;

    case TL_NATIVE:
        tm_mode = Type_mapper::TM_NATIVE_X86;
        break;

    case TL_HLSL:
        tm_mode = Type_mapper::TM_HLSL;
        break;

    case TL_GLSL:
        tm_mode = Type_mapper::TM_GLSL;
        break;

    default:
        MDL_ASSERT(!"unsupported target languange");
        return NULL;
    }

    return m_builder.create<Link_unit_jit>(
        m_builder.get_allocator(),
        m_jitted_code.get(),
        m_compiler.get(),
        target,
        tm_mode,
        sm_version,
        num_texture_spaces,
        num_texture_results,
        &options,
        get_state_mapping(options),
        /*enable_debug=*/false
    );
}

// Compile a link unit into a LLVM-IR using the JIT.
IGenerated_code_executable *Code_generator_jit::compile_unit(
    ICode_generator_thread_context *ctx,
    IModule_cache                  *module_cache,
    ILink_unit const               *iunit,
    bool                           llvm_ir_output)
{
    if (iunit == NULL) {
        return NULL;
    }
    size_t num_funcs = iunit->get_function_count();
    if (num_funcs == 0) {
        return NULL;
    }

    mi::base::Handle<Code_generator_thread_context> tmp;
    if (ctx == NULL) {
        tmp = mi::base::make_handle(create_thread_context());
        ctx = tmp.get();
    }

    Options_impl &options = impl_cast<Options_impl>(ctx->access_options());

    Link_unit_jit &unit = *const_cast<Link_unit_jit *>(impl_cast<Link_unit_jit>(iunit));

    IAllocator        *alloc = get_allocator();
    Allocator_builder builder(alloc);

    // pass the resource to tag map to the code generator
    unit->set_resource_tag_map(unit.get_resource_tag_map());

#ifdef PRINT_TIMINGS
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

    // now finalize the module
    llvm::Module *llvm_module = unit.finalize_module(module_cache);
    mi::base::Handle<IGenerated_code_executable> code_obj(unit.get_code_object());

#ifdef PRINT_TIMINGS
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t3 = t2;
    std::chrono::steady_clock::time_point t4 = t2;
    std::chrono::steady_clock::time_point t5 = t2;
    std::chrono::steady_clock::time_point t6 = t2;
#endif

    if (llvm_module == NULL) {
        // on failure, ensure that the code contains an error message
        if (unit->get_error_message_count() == 0) {
            unit->error(INTERNAL_JIT_BACKEND_ERROR, "Compiling link unit failed");
        }
    } else if (unit.get_target_language() == ICode_generator::TL_NATIVE) {
        // generate executable code
        mi::base::Handle<Generated_code_lambda_function> code(
            code_obj->get_interface<mi::mdl::Generated_code_lambda_function>());

#ifdef PRINT_TIMINGS
        t3 = std::chrono::steady_clock::now();
#endif
        MDL_JIT_module_key module_key = unit->jit_compile(llvm_module);
#ifdef PRINT_TIMINGS
        t4 = std::chrono::steady_clock::now();
#endif
        code->set_llvm_module(module_key);
        unit->fill_function_info(code.get());
#ifdef PRINT_TIMINGS
        t5 = std::chrono::steady_clock::now();
#endif

        // add all generated functions as entry points
        for (size_t i = 0; i < num_funcs; ++i) {
            char const *func_name = unit.get_function_name(i);
            code->add_entry_point(unit->get_entry_point(module_key, func_name));
        }
#ifdef PRINT_TIMINGS
        t6 = std::chrono::steady_clock::now();
#endif

        // copy the render state usage
        code->set_render_state_usage(unit->get_render_state_usage());

        // add all argument block layouts
        for (size_t i = 0, num = unit.get_arg_block_layout_count(); i < num; ++i) {
            code->add_captured_arguments_layout(
                mi::base::make_handle(unit.get_arg_block_layout(i)).get());
        }

        // copy the string constant table
        for (size_t i = 0, n = unit->get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(unit->get_string_constant(i), i);
        }
    } else {
        // generate source code
        mi::base::Handle<Generated_code_source> code(
            code_obj->get_interface<mi::mdl::Generated_code_source>());

        Target_language target = unit.get_target_language();
	if (llvm_ir_output || target == ICode_generator::TL_LLVM_IR) {
            if (options.get_bool_option(MDL_JIT_OPTION_WRITE_BITCODE)) {
                unit->llvm_bc_compile(llvm_module, code->access_src_code());
            } else {
                unit->llvm_ir_compile(llvm_module, code->access_src_code());
            }
	} else {
            switch (target) {
            case ICode_generator::TL_PTX:
                unit->ptx_compile(llvm_module, code->access_src_code());
                break;
            case ICode_generator::TL_HLSL:
            case ICode_generator::TL_GLSL:
                unit->sl_compile(llvm_module, target, options, *code);
                break;
            case ICode_generator::TL_LLVM_IR:
            default:
                MDL_ASSERT(!"unexpected target kind");
                break;
            }
	}
        unit->fill_function_info(code.get());

        // set the read-only data segment
        size_t data_size = 0;
        unsigned char const *data = unit->get_ro_segment(data_size);
        if (data_size > 0) {
            code->add_data_segment("RO", data, data_size);
        }

        // copy the render state usage
        code->set_render_state_usage(unit->get_render_state_usage());

        // add all argument block layouts
        for (size_t i = 0, num = unit.get_arg_block_layout_count(); i < num; ++i) {
            code->add_captured_arguments_layout(
                mi::base::make_handle(unit.get_arg_block_layout(i)).get());
        }

        // copy the string constant table
        for (size_t i = 0, n = unit->get_string_constant_count(); i < n; ++i) {
            code->add_mapped_string(unit->get_string_constant(i), i);
        }

        // it's now safe to drop this module
        delete llvm_module;
    }

#ifdef PRINT_TIMINGS
    std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();

    std::chrono::duration<double> et = t7 - t1;
    printf("CU  |||| Total time                 : %f seconds.\n", et.count());

    et = t2 - t1;
    printf("CU  | Finalize module               : %f seconds.\n", et.count());

    et = t3 - t2;
    printf("CU  | Before jit_compile            : %f seconds.\n", et.count());

    et = t4 - t3;
    printf("CU  | jit_compile                   : %f seconds.\n", et.count());

    et = t5 - t4;
    printf("CU  | Before add entrypoints        : %f seconds.\n", et.count());

    et = t6 - t5;
    printf("CU  | Add entrypoints               : %f seconds.\n", et.count());

    et = t7 - t6;
    printf("CU  | Rest                          : %f seconds.\n", et.count());
#endif

    code_obj->retain();
    return code_obj.get();
}

/// Create a blank layout used for deserialization of target codes.
IGenerated_code_value_layout *Code_generator_jit::create_value_layout() const
{
    return m_builder.create<mi::mdl::Generated_code_value_layout>(
        this->get_allocator());
}

// Calculate the state mapping mode from options.
unsigned Code_generator_jit::get_state_mapping(
    Options_impl const &options)
{
    unsigned res = 0;

    if (options.get_bool_option(MDL_JIT_OPTION_USE_BITANGENT)) {
        res |= Type_mapper::SM_USE_BITANGENT;
    }

    if (options.get_bool_option(MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE)) {
        res |= Type_mapper::SM_INCLUDE_UNIFORM_STATE;
    }

    return res;
}

// Constructor.
Link_unit_jit::Link_unit_jit(
    IAllocator         *alloc,
    Jitted_code        *jitted_code,
    MDL                *compiler,
    Target_language    target_language,
    Type_mapping_mode  tm_mode,
    unsigned           sm_version,
    unsigned           num_texture_spaces,
    unsigned           num_texture_results,
    Options_impl const *options,
    unsigned           state_mapping,
    bool               enable_debug)
: Base(alloc)
, m_arena(alloc)
, m_target_lang(target_language)
, m_opt_pass_gate()
, m_source_only_llvm_context()
, m_code(create_code_object(jitted_code))
, m_code_gen(
    jitted_code,
    compiler,
    /*module_cache=*/NULL,
    access_messages(),
    *get_llvm_context(),
    target_language,
    tm_mode,
    sm_version,
    /*has_tex_handler=*/
    target_language == ICode_generator::TL_NATIVE ||
    target_language == ICode_generator::TL_LLVM_IR ?
        options->get_bool_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU) :
        false, // PTX, HLSL and GLSL code cannot have a tex handler
    Type_mapper::SSM_CORE,
    num_texture_spaces,
    num_texture_results,
    *options,
    /*incremental=*/true,
    state_mapping,
    /*res_manag=*/NULL,
    enable_debug)
, m_resource_attr_map(alloc)
, m_res_manag(create_resource_manager(m_code.get(), options->get_bool_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU)))
, m_arg_block_layouts(alloc)
, m_lambdas(alloc)
, m_dist_funcs(alloc)
, m_resource_tag_map(alloc)
{
    // For native code, we don't need mangling and read-only data segments
    if (m_target_lang != ICode_generator::TL_NATIVE) {
        // enable name mangling
        m_code_gen.enable_name_mangling();

        // enable the read-only data segment
        m_code_gen.enable_ro_data_segment();

        // disable several optimization if SL code generation is active
        if (m_target_lang == ICode_generator::TL_HLSL ||
            m_target_lang == ICode_generator::TL_GLSL)
        {
            m_source_only_llvm_context.setOptPassGate(m_opt_pass_gate);
        }
    }

    m_code_gen.set_resource_manag(m_res_manag);
}

// Destructor.
Link_unit_jit::~Link_unit_jit()
{
    Allocator_builder builder(get_allocator());

    // Note: the IResource_manager does not have an virtual destructor, hence cast
    // to the right type
    switch (m_target_lang) {
    case ICode_generator::TL_NATIVE:
        {
            Generated_code_lambda_function::Lambda_res_manag *res_manag =
                static_cast<Generated_code_lambda_function::Lambda_res_manag *>(m_res_manag);

            builder.destroy(res_manag);
        }
        break;
    case ICode_generator::TL_PTX:
    case ICode_generator::TL_HLSL:
    case ICode_generator::TL_GLSL:
    case ICode_generator::TL_LLVM_IR:
        {
            Generated_code_source::Source_res_manag *res_manag =
                static_cast<Generated_code_source::Source_res_manag *>(m_res_manag);

            builder.destroy(res_manag);
        }
        break;
    }
}

// Creates the code object to be used with this link unit.
IGenerated_code_executable *Link_unit_jit::create_code_object(
    Jitted_code *jitted_code)
{
    Allocator_builder builder(get_allocator());
    switch (m_target_lang) {
    case ICode_generator::TL_NATIVE:
        return builder.create<Generated_code_lambda_function>(jitted_code);
    case ICode_generator::TL_PTX:
        return builder.create<Generated_code_source>(
            get_allocator(), IGenerated_code_executable::CK_PTX);
    case ICode_generator::TL_HLSL:
        return builder.create<Generated_code_source>(
            get_allocator(), IGenerated_code_executable::CK_HLSL);
    case ICode_generator::TL_GLSL:
        return builder.create<Generated_code_source>(
            get_allocator(), IGenerated_code_executable::CK_GLSL);
    case ICode_generator::TL_LLVM_IR:
        return builder.create<Generated_code_source>(
            get_allocator(), IGenerated_code_executable::CK_LLVM_IR);
    }
    MDL_ASSERT(!"unsupported target kind");
    return NULL;
}

// Creates the resource manager to be used with this link unit.
IResource_manager *Link_unit_jit::create_resource_manager(
    IGenerated_code_executable *icode,
    bool use_builtin_resource_handler_cpu)
{
    Allocator_builder builder(get_allocator());
    switch (m_target_lang) {
    case ICode_generator::TL_NATIVE:
        {
            // We need to build this manually, as Allocator_builder doesn't support reference
            // parameters in create<>.
            Generated_code_lambda_function::Lambda_res_manag *res_manag =
                builder.alloc<Generated_code_lambda_function::Lambda_res_manag>(1);
            Generated_code_lambda_function *code =
                static_cast<Generated_code_lambda_function *>(icode);
            new (res_manag) Generated_code_lambda_function::Lambda_res_manag(
                *code, /*resource_map=*/use_builtin_resource_handler_cpu ? NULL : &m_resource_attr_map);
            return res_manag;
        }
    case ICode_generator::TL_PTX:
    case ICode_generator::TL_HLSL:
    case ICode_generator::TL_GLSL:
    case ICode_generator::TL_LLVM_IR:
        return builder.create<Generated_code_source::Source_res_manag>(
            get_allocator(), (mi::mdl::Resource_attr_map const *)NULL);
    }
    MDL_ASSERT(!"unsupported target kind");
    return NULL;
}

// Update the resource attribute maps for the current lambda function to be compiled.
void Link_unit_jit::update_resource_attribute_map(
    Lambda_function const *lambda)
{
    switch (m_target_lang) {
    case ICode_generator::TL_NATIVE:
        {
            Generated_code_lambda_function::Lambda_res_manag *res_manag =
                static_cast<Generated_code_lambda_function::Lambda_res_manag *>(m_res_manag);

            res_manag->import_from_resource_attribute_map(&lambda->get_resource_attribute_map());
        }
        break;
    case ICode_generator::TL_PTX:
    case ICode_generator::TL_HLSL:
    case ICode_generator::TL_GLSL:
    case ICode_generator::TL_LLVM_IR:
        {
            Generated_code_source::Source_res_manag *res_manag =
                static_cast<Generated_code_source::Source_res_manag *>(m_res_manag);

            res_manag->import_resource_attribute_map(&lambda->get_resource_attribute_map());
        }
        break;
    }

    // also ensure that the tags are handled right
    update_resource_tag_map(lambda);
}

// Update the resource map for the current lambda function to be compiled.
void Link_unit_jit::update_resource_tag_map(
    Lambda_function const *lambda)
{
    for (size_t i = 0, n = lambda->get_resource_entries_count(); i < n; ++i) {
        Resource_tag_tuple const *e = lambda->get_resource_entry(i);

        int old_tag = find_resource_tag(e->m_kind, e->m_url, e->m_selector);
        if (old_tag == 0) {
            add_resource_tag_mapping(e->m_kind, e->m_url, e->m_selector, e->m_tag);
        } else {
            MDL_ASSERT(old_tag == e->m_tag && "Tag mismatch in resource table");
        }
    }
}

// Find the assigned tag for a resource in the resource map.
int Link_unit_jit::find_resource_tag(
    Resource_tag_tuple::Kind kind,
    char const               *url,
    char const               *sel) const
{
    // linear search
    for (size_t i = 0, n = m_resource_tag_map.size(); i < n; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        if (e.m_kind== kind && strcmp(e.m_url, url) == 0 && strcmp(e.m_selector, sel) == 0) {
            return e.m_tag;
        }
    }
    return 0;
}

// Add a new entry in the resource map.
void Link_unit_jit::add_resource_tag_mapping(
    Resource_tag_tuple::Kind kind,
    char const               *url,
    char const               *sel,
    int                      tag)
{
    url = url != NULL ? Arena_strdup(m_arena, url) : NULL;
    sel = sel != NULL ? Arena_strdup(m_arena, sel) : NULL;

    m_resource_tag_map.push_back(Resource_tag_tuple(kind, url, sel, tag));
}

// Access messages.
const Messages &Link_unit_jit::access_messages() const
{
    switch (m_target_lang) {
    case ICode_generator::TL_NATIVE:
        {
            mi::base::Handle<Generated_code_lambda_function> native_code(
                m_code->get_interface<mi::mdl::Generated_code_lambda_function>());
            return native_code->access_messages();
        }
    case ICode_generator::TL_PTX:
    case ICode_generator::TL_HLSL:
    case ICode_generator::TL_GLSL:
    case ICode_generator::TL_LLVM_IR:
    default:
        {
            mi::base::Handle<Generated_code_source> source_code(
                m_code->get_interface<mi::mdl::Generated_code_source>());
            return source_code->access_messages();
        }
    }
}

// Get write access to the messages of the generated code.
Messages_impl &Link_unit_jit::access_messages()
{
    switch (m_target_lang) {
    case ICode_generator::TL_NATIVE:
        {
            mi::base::Handle<Generated_code_lambda_function> native_code(
                m_code->get_interface<mi::mdl::Generated_code_lambda_function>());
            return native_code->access_messages();
        }
    case ICode_generator::TL_PTX:
    case ICode_generator::TL_HLSL:
    case ICode_generator::TL_GLSL:
    case ICode_generator::TL_LLVM_IR:
    default:
        {
            mi::base::Handle<Generated_code_source> source_code(
                m_code->get_interface<mi::mdl::Generated_code_source>());
            return source_code->access_messages();
        }
    }
}

// Get the LLVM context to use with this link unit.
llvm::LLVMContext *Link_unit_jit::get_llvm_context()
{
    if (m_target_lang == ICode_generator::TL_NATIVE) {
        mi::base::Handle<Generated_code_lambda_function> native_code(
            m_code->get_interface<mi::mdl::Generated_code_lambda_function>());
        return &native_code->get_llvm_context();
    }
    return &m_source_only_llvm_context;
}

// Add a lambda function to this link unit.
bool Link_unit_jit::add(
    ILambda_function const                    *ilambda,
    IModule_cache                             *module_cache,
    ICall_name_resolver const                 *resolver,
    IGenerated_code_executable::Function_kind  kind,
    size_t                                    *arg_block_index,
    size_t                                    *function_index)
{
    if (arg_block_index == NULL) {
        return false;
    }

    Lambda_function const *lambda = impl_cast<Lambda_function>(ilambda);
    if (lambda == NULL) {
        return false;
    }

    DAG_node const *body = lambda->get_body();
    if (body == NULL && lambda->get_root_expr_count() < 1) {
        // there must be at least one root or a body
        return false;
    }

    Module_cache_scope scope(*this, module_cache);

    // we compile a new lambda, do an update of the resource attribute map
    update_resource_attribute_map(lambda);

    // add to lambda list, as m_code_gen will see it
    m_lambdas.push_back(mi::base::make_handle_dup(lambda));

    size_t func_index = m_code_gen.get_current_exported_function_count();

    llvm::Function *func = NULL;
    size_t next_arg_block_index =
        *arg_block_index != ~0 ? *arg_block_index : m_arg_block_layouts.size();
    if (body != NULL) {
        func = m_code_gen.compile_lambda(
            /*incremental=*/true, *lambda, resolver, /*transformer=*/NULL, next_arg_block_index);
    } else {
        func = m_code_gen.compile_switch_lambda(
            /*incremental=*/true, *lambda, resolver, next_arg_block_index);
    }

    if (func != NULL) {
        if (m_code_gen.get_captured_arguments_llvm_type() != NULL && *arg_block_index == ~0) {
            IAllocator        *alloc = get_allocator();
            Allocator_builder builder(alloc);
            m_arg_block_layouts.push_back(
                mi::base::make_handle(
                    builder.create<Generated_code_value_layout>(alloc, &m_code_gen)));
            MDL_ASSERT(next_arg_block_index == m_arg_block_layouts.size() - 1);
            *arg_block_index = next_arg_block_index;
        }

        // pass out the function index
        if (function_index != NULL) {
            *function_index = func_index;
        }

        return true;
    }
    return false;
}

// Add a distribution function to this link unit.
bool Link_unit_jit::add(
    IDistribution_function const *idist_func,
    IModule_cache                *module_cache,
    ICall_name_resolver const    *resolver,
    size_t                       *arg_block_index,
    size_t                       *main_function_indices,
    size_t                        num_main_function_indices)
{
    Distribution_function const *dist_func = impl_cast<Distribution_function>(idist_func);
    if (dist_func == NULL) {
        return false;
    }

    // wrong size of main_function_indices array?
    if (main_function_indices != NULL &&
        num_main_function_indices != dist_func->get_main_function_count())
    {
        return false;
    }

    Module_cache_scope scope(*this, module_cache);

    mi::base::Handle<ILambda_function> root_lambda_handle(dist_func->get_root_lambda());
    Lambda_function const *root_lambda = impl_cast<Lambda_function>(root_lambda_handle.get());

    // we compile a new lambda, do an update of the resource attribute map
    update_resource_attribute_map(root_lambda);

    // Add to distribution function list, as m_code_gen will see it
    m_dist_funcs.push_back(mi::base::make_handle_dup(idist_func));

    size_t next_arg_block_index =
        *arg_block_index != ~0 ? *arg_block_index : m_arg_block_layouts.size();
    LLVM_code_generator::Function_vector llvm_funcs(get_allocator());
    llvm::Module *module = m_code_gen.compile_distribution_function(
        /*incremental=*/ true,
        *dist_func,
        resolver,
        llvm_funcs,
        next_arg_block_index,
        main_function_indices);

    if (module == NULL) {
        return false;
    }

    if (m_code_gen.get_captured_arguments_llvm_type() != NULL && *arg_block_index == ~0) {
        IAllocator        *alloc = get_allocator();
        Allocator_builder builder(alloc);
        m_arg_block_layouts.push_back(
            mi::base::make_handle(
                builder.create<Generated_code_value_layout>(alloc, &m_code_gen)));
        MDL_ASSERT(next_arg_block_index == m_arg_block_layouts.size() - 1);
        *arg_block_index = next_arg_block_index;
    }

    return true;
}

// Get the number of functions in this link unit.
size_t Link_unit_jit::get_function_count() const
{
    return m_code_gen.get_current_exported_function_count();
}

// Get the name of the i'th function inside this link unit.
char const *Link_unit_jit::get_function_name(size_t i) const
{
    return m_code->get_function_name(i);
}

// Returns the distribution kind of the i'th function inside this link unit.
IGenerated_code_executable::Distribution_kind Link_unit_jit::get_distribution_kind(size_t i) const
{
    return m_code->get_distribution_kind(i);
}

// Returns the function kind of the i'th function inside this link unit.
IGenerated_code_executable::Function_kind Link_unit_jit::get_function_kind(size_t i) const
{
    return m_code->get_function_kind(i);
}

// Get the index of the target argument block layout for the i'th function inside this link
// unit if used.
size_t Link_unit_jit::get_function_arg_block_layout_index(size_t i) const
{
    return m_code->get_function_arg_block_layout_index(i);
}

// Get the LLVM function of the i'th function inside this link unit.
llvm::Function *Link_unit_jit::get_function(size_t i) const
{
    LLVM_code_generator::Exported_function *exp_func = m_code_gen.get_current_exported_function(i);
    if (exp_func == NULL) {
        return NULL;
    }

    return exp_func->func;
}

// Returns the prototype of the i'th function inside this link unit.
char const *Link_unit_jit::get_function_prototype(
    size_t index,
    IGenerated_code_executable::Prototype_language lang) const
{
    return m_code->get_function_prototype(index, lang);
}

// Get the number of target argument block layouts used by this link unit.
size_t Link_unit_jit::get_arg_block_layout_count() const
{
    return m_arg_block_layouts.size();
}

// Get the i'th target argument block layout used by this link unit.
IGenerated_code_value_layout const *Link_unit_jit::get_arg_block_layout(size_t i) const
{
    if (i < m_arg_block_layouts.size()) {
        IGenerated_code_value_layout const *layout = m_arg_block_layouts[i].get();
        layout->retain();
        return layout;
    }
    return NULL;
}

// Get the LLVM module.
llvm::Module const *Link_unit_jit::get_llvm_module() const
{
    return m_code_gen.get_llvm_module();
}

// Create the jit code generator.
ICode_generator *create_code_generator_jit(IAllocator *alloc, MDL *mdl)
{
    return Code_generator_jit::create_code_generator(alloc, mdl);
}

// Get the jitted code singleton.
Jitted_code *create_jitted_code_singleton(IAllocator *alloc)
{
    return Jitted_code::get_instance(alloc);
}

// Terminate the jitted code singleton.
void terminate_jitted_code_singleton(Jitted_code *jitted_code)
{
    if (jitted_code != NULL) {
        jitted_code->release();
    }
}

} // mdl
} // mi

