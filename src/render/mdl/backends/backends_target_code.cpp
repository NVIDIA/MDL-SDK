/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

#include "pch.h"

#include <cstring>
#include <base/system/version/i_version.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ibuffer.h>
#include <render/mdl/runtime/i_mdlrt_resource_handler.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <mdl/jit/generator_jit/generator_jit_libbsdf_data.h>
#include <mdl/jit/generator_jit/generator_jit_generated_code_value_layout.h>
#include <api/api/neuray/neuray_mdl_execution_context_impl.h>
#include <api/api/neuray/neuray_transaction_impl.h>
#include <api/api/neuray/neuray_value_impl.h>
#include "backends_backends.h"
#include "backends_target_code.h"

namespace MI {

namespace BACKENDS {

namespace {
// ---------------------- Internal target resource callback class ---------------------

/// Implementation of the internal version of the #mi::neuraylib::ITarget_resource_callback
/// callback interface operating on MI::MDL::IValue_resource objects.
class Target_resource_callback_internal : public ITarget_resource_callback_internal
{
public:
    /// Constructor.
    ///
    /// \param transaction  the transaction to resolve the textures
    /// \param target_code  the target code providing the resource indices
    Target_resource_callback_internal(DB::Transaction* transaction, Target_code *target_code)
    : m_transaction(transaction)
    , m_target_code(target_code)
    {
    }

    /// Returns a resource index for the given resource value usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// The value 0 is always an invalid resource reference.
    /// For #mi::mdl::IValue_texture values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_texture().
    /// For mi::mdl::IValue_light_profile values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_light_profile().
    /// For mi::mdl::IValue_bsdf_measurement values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_bsdf_measurement().
    ///
    /// See \ref mi_neuray_ptx_texture_lookup_call_modes for more details about texture handlers
    /// for the PTX backend.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    mi::Uint32 get_resource_index(MI::MDL::IValue_resource const *resource) final {
        return m_target_code->get_known_resource_index(m_transaction, resource);
    }


    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    mi::Uint32 get_string_index(MI::MDL::IValue_string const *s) final {
        return m_target_code->get_string_index(s->get_value());
    }

private:
    DB::Transaction *m_transaction;
    Target_code *m_target_code;
};

} // anonymous


// --------------------- Target code --------------------

Target_code::Target_code()
    : m_native_code(nullptr)
    , m_backend_kind(static_cast<mi::neuraylib::IMdl_backend_api::Mdl_backend_kind>(-1))
    , m_code()
    , m_code_segments()
    , m_code_segment_descriptions()
    , m_callable_function_infos()
    , m_texture_table()
    , m_body_texture_count(0)
    , m_light_profile_table()
    , m_body_light_profile_count(0)
    , m_bsdf_measurement_table()
    , m_body_bsdf_measurement_count(0)
    , m_string_constant_table()
    , m_data_segments()
    , m_cap_arg_layouts()
    , m_cap_arg_blocks()
    , m_rh(nullptr)
    , m_render_state_usage(~0u)
    , m_string_args_mapped_to_ids(true)
    , m_use_builtin_resource_handler(false)
{
}

// Constructor from executable code.
Target_code::Target_code(
    mi::mdl::IGenerated_code_executable* code,
    MI::DB::Transaction* transaction,
    bool string_ids,
    bool use_derivatives,
    bool use_builtin_resource_handler,
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind be_kind)
  : Target_code()
{
    m_backend_kind = be_kind;
    m_string_args_mapped_to_ids = string_ids;
    m_use_builtin_resource_handler = use_builtin_resource_handler;
    finalize(code, transaction, use_derivatives);

    size_t num_layouts = code->get_captured_argument_layouts_count();
    m_cap_arg_blocks.resize(num_layouts); // already prepare the empty argument block slots

    for (size_t i = 0; i < num_layouts; ++i) {
        mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
            code->get_captured_arguments_layout(i));
        m_cap_arg_layouts.push_back(
            mi::base::make_handle(
                new Target_value_layout(layout.get(), m_string_args_mapped_to_ids)));
    }
}


// Constructor for link mode.
Target_code::Target_code(
    bool string_ids,
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind be_kind)
  : Target_code()
{
    m_backend_kind = be_kind;
    m_string_args_mapped_to_ids = string_ids;
    m_use_builtin_resource_handler = true;
}

Target_code::~Target_code()
{
    if (m_native_code.is_valid_interface()) {
        m_native_code->term();

        if(m_rh)
            delete m_rh;
        m_rh = nullptr;
    }
}

void Target_code::finalize(
    mi::mdl::IGenerated_code_executable* code,
    MI::DB::Transaction* transaction,
    bool use_derivatives)
{
    m_native_code = mi::base::make_handle(
        code->get_interface<mi::mdl::IGenerated_code_lambda_function>());
    m_render_state_usage = code->get_state_usage();

    if (m_native_code.is_valid_interface()) {
        if(m_use_builtin_resource_handler)
            m_rh = new MDLRT::Resource_handler(use_derivatives);

        m_native_code->init(transaction, NULL, m_rh);
    } else {
        // only source code itself
        size_t size = 0;
        char const *src = code->get_source_code(size);

        m_code = std::string(src, size);
    }

    // copy function infos to target code
    for (size_t i = 0, n = code->get_function_count(); i < n; ++i) {
        m_callable_function_infos.push_back(
            Callable_function_info(
                code->get_function_name(i),
                Distribution_kind(code->get_distribution_kind(i)),
                Function_kind(code->get_function_kind(i)),
                code->get_function_arg_block_layout_index(i),
                code->get_function_state_usage(i)));
        Callable_function_info &info = m_callable_function_infos.back();
        for (mi::mdl::IGenerated_code_executable::Prototype_language lang =
                mi::mdl::IGenerated_code_executable::Prototype_language(0);
            lang < mi::mdl::IGenerated_code_executable::PL_NUM_LANGUAGES;
            lang = mi::mdl::IGenerated_code_executable::Prototype_language(lang + 1))
        {
            char const *prototype = code->get_function_prototype(i, lang);
            if (prototype == NULL)
                info.m_prototypes.push_back(std::string());
            else
                info.m_prototypes.push_back(prototype);
        }
        for (size_t handle_index = 0, num_handles = code->get_function_df_handle_count(i);
                handle_index < num_handles; ++handle_index) {
            info.m_df_handle_name_table.push_back(code->get_function_df_handle(i, handle_index));
        }
    }
}


mi::neuraylib::IMdl_backend_api::Mdl_backend_kind Target_code::get_backend_kind() const
{
    return m_backend_kind;
}

const char* Target_code::get_code() const
{
    return m_code.c_str();
}

mi::Size Target_code::get_code_size() const
{
    return m_code.size();
}

mi::Size Target_code::get_callable_function_count() const
{
    return m_callable_function_infos.size();
}

const char* Target_code::get_callable_function(mi::Size index) const
{
    if( index < m_callable_function_infos.size())
        return m_callable_function_infos[ index].m_name.c_str();
    return NULL;
}

// Returns the prototype of a callable function in the target code.
const char* Target_code::get_callable_function_prototype(
    mi::Size index,
    mi::neuraylib::ITarget_code::Prototype_language lang) const
{
    if( index >= m_callable_function_infos.size() ||
            lang >= m_callable_function_infos[ index].m_prototypes.size()) {
        return NULL;
    }
    return m_callable_function_infos[ index].m_prototypes[ lang].c_str();
}

// Returns the kind of a callable function in the target code.
mi::neuraylib::ITarget_code::Distribution_kind Target_code::get_callable_function_distribution_kind(
    mi::Size index) const
{
    if (index < m_callable_function_infos.size())
        return m_callable_function_infos[index].m_dist_kind;
    return mi::neuraylib::ITarget_code::DK_INVALID;
}

// Returns the kind of a callable function in the target code.
mi::neuraylib::ITarget_code::Function_kind Target_code::get_callable_function_kind(
    mi::Size index) const
{
    if( index < m_callable_function_infos.size())
        return m_callable_function_infos[ index].m_kind;
    return mi::neuraylib::ITarget_code::FK_INVALID;
}

// Get the index of the target argument block to use with a callable function.
Size Target_code::get_callable_function_argument_block_index(mi::Size index) const
{
    if( index < m_callable_function_infos.size())
        return m_callable_function_infos[ index].m_arg_block_index;
    return ~0;
}

// Get the number of distribution function handles referenced by a callable function.
Size Target_code::get_callable_function_df_handle_count(Size func_index) const
{
    if( func_index < m_callable_function_infos.size())
        return m_callable_function_infos[ func_index].m_df_handle_name_table.size();
    return 0;
}

// Get the name of a distribution function handle referenced by a callable function.
const char* Target_code::get_callable_function_df_handle(
    Size func_index,
    Size handle_index) const
{
    if( func_index >= m_callable_function_infos.size() ||
            handle_index >= m_callable_function_infos[ func_index].m_df_handle_name_table.size())
        return NULL;

    return m_callable_function_infos[ func_index].m_df_handle_name_table[ handle_index].c_str();
}

// Returns the potential render state usage of callable function in the target code.
mi::neuraylib::ITarget_code::State_usage Target_code::get_callable_function_render_state_usage(
    Size index) const
{
    if( index < m_callable_function_infos.size())
        return m_callable_function_infos[ index].m_state_usage;
    return 0;
}

mi::Size Target_code::get_texture_count() const
{
    return m_texture_table.size();
}

mi::Size Target_code::get_body_texture_count() const
{
    return m_body_texture_count;
}

const char* Target_code::get_texture( mi::Size index) const
{
    if( index < m_texture_table.size()) {
        return m_texture_table[ index].get_db_name();
    }
    return NULL;
}

const char* Target_code::get_texture_url(mi::Size index) const
{
    if (index < m_texture_table.size()) {
        return m_texture_table[index].get_mdl_url();
    }
    return NULL;
}

const char* Target_code::get_texture_owner_module(mi::Size index) const
{
    if (index < m_texture_table.size()) {
        return m_texture_table[index].get_owner();
    }
    return NULL;
}

mi::neuraylib::ITarget_code::Gamma_mode Target_code::get_texture_gamma(mi::Size index) const
{
    if (index < m_texture_table.size()) {
        float gamma = m_texture_table[index].get_gamma();
        if (gamma == 0.0f)
            return mi::neuraylib::ITarget_code::GM_GAMMA_DEFAULT;
        else if (gamma == 1.0f)
            return mi::neuraylib::ITarget_code::GM_GAMMA_LINEAR;
        else if (gamma == 2.2f)
            return mi::neuraylib::ITarget_code::GM_GAMMA_SRGB;
    }
    return mi::neuraylib::ITarget_code::GM_GAMMA_UNKNOWN;
}

Target_code::Texture_shape Target_code::get_texture_shape( mi::Size index) const
{
    if( index < m_texture_table.size()) {
        return m_texture_table[ index].get_texture_shape();
    }
    return Target_code::Texture_shape_invalid;
}

const mi::Float32* Target_code::get_texture_df_data(
    mi::Size index,
    mi::Size &rx,
    mi::Size &ry,
    mi::Size &rz) const
{
    if (index < m_texture_table.size() &&
        m_texture_table[index].get_texture_shape() == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data) {

        return get_df_data_texture(m_texture_table[index].get_df_data_kind(), rx, ry, rz);
    }
    return nullptr;
}

static mi::neuraylib::Df_data_kind convert_df_data_kind(
    mi::mdl::IValue_texture::Bsdf_data_kind kind)
{
    switch (kind)
    {
    case  mi::mdl::IValue_texture::BDK_NONE:
        return mi::neuraylib::DFK_NONE;
    case mi::mdl::IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        return mi::neuraylib::DFK_BACKSCATTERING_GLOSSY_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER:
        return mi::neuraylib::DFK_BECKMANN_SMITH_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_BECKMANN_VC_MULTISCATTER:
        return mi::neuraylib::DFK_BECKMANN_VC_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_GGX_SMITH_MULTISCATTER:
        return mi::neuraylib::DFK_GGX_SMITH_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_GGX_VC_MULTISCATTER:
        return mi::neuraylib::DFK_GGX_VC_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_SHEEN_MULTISCATTER:
        return mi::neuraylib::DFK_SHEEN_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER:
        return mi::neuraylib::DFK_SIMPLE_GLOSSY_MULTISCATTER;
    case mi::mdl::IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER:
        return mi::neuraylib::DFK_WARD_GEISLER_MORODER_MULTISCATTER;
    default:
        break;
    }
    return mi::neuraylib::DFK_INVALID;
}

mi::neuraylib::Df_data_kind Target_code::get_texture_df_data_kind(Size index) const
{
    if (index < m_texture_table.size()) {
        return convert_df_data_kind(m_texture_table[index].get_df_data_kind());
    }
    return mi::neuraylib::DFK_INVALID;
}

mi::Size Target_code::get_light_profile_count() const
{
    return m_light_profile_table.size();
}

mi::Size Target_code::get_body_light_profile_count() const
{
    return m_body_light_profile_count;
}

const char* Target_code::get_light_profile(mi::Size index) const
{
    if( index < m_light_profile_table.size()) {
        return m_light_profile_table[index].get_db_name();
    }
    return NULL;
}

const char* Target_code::get_light_profile_url(mi::Size index) const
{
    if (index < m_light_profile_table.size()) {
        return m_light_profile_table[index].get_mdl_url();
    }
    return NULL;
}

const char* Target_code::get_light_profile_owner_module(mi::Size index) const
{
    if (index < m_light_profile_table.size()) {
        return m_light_profile_table[index].get_owner();
    }
    return NULL;
}

Size Target_code::get_bsdf_measurement_count() const
{
    return m_bsdf_measurement_table.size();
}

Size Target_code::get_body_bsdf_measurement_count() const
{
    return m_body_bsdf_measurement_count;
}

const char* Target_code::get_bsdf_measurement(mi::Size index) const
{
    if (index < m_bsdf_measurement_table.size()) {
        return m_bsdf_measurement_table[index].get_db_name();
    }
    return NULL;
}

const char* Target_code::get_bsdf_measurement_url(mi::Size index) const
{
    if (index < m_bsdf_measurement_table.size()) {
        return m_bsdf_measurement_table[index].get_mdl_url();
    }
    return NULL;
}

const char* Target_code::get_bsdf_measurement_owner_module(mi::Size index) const
{
    if (index < m_bsdf_measurement_table.size()) {
        return m_bsdf_measurement_table[index].get_owner();
    }
    return NULL;
}

// Returns the number of string constants used by the target code.
Size Target_code::get_string_constant_count() const
{
    return m_string_constant_table.size();
}

// Returns the string constant used by the target code.
const char* Target_code::get_string_constant(Size index) const
{
    if (index < m_string_constant_table.size()) {
        return m_string_constant_table[index].c_str();
    }
    return NULL;
}

mi::Size Target_code::get_ro_data_segment_count() const
{
    return m_data_segments.size();
}

const char* Target_code::get_ro_data_segment_name(mi::Size index) const
{
    if( index >= m_data_segments.size())
        return NULL;
    const Segment& segment = m_data_segments[index];
    return segment.get_name();
}

mi::Size Target_code::get_ro_data_segment_size(mi::Size index) const
{
    if( index >= m_data_segments.size())
        return 0;
    const Segment& segment = m_data_segments[index];
    return segment.get_size();
}

const char* Target_code::get_ro_data_segment_data(mi::Size index) const
{
    if( index >= m_data_segments.size())
        return NULL;
    const Segment& segment = m_data_segments[index];
    return (const char*) segment.get_data();
}


// reduce redundant code be wrapping bsdf, edf, ... calls
mi::Sint32 Target_code::execute_df_init_function(
    mi::neuraylib::ITarget_code::Distribution_kind dist_kind,
    mi::Size index,
    mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    if (!m_native_code.is_valid_interface()) return -2;
    if (index >= m_callable_function_infos.size()) return -2;
    if (m_callable_function_infos[index].m_dist_kind != dist_kind) return -2;
    if (m_callable_function_infos[index].m_kind != mi::neuraylib::ITarget_code::FK_DF_INIT)
        return -2;

    const char *args_data = NULL;
    if (cap_args != NULL)
        args_data = cap_args->get_data();
    else
    {
        mi::Size block_index = get_callable_function_argument_block_index(index);
        if (block_index != mi::Size(~0) &&
            block_index < m_cap_arg_blocks.size() &&
            m_cap_arg_blocks[block_index])
        {
            args_data = m_cap_arg_blocks[block_index]->get_data();
        }
    }

    return m_native_code->run_init(
        index,
        // ugly cast necessary because the C++ I/F cannot handle the layout options
        reinterpret_cast<mi::mdl::Shading_state_material*>(&state),
        tex_handler,
        args_data) ? 0 : -1;
}

// reduce redundant code be wrapping bsdf, edf, ... calls
mi::Sint32 Target_code::execute_generic_function(
    mi::neuraylib::ITarget_code::Distribution_kind dist_kind,
    mi::neuraylib::ITarget_code::Function_kind func_kind,
    mi::Size index,
    void *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    if (!m_native_code.is_valid_interface()) return -2;
    if (index >= m_callable_function_infos.size()) return -2;
    if (m_callable_function_infos[index].m_dist_kind != dist_kind) return -2;
    if (m_callable_function_infos[index].m_kind != func_kind) return -2;

    const char *args_data = NULL;
    if (cap_args != NULL)
        args_data = cap_args->get_data();
    else
    {
        mi::Size block_index = get_callable_function_argument_block_index(index);
        if (block_index != mi::Size(~0) &&
            block_index < m_cap_arg_blocks.size() &&
            m_cap_arg_blocks[block_index])
        {
            args_data = m_cap_arg_blocks[block_index]->get_data();
        }
    }

    return m_native_code->run_generic(
        index,
        data,
        // ugly cast necessary because the C++ I/F cannot handle the layout options
        reinterpret_cast<const mi::mdl::Shading_state_material*>(&state),
        tex_handler,
        args_data) ? 0 : -1;
}


mi::Sint32 Target_code::execute(
    mi::Size index,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args,
    void* result) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_NONE,
        mi::neuraylib::ITarget_code::FK_LAMBDA, index, result, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_environment(
    mi::Size index,
    const mi::neuraylib::Shading_state_environment& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    mi::Spectrum_struct* result) const
{
    if (!m_native_code.is_valid_interface()) return -2;
    if (index >= m_callable_function_infos.size()) return -2;
    if (m_callable_function_infos[index].m_kind != FK_ENVIRONMENT) return -2;

    return m_native_code->run_environment(
        index,
        // ugly cast necessary because the libmdl I/F uses RGB_color*
        reinterpret_cast<mi::mdl::RGB_color*>(result),
        // ugly cast necessary because the C++ I/F cannot handle the layout options
        reinterpret_cast<const mi::mdl::Shading_state_environment*>(&state),
        tex_handler) ? 0 : -1;
}

mi::Sint32 Target_code::execute_bsdf_init(
    mi::Size index,
    mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_df_init_function(mi::neuraylib::ITarget_code::DK_BSDF,
        index, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_bsdf_sample(
    mi::Size index,
    mi::neuraylib::Bsdf_sample_data *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_BSDF,
        mi::neuraylib::ITarget_code::FK_DF_SAMPLE, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_bsdf_evaluate(
    mi::Size index,
    mi::neuraylib::Bsdf_evaluate_data_base *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_BSDF,
        mi::neuraylib::ITarget_code::FK_DF_EVALUATE, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_bsdf_pdf(
    mi::Size index,
    mi::neuraylib::Bsdf_pdf_data *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_BSDF,
        mi::neuraylib::ITarget_code::FK_DF_PDF, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_bsdf_auxiliary(
    mi::Size index,
    mi::neuraylib::Bsdf_auxiliary_data_base *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_BSDF,
        mi::neuraylib::ITarget_code::FK_DF_AUXILIARY, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_edf_init(
    mi::Size index,
    mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_df_init_function(mi::neuraylib::ITarget_code::DK_EDF,
        index, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_edf_sample(
    mi::Size index,
    mi::neuraylib::Edf_sample_data *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_EDF,
        mi::neuraylib::ITarget_code::FK_DF_SAMPLE, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_edf_evaluate(
    mi::Size index,
    mi::neuraylib::Edf_evaluate_data_base *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_EDF,
        mi::neuraylib::ITarget_code::FK_DF_EVALUATE, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_edf_pdf(
    mi::Size index,
    mi::neuraylib::Edf_pdf_data *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_EDF,
        mi::neuraylib::ITarget_code::FK_DF_PDF, index, data, state, tex_handler, cap_args);
}

mi::Sint32 Target_code::execute_edf_auxiliary(
    mi::Size index,
    mi::neuraylib::Edf_auxiliary_data_base *data,
    const mi::neuraylib::Shading_state_material& state,
    mi::neuraylib::Texture_handler_base* tex_handler,
    const mi::neuraylib::ITarget_argument_block *cap_args) const
{
    return execute_generic_function(mi::neuraylib::ITarget_code::DK_EDF,
        mi::neuraylib::ITarget_code::FK_DF_AUXILIARY, index, data, state, tex_handler, cap_args);
}

mi::neuraylib::ITarget_code::State_usage Target_code::get_render_state_usage() const
{
    return m_render_state_usage;
}

size_t Target_code::add_function(
    const std::string& name,
    Distribution_kind dist_kind,
    Function_kind kind,
    mi::Size arg_block_index,
    State_usage state_usage)
{
    size_t idx = m_callable_function_infos.size();
    m_callable_function_infos.push_back(
        Callable_function_info(name, dist_kind, kind, arg_block_index, state_usage));
    return idx;
}

// Set a function prototype for a callable function.
void Target_code::set_function_prototype(
    size_t index,
    Prototype_language lang,
    const std::string& prototype)
{
    ASSERT( M_BACKENDS, index < m_callable_function_infos.size());
    if( lang >= m_callable_function_infos[ index].m_prototypes.size()) {
        m_callable_function_infos[index].m_prototypes.resize(lang + 1);
    }
    m_callable_function_infos[index].m_prototypes[lang] = prototype;
}

void Target_code::add_texture_index(
    size_t index,
    const std::string& name,
    const std::string& mdl_url,
    float gamma,
    Texture_shape shape,
    mi::mdl::IValue_texture::Bsdf_data_kind df_data_kind)
{
    if( index >= m_texture_table.size()) {
        m_texture_table.resize(index + 1, Target_code::Texture_info(
            /*db_name=*/"",
            /*mdl_url=*/"",
            /*owner=*/"",
            /*gamma=*/0.0f,
            /*texture_shape=*/Texture_shape_invalid,
            /*df_data_kind=*/ mi::mdl::IValue_texture::BDK_NONE));
    }

    std::string owner = MDL::get_resource_owner_prefix( mdl_url);
    std::string url   = MDL::strip_resource_owner_prefix( mdl_url);
    m_texture_table[index] = Target_code::Texture_info(
        name, url, owner, gamma, shape, df_data_kind);
}

// Registers a used light profile index.
void Target_code::add_light_profile_index(
    size_t index,
    const std::string& name,
    const std::string& mdl_url)
{
    if( index >= m_light_profile_table.size()) {
        m_light_profile_table.resize( index + 1,
            Target_code::Resource_info(
            /*db_name=*/"",
            /*mdl_url=*/"",
            /*owner=*/""));
    }

    std::string owner = MDL::get_resource_owner_prefix( mdl_url);
    std::string url   = MDL::strip_resource_owner_prefix( mdl_url);
    m_light_profile_table[index] = Target_code::Resource_info(
        name, url, owner);
}

// Registers a used bsdf measurement index.
void Target_code::add_bsdf_measurement_index(
    size_t index,
    const std::string& name,
    const std::string& mdl_url)
{
    if( index >= m_bsdf_measurement_table.size()) {
        m_bsdf_measurement_table.resize(index + 1, Target_code::Resource_info(
            /*db_name=*/"",
            /*mdl_url=*/"",
            /*owner=*/""));
    }

    std::string owner = MDL::get_resource_owner_prefix( mdl_url);
    std::string url   = MDL::strip_resource_owner_prefix( mdl_url);
    m_bsdf_measurement_table[index] = Target_code::Resource_info( name, url, owner);
}

// Registers a used string constant index.
void Target_code::add_string_constant_index(
    size_t index,
    const std::string& scons)
{
    if (index >= m_string_constant_table.size()) {
        m_string_constant_table.resize(index + 1, "");
    }
    m_string_constant_table[index] = scons;
}

// Set the body resource counts.
void Target_code::set_body_resource_counts(
    mi::Size body_texture_counts,
    mi::Size body_light_profile_counts,
    mi::Size body_bsdf_measurement_counts)
{
    m_body_texture_count = body_texture_counts;
    m_body_light_profile_count = body_light_profile_counts;
    m_body_bsdf_measurement_count = body_bsdf_measurement_counts;
}

void Target_code::add_ro_segment(
    const char* name, 
    const unsigned char* data, 
    mi::Size size)
{
    m_data_segments.push_back(Target_code::Segment(name, data, size));
}

mi::Size Target_code::get_code_segment_count() const
{
    return m_code_segments.size();
}

const char* Target_code::get_code_segment( mi::Size index) const
{
    if( index < m_code_segments.size())
        return m_code_segments[index].c_str();
    return NULL;
}

mi::Size Target_code::get_code_segment_size( mi::Size index) const
{
    if( index < m_code_segments.size())
        return m_code_segments[index].size();
    return 0;
}

const char* Target_code::get_code_segment_description( mi::Size index) const
{
    if( index < m_code_segment_descriptions.size())
        return m_code_segment_descriptions[index].c_str();
    return NULL;
}

// Returns the number of target argument blocks / block layouts.
Size Target_code::get_argument_block_count() const
{
    return m_cap_arg_blocks.size();
}

// Get a target argument block if available.
const mi::neuraylib::ITarget_argument_block *Target_code::get_argument_block(Size index) const
{
    if ( index >= m_cap_arg_blocks.size())
        return NULL;
    mi::neuraylib::ITarget_argument_block *arg_block = m_cap_arg_blocks[index].get();
    if (!arg_block)
        return NULL;
    arg_block->retain();
    return arg_block;
}

// Returns the number of target argument blocks / block layouts.
Size Target_code::get_argument_layout_count() const
{
    return m_cap_arg_layouts.size();
}

// Get the captured arguments block layout if available.
mi::neuraylib::ITarget_value_layout const * Target_code::get_argument_block_layout(Size index) const
{
    if ( index >= m_cap_arg_layouts.size())
        return NULL;
    mi::neuraylib::ITarget_value_layout const *layout = m_cap_arg_layouts[index].get();
    layout->retain();
    return layout;
}

// Create a target argument block of the class-compiled material for this target code.
mi::neuraylib::ITarget_argument_block *Target_code::create_argument_block(
    Size index,
    const mi::neuraylib::ICompiled_material* material,
    mi::neuraylib::ITarget_resource_callback *resource_callback) const
{
    if ( material == NULL || resource_callback == NULL || index >= m_cap_arg_layouts.size())
        return NULL;

    mi::neuraylib::ITarget_value_layout const *layout = m_cap_arg_layouts[index].get();
    mi::Size num_args = material->get_parameter_count();
    if ( num_args != layout->get_num_elements())
        return NULL;

    Target_argument_block *arg_block = new Target_argument_block( layout->get_size());

    for ( mi::Size i = 0; i < num_args; ++i) {
        mi::neuraylib::Target_value_layout_state state = layout->get_nested_state( i);
        mi::base::Handle<const mi::neuraylib::IValue> arg_val( material->get_argument( i));
        layout->set_value(
            arg_block->get_data(),
            arg_val.get(),
            resource_callback,
            state);
    }

    return arg_block;
}

// Initializes the target argument block for the class-compiled material which was used
// to generate this target code and adds all resources from the arguments to the target code
// resource lists.
void Target_code::init_argument_block(
    Size index,
    MI::DB::Transaction* transaction,
    const MDL::IValue_list* args)
{
    ASSERT( M_BACKENDS, index < m_cap_arg_blocks.size() &&
        "captured argument block not prepared");
    if (!args || index >= m_cap_arg_blocks.size())
        return;

    // Argument block already initialized? Do nothing
    if (m_cap_arg_blocks[index])
        return;

    ASSERT(M_BACKENDS,
        index < m_cap_arg_layouts.size() && m_cap_arg_layouts[index] &&
        "captured arguments but no layout");
    if (index >= m_cap_arg_layouts.size() || !m_cap_arg_layouts[index])
        return;

    Target_value_layout const *layout = m_cap_arg_layouts[index].get();
    mi::Size num_args = args->get_size();
    if ( num_args != layout->get_num_elements())
        return;

    Target_argument_block *block = new Target_argument_block( layout->get_size());
    m_cap_arg_blocks[index] = mi::base::make_handle(block);

    Target_resource_callback_internal resource_callback(transaction, this);

    for (mi::Size i = 0; i < num_args; ++i) {
        mi::neuraylib::Target_value_layout_state state = layout->get_nested_state( i);
        mi::base::Handle<const MI::MDL::IValue> arg_val( args->get_value( i));
        layout->set_value(
            block->get_data(),
            arg_val.get(),
            &resource_callback,
            state);
    }
}

// Add a target argument block layout to this target code.
mi::Size Target_code::add_argument_block_layout(Target_value_layout *layout)
{
    m_cap_arg_layouts.push_back(mi::base::make_handle_dup(layout));
    m_cap_arg_blocks.push_back(mi::base::Handle<mi::neuraylib::ITarget_argument_block>());
    return m_cap_arg_layouts.size() - 1;
}

// Get the string identifier for a given string inside the constant table or 0
// if the string is not known.
mi::Uint32 Target_code::get_string_index(char const* string) const
{
    if (string == NULL)
        return 0u;

    // slow linear search here, but the number of string is expected to be small
    std::string str(string);
    for (size_t i = 1, n = m_string_constant_table.size(); i < n; ++i) {
        if (m_string_constant_table[i] == str)
            return mi::Uint32(i);
    }
    return 0u;
}

const mi::Float32* Target_code::get_df_data_texture(
    mi::mdl::IValue_texture::Bsdf_data_kind kind,
    mi::Size &rx,
    mi::Size &ry,
    mi::Size &rz)
{
    size_t w = 0, h = 0, d = 0;
    mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data_resolution(
        kind, w, h, d);
    rx = w;
    ry = h;
    rz = d;
    size_t s;
    return reinterpret_cast<const mi::Float32*>(
        mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data(kind, s));
}

/// Returns the resource index for use in an \c ITarget_argument_block of resources already
/// known when this \c ITarget_code object was generated.
mi::Uint32 Target_code::get_known_resource_index(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IValue_resource const *resource) const
{
    if (transaction == NULL || resource == NULL) return 0;

    // TODO: This should be moved into api/api/mdl to not have mi::neuraylib objects in this module
    NEURAY::Transaction_impl* transaction_impl =
        static_cast<NEURAY::Transaction_impl*>(transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    ASSERT(M_BACKENDS, db_transaction);

    // copied from NEURAY::get_internal_value<T>
    mi::base::Handle<const NEURAY::IValue_wrapper> resource_wrapper(
        resource->get_interface<NEURAY::IValue_wrapper>());
    if (!resource_wrapper) return 0;
    mi::base::Handle<const MDL::IValue> value(resource_wrapper->get_internal_value());
    if (!value) return 0;

    mi::base::Handle<const MDL::IValue_resource> resource_int(
        value->get_interface<MDL::IValue_resource>());
    if (!resource_int) return 0;

    return get_known_resource_index(db_transaction, resource_int.get());
}

/// Returns the resource index for use in an \c ITarget_argument_block of resources already
/// known when this \c Target_code object was generated.
mi::Uint32 Target_code::get_known_resource_index(
    MI::DB::Transaction* transaction,
    MI::MDL::IValue_resource const *resource) const
{
    DB::Tag tag = DB::Tag(resource->get_value());

    if (m_native_code.is_valid_interface()) {
        return m_native_code->get_known_resource_index(tag.get_uint());
    }
    bool is_resolved = true;
    char const *db_name = transaction->tag_to_name(tag);
    char const *mdl_url = NULL, *owner_module = NULL;
    if (db_name == NULL) {
        mdl_url = resource->get_unresolved_mdl_url();
        if (!mdl_url || mdl_url[0] == '\0') // none given
            return 0;
        is_resolved = false;
        owner_module = resource->get_owner_module();
        if (owner_module == NULL)
            owner_module = "";
    }

    switch (resource->get_kind()) {
    case MDL::IValue::VK_TEXTURE:
    {
        // skip first texture, which is always the invalid resource
        for (mi::Size i = 1, n = get_texture_count(); i < n; ++i) {

            if (is_resolved) {
                const char *texture_db_name = get_texture(i);
                if (texture_db_name && texture_db_name[0] != '\0') {
                    if (strcmp(texture_db_name, db_name) == 0)
                        return mi::Uint32(i);
                }
            } else {
                // handle unresolved resources

                const char *texture_mdl_url = get_texture_url(i);
                if (!texture_mdl_url || texture_mdl_url[0] == '\0')
                    continue;

                if (strcmp(mdl_url, texture_mdl_url) == 0) {

                    // also compare owner modules
                    const char *texture_owner_module = get_texture_owner_module(i);
                    if (!texture_owner_module)
                        texture_owner_module = "";

                    if (strcmp(owner_module, texture_owner_module) == 0)
                        return mi::Uint32(i);
                }
            }
        }
        return 0;
    }

    case MDL::IValue::VK_LIGHT_PROFILE:
    {
        // skip first light profile, which is always the invalid resource
        for (mi::Size i = 1, n = get_light_profile_count(); i < n; ++i) {
            const char *lp_db_name = get_light_profile(i);
            if (lp_db_name)
                if (strcmp(lp_db_name, db_name) == 0)
                    return mi::Uint32(i);
        }
        return 0;
    }

    case MDL::IValue::VK_BSDF_MEASUREMENT:
    {
        // skip first bsdf measurement, which is always the invalid resource
        for (mi::Size i = 1, n = get_bsdf_measurement_count(); i < n; ++i) {
            const char *bm_db_name = get_bsdf_measurement(i);
            if (bm_db_name)
                if (strcmp(bm_db_name, db_name) == 0)
                    return mi::Uint32(i);
        }

        return 0;
    }

    case MDL::IValue::VK_BOOL:
    case MDL::IValue::VK_INT:
    case MDL::IValue::VK_ENUM:
    case MDL::IValue::VK_FLOAT:
    case MDL::IValue::VK_DOUBLE:
    case MDL::IValue::VK_STRING:
    case MDL::IValue::VK_VECTOR:
    case MDL::IValue::VK_MATRIX:
    case MDL::IValue::VK_COLOR:
    case MDL::IValue::VK_ARRAY:
    case MDL::IValue::VK_STRUCT:
    case MDL::IValue::VK_INVALID_DF:
    case MDL::IValue::VK_FORCE_32_BIT:
        ASSERT(M_BACKENDS, !"Unsupported MDL resource type");
        break;
    }

    // not found -> invalid resource reference
    return 0;
}

// -------------------------------------------------------------------------------------------------
// Serialization and Deserialization of the Target Code and its data structures

const MI::SERIAL::Serializable* Callable_function_info::serialize(MI::SERIAL::Serializer* serializer) const
{
    serializer->write(m_name);
    serializer->write(static_cast<mi::Sint32>(m_dist_kind));
    serializer->write(static_cast<mi::Sint32>(m_kind));
    MI::SERIAL::write(serializer, m_prototypes);
    serializer->write(m_arg_block_index);
    MI::SERIAL::write(serializer, m_df_handle_name_table);
    serializer->write(static_cast<mi::Sint32>(m_state_usage));
    return this + 1;
}

MI::SERIAL::Serializable* Callable_function_info::deserialize(MI::SERIAL::Deserializer* deserializer)
{
    deserializer->read(&m_name);
    mi::Sint32 value;
    deserializer->read(&value);
    m_dist_kind = static_cast<mi::neuraylib::ITarget_code::Distribution_kind>(value);
    deserializer->read(&value);
    m_kind = static_cast<mi::neuraylib::ITarget_code::Function_kind>(value);
    MI::SERIAL::read(deserializer, &m_prototypes);
    deserializer->read(&m_arg_block_index);
    MI::SERIAL::read(deserializer, &m_df_handle_name_table);
    deserializer->read(&value);
    m_state_usage = static_cast<mi::neuraylib::ITarget_code::State_usage>(value);
    return this + 1;
}

const MI::SERIAL::Serializable* Target_code::Resource_info::serialize(MI::SERIAL::Serializer* serializer) const
{
    serializer->write(m_db_name);
    serializer->write(m_mdl_url);
    serializer->write(m_owner_module);
    return this + 1;
}

MI::SERIAL::Serializable* Target_code::Resource_info::deserialize(MI::SERIAL::Deserializer* deserializer)
{
    deserializer->read(&m_db_name);
    deserializer->read(&m_mdl_url);
    deserializer->read(&m_owner_module);
    return this + 1;
}

const MI::SERIAL::Serializable* Target_code::Texture_info::serialize(MI::SERIAL::Serializer* serializer) const
{
    Target_code::Resource_info::serialize(serializer);
    serializer->write(m_gamma);
    serializer->write(static_cast<mi::Sint32>(m_texture_shape));
    serializer->write(static_cast<mi::Sint32>(m_df_data_kind));
    return this + 1;
}

MI::SERIAL::Serializable* Target_code::Texture_info::deserialize(MI::SERIAL::Deserializer* deserializer)
{
    Target_code::Resource_info::deserialize(deserializer);
    deserializer->read(&m_gamma);
    mi::Sint32 value;
    deserializer->read(&value);
    m_texture_shape = static_cast<mi::neuraylib::ITarget_code::Texture_shape>(value);
    deserializer->read(&value);
    m_df_data_kind = static_cast<mi::mdl::IValue_texture::Bsdf_data_kind>(value);
    return this + 1;
}

bool Target_code::supports_serialization() const
{
    return m_backend_kind == mi::neuraylib::IMdl_backend_api::Mdl_backend_kind::MB_CUDA_PTX ||
           m_backend_kind == mi::neuraylib::IMdl_backend_api::Mdl_backend_kind::MB_HLSL;
}

const MI::SERIAL::Serializable* Target_code::Segment::serialize(MI::SERIAL::Serializer* serializer) const
{
    serializer->write(m_name);
    MI::SERIAL::write(serializer, m_data);
    return this + 1;
}

MI::SERIAL::Serializable* Target_code::Segment::deserialize(MI::SERIAL::Deserializer* deserializer)
{
    deserializer->read(&m_name);
    MI::SERIAL::read(deserializer, &m_data);
    return this + 1;
}

namespace {

    static const char* MDL_TCI_HEADER = "MDLTCI\0\0";                     // 8 byte marker
    static const mi::Uint16 MDL_TCI_CURRENT_PROTOCOL = (1u << 8u) + 1u;   // 1.1
    static const std::string MDL_SDK_VERSION = MI::VERSION::get_platform_version();
    static const std::string MDL_SDK_OS = MI::VERSION::get_platform_os();

    /// Wraps a memory block identified by a pointer and a length as mi::neuraylib::IBuffer.
    class Copy_buffer
        : public mi::base::Interface_implement<mi::neuraylib::IBuffer>,
        public boost::noncopyable
    {
    public:
        Copy_buffer(const mi::Uint8* data, mi::Size data_size)
            : m_data(data, data + data_size)
        {
        }
        const mi::Uint8* get_data() const { return m_data.data(); }
        mi::Size get_data_size() const { return m_data.size(); }
    private:
        const std::vector<mi::Uint8> m_data;
    };

} // anonymous namespace

const mi::neuraylib::IBuffer* Target_code::serialize(mi::neuraylib::IMdl_execution_context* context) const
{
    if (context)
        context->clear_messages();

    if (!supports_serialization())
    {
        if (context)
            context->add_message(mi::neuraylib::IMessage::MSG_COMILER_BACKEND,
                mi::base::details::MESSAGE_SEVERITY_ERROR, -1, "Serialization failed. "
                "The back-end that produces this target code does not support serialization.");
        return nullptr;
    }

    // options
    bool serialize_instance_data;
    if (!context ||
        context->get_option("serialize_class_instance_data", serialize_instance_data) != 0)
            serialize_instance_data = true;

    SERIAL::Buffer_serializer serializer;

    // header
    serializer.write(MDL_TCI_HEADER, 8);
    serializer.write(MDL_TCI_CURRENT_PROTOCOL);
    serializer.write(MDL_SDK_VERSION);
    serializer.write(MDL_SDK_OS);

    // target code info data
    serializer.write(static_cast<mi::Sint32>(m_backend_kind));
    serializer.write(m_code);
    MI::SERIAL::write(&serializer, m_code_segments);
    MI::SERIAL::write(&serializer, m_code_segment_descriptions);
    MI::SERIAL::write(&serializer, m_callable_function_infos);
    serializer.write(m_body_texture_count);
    serializer.write(m_body_light_profile_count);
    serializer.write(m_body_bsdf_measurement_count);

    if (serialize_instance_data)
    {
        MI::SERIAL::write(&serializer, m_texture_table);
        MI::SERIAL::write(&serializer, m_light_profile_table);
        MI::SERIAL::write(&serializer, m_bsdf_measurement_table);
    }
    else
    {
        auto copy_texture_table = m_texture_table;
        copy_texture_table.resize(m_body_texture_count);
        MI::SERIAL::write(&serializer, copy_texture_table);

        auto copy_light_profile_table = m_light_profile_table;
        copy_light_profile_table.resize(m_body_light_profile_count);
        MI::SERIAL::write(&serializer, copy_light_profile_table);

        auto copy_bsdf_measurement_table = m_bsdf_measurement_table;
        copy_bsdf_measurement_table.resize(m_body_bsdf_measurement_count);
        MI::SERIAL::write(&serializer, copy_bsdf_measurement_table);
    }

    MI::SERIAL::write(&serializer, m_string_constant_table);
    serializer.write(m_render_state_usage);
    MI::SERIAL::write(&serializer, m_data_segments);

    size_t arg_layout_count = m_cap_arg_layouts.size();
    serializer.write_size_t(arg_layout_count);
    for (size_t i = 0; i < arg_layout_count; ++i) {
        mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> internal_layout(
            m_cap_arg_layouts[i]->get_internal_layout());

        // Handle different types if required
        mi::mdl::Generated_code_value_layout const* internal_layout_impl =
            mi::mdl::impl_cast<mi::mdl::Generated_code_value_layout>(internal_layout.get());
            //static_cast<mi::mdl::Generated_code_value_layout const*>();

        size_t data_size = 0;
        char const* data = internal_layout_impl->get_layout_data(data_size);
        bool map_strings = internal_layout_impl->get_strings_mapped_to_ids();

        serializer.write_size_t(data_size);
        serializer.write(data, data_size);
        serializer.write(map_strings);
    }

    size_t arg_block_count = serialize_instance_data ? m_cap_arg_blocks.size() : 0;
    serializer.write_size_t(arg_block_count);
    for (size_t i = 0; i < arg_block_count; ++i) {
        mi::base::Handle<mi::neuraylib::ITarget_argument_block const> block(m_cap_arg_blocks[i]);
        size_t block_size = size_t(block->get_size());
        const char* block_data = block->get_data();
        serializer.write_size_t(block_size);
        if (block_size == 0)
            continue;
        serializer.write(block_data, block_size);
    }

    serializer.write(m_string_args_mapped_to_ids);
    serializer.write(m_use_builtin_resource_handler);

    mi::base::Handle<mi::neuraylib::IBuffer> buffer(new Copy_buffer(
        serializer.get_buffer(), serializer.get_buffer_size()));
    buffer->retain();
    return buffer.get();
}

bool Target_code::deserialize(
    mi::mdl::ICode_generator* code_gen,
    const mi::neuraylib::IBuffer* buffer,
    mi::neuraylib::IMdl_execution_context* context)
{
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset(buffer->get_data(), buffer->get_data_size());

    // header
    std::vector<char> mdl_tci_header(8, '\0');
    if(buffer->get_data_size() > 8)
        deserializer.read(mdl_tci_header.data(), 8);
    if (memcmp(mdl_tci_header.data(), MDL_TCI_HEADER, mdl_tci_header.size()) != 0)
    {
        if (context)
            context->add_message(mi::neuraylib::IMessage::MSG_COMILER_BACKEND,
                mi::base::details::MESSAGE_SEVERITY_ERROR, -2, "Deserialization failed. "
                "Corrupt input data, invalid header.");
        return false;
    }

    mi::Uint16 mdl_tci_protocol_version;
    deserializer.read(&mdl_tci_protocol_version);
    if (mdl_tci_protocol_version != MDL_TCI_CURRENT_PROTOCOL)
    {
        if (context)
            context->add_message(mi::neuraylib::IMessage::MSG_COMILER_BACKEND,
                mi::base::details::MESSAGE_SEVERITY_INFO, 1, "Deserialization invalid. "
                "Protocol version mismatch.");
        return false;
    }

    std::string mdl_sdk_version;
    std::string mdl_sdk_os;
    deserializer.read(&mdl_sdk_version);
    deserializer.read(&mdl_sdk_os);
    if (mdl_sdk_version != MDL_SDK_VERSION || mdl_sdk_os != MDL_SDK_OS)
    {
        if (context)
            context->add_message(mi::neuraylib::IMessage::MSG_COMILER_BACKEND,
                mi::base::details::MESSAGE_SEVERITY_INFO, 2, "Deserialization invalid. "
                "MDL SDK version mismatch.");
        return false;
    }

    // target code info data
    mi::Sint32 value;
    deserializer.read(&value);
    m_backend_kind = static_cast<mi::neuraylib::IMdl_backend_api::Mdl_backend_kind>(value);
    deserializer.read(&m_code);
    MI::SERIAL::read(&deserializer, &m_code_segments);
    MI::SERIAL::read(&deserializer, &m_code_segment_descriptions);
    MI::SERIAL::read(&deserializer, &m_callable_function_infos);
    deserializer.read(&m_body_texture_count);
    deserializer.read(&m_body_light_profile_count);
    deserializer.read(&m_body_bsdf_measurement_count);
    MI::SERIAL::read(&deserializer, &m_texture_table);
    MI::SERIAL::read(&deserializer, &m_light_profile_table);
    MI::SERIAL::read(&deserializer, &m_bsdf_measurement_table);
    MI::SERIAL::read(&deserializer, &m_string_constant_table);
    deserializer.read(&m_render_state_usage);
    MI::SERIAL::read(&deserializer, &m_data_segments);

    // Argument Layouts
    size_t arg_layout_count;
    deserializer.read_size_t(&arg_layout_count);
    m_cap_arg_layouts.resize(arg_layout_count);
    for (size_t i = 0; i < arg_layout_count; ++i)
    {
        // Handle different types if required
        mi::base::Handle<mi::mdl::IGenerated_code_value_layout> internal_layout;

        switch (m_backend_kind)
        {
            case mi::neuraylib::IMdl_backend_api::MB_HLSL:
            case mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX:
            {
                mi::base::Handle<mi::mdl::ICode_generator_jit> code_gen_jit(
                    code_gen->get_interface<mi::mdl::ICode_generator_jit>());
                internal_layout = code_gen_jit->create_value_layout();
                break;
            }
            default:
                ASSERT(M_BACKENDS, false && "Back-end not supported. Serialization failed.");
                return false;
        }

        if (!internal_layout)
            continue;

        mi::mdl::Generated_code_value_layout* internal_layout_impl =
            mi::mdl::impl_cast<mi::mdl::Generated_code_value_layout>(internal_layout.get());

        // deserialize
        size_t data_size;
        deserializer.read_size_t(&data_size);
        std::vector<char> layout_data(data_size);
        deserializer.read(layout_data.data(), data_size);
        bool map_strings;
        deserializer.read(&map_strings);

        internal_layout_impl->set_layout_data(layout_data.data(), data_size);
        internal_layout_impl->set_strings_mapped_to_ids(map_strings);

        // compose the layout
        m_cap_arg_layouts[i] = new MI::BACKENDS::Target_value_layout(
            internal_layout_impl, map_strings);
    }

    // Argument Blocks
    size_t arg_block_count;
    deserializer.read_size_t(&arg_block_count);
    m_cap_arg_blocks.resize(arg_block_count);
    for (size_t i = 0; i < arg_block_count; ++i)
    {
        size_t block_size;
        deserializer.read_size_t(&block_size);

        if (block_size == 0) {
            m_cap_arg_blocks[i] = nullptr;
            continue;
        }

        m_cap_arg_blocks[i] = new Target_argument_block(block_size);
        deserializer.read(m_cap_arg_blocks[i]->get_data(), block_size);
    }

    deserializer.read(&m_string_args_mapped_to_ids);
    deserializer.read(&m_use_builtin_resource_handler);
    return true;
}

} // namespace BACKENDS

} // namespace MI

