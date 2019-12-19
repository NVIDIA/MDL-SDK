/***************************************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/mdl/mdl_code_generators.h>
#include <mi/neuraylib/icompiled_material.h>
#include <render/mdl/runtime/i_mdlrt_resource_handler.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <mdl/jit/generator_jit/generator_jit_libbsdf_data.h>
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
    mi::Uint32 get_resource_index(MI::MDL::IValue_resource const *resource) NEURAY_FINAL {
        return m_target_code->get_known_resource_index(m_transaction, resource);
    }


    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    mi::Uint32 get_string_index(MI::MDL::IValue_string const *s) NEURAY_FINAL {
        return m_target_code->get_string_index(s->get_value());
    }

private:
    DB::Transaction* m_transaction;
    Target_code *m_target_code;
};

} // anonymous


// --------------------- Target code --------------------

// Constructor from executable code.
Target_code::Target_code(
    mi::mdl::IGenerated_code_executable* code,
    MI::DB::Transaction* transaction,
    bool string_ids,
    bool use_derivatives,
    bool use_builtin_resource_handler)
  : m_native_code(),
    m_code(),
    m_code_segments(),
    m_code_segment_descriptions(),
    m_callable_function_infos(),
    m_texture_table(),
    m_light_profile_table(),
    m_bsdf_measurement_table(),
    m_string_constant_table(),
    m_data_segments(),
    m_data(),
    m_cap_arg_layouts(),
    m_cap_arg_blocks(),
    m_rh( NULL),
    m_render_state_usage(~0u),
    m_string_args_mapped_to_ids(string_ids),
    m_use_builtin_resource_handler(use_builtin_resource_handler)
{
    finalize(code, transaction, use_derivatives);

    size_t num_layouts = code->get_captured_argument_layouts_count();
    m_cap_arg_blocks.resize(num_layouts);   // already prepare the empty argument block slots

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
    bool string_ids)
  : m_native_code(),
    m_code(),
    m_code_segments(),
    m_code_segment_descriptions(),
    m_callable_function_infos(),
    m_texture_table(),
    m_light_profile_table(),
    m_bsdf_measurement_table(),
    m_string_constant_table(),
    m_data_segments(),
    m_data(),
    m_cap_arg_layouts(),
    m_cap_arg_blocks(),
    m_rh( NULL),
    m_render_state_usage( ~0u),
    m_string_args_mapped_to_ids(string_ids),
    m_use_builtin_resource_handler(true)
{
}

Target_code::~Target_code()
{
    for (mi::Size i = 0, n = m_data.size(); i < n; ++i) {
        const unsigned char* data = m_data[i];
        delete [] data;
    }

    if (m_native_code.is_valid_interface()) {
        m_native_code->term();

        delete m_rh;
        m_rh = NULL;
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
                code->get_function_arg_block_layout_index(i)));
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

const char* Target_code::get_callable_function( mi::Size index) const
{
    if( index < m_callable_function_infos.size())
        return m_callable_function_infos[ index].m_name.c_str();
    return NULL;
}

// Returns the prototype of a callable function in the target code.
const char* Target_code::get_callable_function_prototype(
    mi::Size index,
    Prototype_language lang) const
{
    if( index >= m_callable_function_infos.size() ||
            lang >= m_callable_function_infos[ index].m_prototypes.size()) {
        return NULL;
    }
    return m_callable_function_infos[ index].m_prototypes[ lang].c_str();
}

// Returns the kind of a callable function in the target code.
mi::neuraylib::ITarget_code::Distribution_kind 
    Target_code::get_callable_function_distribution_kind(
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
Size Target_code::get_callable_function_argument_block_index( mi::Size index) const
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
const char* Target_code::get_callable_function_df_handle(Size func_index, Size handle_index) const
{
    if( func_index >= m_callable_function_infos.size() ||
            handle_index >= m_callable_function_infos[ func_index].m_df_handle_name_table.size())
        return NULL;

    return m_callable_function_infos[ func_index].m_df_handle_name_table[ handle_index].c_str();
}

mi::Size Target_code::get_texture_count() const
{
    return m_texture_table.size();
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

        size_t w=0, h=0, d=0;
        mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data_resolution(
            m_texture_table[index].get_df_data_kind(), w, h, d);
        rx = w;
        ry = h;
        rz = d;
        size_t s;
        return reinterpret_cast<const mi::Float32*>(
            mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data(m_texture_table[index].get_df_data_kind(), s));
    }
    return nullptr;
}

mi::Size Target_code::get_light_profile_count() const
{
    return m_light_profile_table.size();
}

const char* Target_code::get_light_profile( mi::Size index) const
{
    if( index < m_light_profile_table.size()) {
        return m_light_profile_table[index].c_str();
    }
    return NULL;
}

Size Target_code::get_bsdf_measurement_count() const
{
    return m_bsdf_measurement_table.size();
}

const char* Target_code::get_bsdf_measurement(mi::Size index) const
{
    if( index < m_bsdf_measurement_table.size()) {
        return m_bsdf_measurement_table[index].c_str();
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

const char* Target_code::get_ro_data_segment_name( mi::Size index) const
{
    if( index >= m_data_segments.size())
        return NULL;
    const Segment& segment = m_data_segments[index];
    return segment.get_name();
}

mi::Size Target_code::get_ro_data_segment_size( mi::Size index) const
{
    if( index >= m_data_segments.size())
        return 0;
    const Segment& segment = m_data_segments[index];
    return segment.get_size();
}

const char* Target_code::get_ro_data_segment_data( mi::Size index) const
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

Target_code::State_usage Target_code::get_render_state_usage() const
{
    return m_render_state_usage;
}

size_t Target_code::add_function(
    const std::string& name,
    Distribution_kind dist_kind,
    Function_kind kind,
    mi::Size arg_block_index)
{
    size_t idx = m_callable_function_infos.size();
    m_callable_function_infos.push_back( 
        Callable_function_info( name, dist_kind, kind, arg_block_index));
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
        m_callable_function_infos[ index].m_prototypes.resize( lang + 1);
    }
    m_callable_function_infos[ index].m_prototypes[ lang] = prototype;
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
        m_texture_table.resize( index + 1, Texture_info(
            /*db_name=*/"",
            /*mdl_url=*/"",
            /*owner=*/"",
            /*gamma=*/0.0f,
            /*texture_shape=*/Texture_shape_invalid,
            /*df_data_kind=*/ mi::mdl::IValue_texture::BDK_NONE));
    }
    std::string owner, url;
    size_t p = mdl_url.find('|');
    if (p != std::string::npos) {
        owner = mdl_url.substr(0, p);
        url = mdl_url.substr(p + 1);
    }
    else
        url = mdl_url;

    m_texture_table[index] = Texture_info(name, url, owner, gamma, shape, df_data_kind);
}

// Registers a used light profile index.
void Target_code::add_light_profile_index( size_t index, const std::string& name)
{
    if( index >= m_light_profile_table.size()) {
        m_light_profile_table.resize( index + 1, "");
    }
    m_light_profile_table[ index] = name;
}

// Registers a used bsdf measurement index.
void Target_code::add_bsdf_measurement_index( size_t index, const std::string& name)
{
    if( index >= m_bsdf_measurement_table.size()) {
        m_bsdf_measurement_table.resize( index + 1, "");
    }
    m_bsdf_measurement_table[ index] = name;
}

// Registers a used string constant index.
void Target_code::add_string_constant_index(size_t index, const std::string& scons)
{
    if (index >= m_string_constant_table.size()) {
        m_string_constant_table.resize(index + 1, "");
    }
    m_string_constant_table[index] = scons;
}

void Target_code::add_ro_segment( const char* name, const unsigned char* data, mi::Size size)
{
    unsigned char* segment = NULL;
    if( size > 0) {
        segment = new unsigned char[size];
        m_data.push_back( segment);

        memcpy( segment, data, size);
    }
    m_data_segments.push_back( Segment( name, segment, size));
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
    ASSERT(M_BACKENDS, m_cap_arg_blocks.size() == m_cap_arg_layouts.size());
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

// Get the captured arguments block layout if available.
mi::neuraylib::ITarget_value_layout const *Target_code::get_argument_block_layout(Size index) const
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
    if ( !material || index >= m_cap_arg_layouts.size())
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
    if ( !args || index >= m_cap_arg_blocks.size())
        return;

    // Argument block already initialized? Do nothing
    if ( m_cap_arg_blocks[index])
        return;

    ASSERT( M_BACKENDS, index < m_cap_arg_layouts.size() && m_cap_arg_layouts[index] &&
        "captured arguments but no layout");
    if ( index >= m_cap_arg_layouts.size() || !m_cap_arg_layouts[index])
        return;

    Target_value_layout const *layout = m_cap_arg_layouts[index].get();
    mi::Size num_args = args->get_size();
    if ( num_args != layout->get_num_elements())
        return;

    Target_argument_block *block = new Target_argument_block( layout->get_size());
    m_cap_arg_blocks[index] = mi::base::make_handle(block);

    Target_resource_callback_internal resource_callback(transaction, this);

    for ( mi::Size i = 0; i < num_args; ++i) {
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
mi::Uint32 Target_code::get_string_index(char const *s) const
{
    if (s == NULL)
        return 0u;

    // slow linear search here, but the number of string is expected to be small
    std::string str(s);
    for (size_t i = 1, n = m_string_constant_table.size(); i < n; ++i) {
        if (m_string_constant_table[i] == str)
            return mi::Uint32(i);
    }
    return 0u;
}

/// Returns the resource index for use in an \c ITarget_argument_block of resources already
/// known when this \c ITarget_code object was generated.
mi::Uint32 Target_code::get_known_resource_index(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IValue_resource const *resource) const
{
    if ( transaction == NULL || resource == NULL) return 0;

    // TODO: This should be moved into api/api/mdl to not have mi::neuraylib objects in this module
    NEURAY::Transaction_impl *transaction_impl =
        static_cast<NEURAY::Transaction_impl *>( transaction);
    DB::Transaction *db_transaction = transaction_impl->get_db_transaction();
    ASSERT( M_BACKENDS, db_transaction);

    // copied from NEURAY::get_internal_value<T>
    mi::base::Handle<const NEURAY::IValue_wrapper> resource_wrapper(
        resource->get_interface<NEURAY::IValue_wrapper>());
    if ( !resource_wrapper) return 0;
    mi::base::Handle<const MDL::IValue> value( resource_wrapper->get_internal_value());
    if ( !value) return 0;

    mi::base::Handle<const MDL::IValue_resource> resource_int(
        value->get_interface<MDL::IValue_resource>());
    if ( !resource_int) return 0;

    return get_known_resource_index( db_transaction, resource_int.get());
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

} // namespace BACKENDS

} // namespace MI

