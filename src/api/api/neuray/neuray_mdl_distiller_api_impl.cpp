/***************************************************************************************************
 * Copyright (c) 2017-2024, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the IMdl_distiller_api implementation.
 **/

#include "pch.h"

#include "neuray_mdl_distiller_api_impl.h"

#include <string>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/ivector.h>
#include <mdl/distiller/dist/i_dist.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <render/baker/baker/i_baker.h>
#include <mi/mdl/mdl_distiller_rules.h>
#include <mi/mdl/mdl_distiller_options.h>
#include <mi/mdl/mdl_distiller_plugin_api.h>

#include "neuray_compiled_material_impl.h"
#include "neuray_transaction_impl.h"

namespace MI {

namespace NEURAY {

namespace {

// Unused, kept for possible later use
// /// Reads a string option value from an IMap.
// static void get_option( const mi::IMap* options, const char* name, std::string& value)
// {
//     if( !options)
//         return;
//     mi::base::Handle<const mi::IString> istring(
//         options->get_value<const mi::IString>( name));
//     if( istring.is_valid_interface())
//         value = istring->get_c_str();
// }

/// Reads a bool option value from an IMap.
void get_option( const mi::IMap* options, const char* name, bool& value)
{
    if( !options)
        return;
    mi::base::Handle<const mi::IBoolean> ivalue(
        options->get_value<const mi::IBoolean>( name));
    if( ivalue.is_valid_interface())
        ivalue->get_value( value);
}

/// Reads a Sint32 option value from an IMap.
void get_option( const mi::IMap* options, const char* name, int& value)
{
    if( !options)
        return;
    mi::base::Handle<const mi::ISint32> ivalue(
        options->get_value<const mi::ISint32>( name));
    if( ivalue.is_valid_interface())
        ivalue->get_value( value);
}

/// Reads a Float32 option value from an IMap.
void get_option( const mi::IMap* options, const char* name, float& value)
{
    if( !options)
        return;
    mi::base::Handle<const mi::IFloat32> ivalue(
        options->get_value<const mi::IFloat32>( name));
    if( ivalue.is_valid_interface())
        ivalue->get_value( value);
}

} // anonymous namespace

Mdl_distiller_api_impl::Mdl_distiller_api_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray),
    m_dist_module( true),
    m_baker_module( true)
{
}

Mdl_distiller_api_impl::~Mdl_distiller_api_impl()
{
    m_neuray = nullptr;
}

mi::Size Mdl_distiller_api_impl::get_target_count() const
{
    return m_dist_module->get_target_count();
}

const char* Mdl_distiller_api_impl::get_target_name( mi::Size index) const
{
    return m_dist_module->get_target_name(index);
}

mi::Size Mdl_distiller_api_impl::get_required_module_count(const char *target) const
{
    return m_dist_module->get_required_module_count(target);
}

const char* Mdl_distiller_api_impl::get_required_module_name(
    const char *target, mi::Size index) const
{
    return m_dist_module->get_required_module_name(target, index);
}

const char* Mdl_distiller_api_impl::get_required_module_code(
    const char *target, mi::Size index) const
{
    return m_dist_module->get_required_module_code(target, index);
}

namespace {

class Debug_print_stream : public mi::mdl::IOutput_stream {
private:
    char buffer[120] = {0};
    size_t buf_len{0};

public:
    /// Write a character to the output stream.
    virtual void write_char(char c) {
        if (c == '\n') {
            flush();
            return;
        }
        if (buf_len >= sizeof(buffer) - 2) {
            buffer[buf_len++] = c;
            flush();
            return;
        }
        buffer[buf_len++] = c;
    }

    /// Write a string to the stream.
    virtual void write(const char *string) {
        for (char const *p = string; *p; ++p) {
            write_char(*p);
        }
    }

    /// Flush stream.
    virtual void flush() {
        buffer[buf_len] = '\0';
        LOG::mod_log->info(SYSTEM::M_DIST, LOG::ILogger::C_COMPILER, ">>> %s", buffer);
        buf_len = 0;
    }

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully
    /// removed, false otherwise
    virtual bool unput(char c) {
        // unsupported
        return false;
    }

    // from IInterface: This object is not reference counted.
    virtual Uint32 retain() const { return 1; }
    virtual Uint32 release() const { return 1; }
    virtual const mi::base::IInterface* get_interface(mi::base::Uuid const &interface_id) const {
        return nullptr;
    }
    virtual mi::base::IInterface* get_interface(mi::base::Uuid const &interface_id) {
        return nullptr;
    }
    virtual mi::base::Uuid get_iid() const { return IID(); }

};

/// The event handler for distilling events.
class Rule_matcher_event : public mi::mdl::IRule_matcher_event
{
    /// Options controlling the trace level.
    const mi::mdl::Distiller_options* m_options;

    /// A DAG path is checked against a rule set.
    ///
    /// \param rule_set_name   the name of the rule set
    /// \param dag_path        the DAG path to a node that is currently checked
    virtual void path_check_event(
        char const *rule_set_name,
        char const *dag_path)
    {
        if ( m_options->trace >=2) {
            LOG::mod_log->info(
                SYSTEM::M_DIST,
                LOG::ILogger::C_COMPILER,
                "Check rule set '%s' on path '%s'",
                rule_set_name, dag_path);
        }
    }

    /// A rule has matched.
    ///
    /// \param rule_set_name   the name of the rule set
    /// \param rule_name       the name of the rule that matched
    /// \param file_name       if non-NULL, the file name where the rule was declared
    /// \param line_number     if non-ZERO, the line number where the rule was declared
    virtual void rule_match_event(
        char const *rule_set_name,
        unsigned   rule_id,
        char const *rule_name,
        char const *file_name,
        unsigned   line_number)
    {
        if ( m_options->trace >=1) {
            if (file_name == nullptr)
                file_name = "<unknown>";
            LOG::mod_log->info(
                SYSTEM::M_DIST,
                LOG::ILogger::C_COMPILER,
                "Rule <%u> matches in %s(%u): '%s' for node '%s'",
                rule_id, file_name, line_number, rule_set_name, rule_name);
        }
    }

    /// A postcondition has failed.
    ///
    /// \param rule_set_name   the name of the rule set
    virtual void postcondition_failed(
        char const *rule_set_name)
    {
        LOG::mod_log->error(
            SYSTEM::M_DIST,
            LOG::ILogger::C_COMPILER, "Postcondition for '%s' failed", rule_set_name);
    }

    /// A postcondition has failed for a given path.
    ///
    /// \param path   the path that failed
    virtual void postcondition_failed_path(
        char const *path)
    {
        LOG::mod_log->error(
            SYSTEM::M_DIST,
            LOG::ILogger::C_COMPILER, "Postcond check failed for path '%s'.", path);
    }

    virtual void debug_print(
        mi::mdl::IDistiller_plugin_api &plugin_api,
        char const *rule_set_name,
        unsigned   rule_id,
        char const *rule_name,
        char const *file_name,
        unsigned   line_number,
        char const *var_name,
        mi::mdl::DAG_node const *value)
    {
        if (file_name == nullptr)
            file_name = "<unknown>";
        if (var_name == nullptr)
            var_name = "<unknown>";
        LOG::mod_log->info(
            SYSTEM::M_DIST,
            LOG::ILogger::C_COMPILER,
            "Rule <%u> %s:%u: matching rule %s::%s:",
            rule_id, file_name, line_number, rule_set_name, rule_name);
        LOG::mod_log->info(
            SYSTEM::M_DIST,
            LOG::ILogger::C_COMPILER,
            ">>> %s = ", var_name);
        Debug_print_stream outs;
        plugin_api.debug_node(&outs, value);
        outs.flush();
    }

public:
    /// Initialize match event handler with options to control trace level.
    Rule_matcher_event( const mi::mdl::Distiller_options* options) : m_options(options) {}

};

}  // anonymous

mi::neuraylib::ICompiled_material* Mdl_distiller_api_impl::distill_material(
    const mi::neuraylib::ICompiled_material* material,
    const char* target,
    const mi::IMap* distiller_options,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !material || !target) {
        *errors = -1;
        return nullptr;
    }

    mi::mdl::Distiller_options options;
    get_option( distiller_options, "layer_normal", options.layer_normal);
    get_option( distiller_options, "top_layer_weight", options.top_layer_weight);
    get_option( distiller_options, "merge_metal_and_base_color",
                options.merge_metal_and_base_color);
    get_option( distiller_options, "merge_transmission_and_base_color",
                options.merge_transmission_and_base_color);
    get_option( distiller_options, "target_material_model_mode",
                options.target_material_model_mode);
    get_option( distiller_options, "_dbg_quiet", options.quiet);
    get_option( distiller_options, "_dbg_verbosity", options.verbosity);
    get_option( distiller_options, "_dbg_trace", options.trace);
    get_option( distiller_options, "_dbg_debug_print", options.debug_print);

    const Compiled_material_impl* material_impl
        = static_cast<const Compiled_material_impl*>( material);
    const MDL::Mdl_compiled_material* db_material
        = material_impl->get_db_element();

    Transaction_impl* transaction = material_impl->get_transaction();
    DB::Transaction* db_transaction = material_impl->get_db_transaction();

    MDL::load_distilling_support_module(db_transaction);

    MDL::Mdl_call_resolver resolver(db_transaction);

    MDL::Mdl_material_instance_builder builder;
    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> dag_material_instance(
        builder.create_material_instance( db_transaction, db_material));
    ASSERT( M_NEURAY_API, dag_material_instance);
    if( !dag_material_instance)
        return nullptr;

    Rule_matcher_event event_handler( &options);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag::IMaterial_instance>
        new_dag_material_instance(
            m_dist_module->distill(
                resolver,
                (options.trace != 0 || options.debug_print) ? &event_handler : nullptr,
                dag_material_instance.get(),
                target,
                &options,
                errors));
    if( !new_dag_material_instance)
        return nullptr;

    auto new_db_material = std::make_shared<MDL::Mdl_compiled_material>(
        db_transaction,
        new_dag_material_instance.get(),
        /*module_name*/ nullptr,
        db_material->get_mdl_meters_per_scene_unit(),
        db_material->get_mdl_wavelength_min(),
        db_material->get_mdl_wavelength_max(),
        /*load_resources*/ false);

    mi::neuraylib::ICompiled_material* new_material
        = transaction->create<mi::neuraylib::ICompiled_material>( "__Compiled_material");
    Compiled_material_impl* new_material_impl
        = static_cast<Compiled_material_impl*>( new_material);
    new_material_impl->swap( *new_db_material);

    return new_material;
}

const mi::neuraylib::IBaker* Mdl_distiller_api_impl::create_baker(
    const mi::neuraylib::ICompiled_material* material,
    const char* path,
    mi::neuraylib::Baker_resource resource,
    Uint32 gpu_device_id) const
{
    const Compiled_material_impl* material_impl
        = static_cast<const Compiled_material_impl*>( material);
    DB::Transaction* db_transaction = material_impl->get_db_transaction();
    const MDL::Mdl_compiled_material* db_material = material_impl->get_db_element();

    std::string pixel_type;
    bool is_uniform;
    mi::base::Handle<const BAKER::IBaker_code> baker_code(
        m_baker_module->create_baker_code(
            db_transaction, db_material, path, resource, gpu_device_id, pixel_type, is_uniform));
    if( !baker_code)
        return nullptr;

    return new Baker_impl( db_transaction, baker_code.get(), pixel_type.c_str(), is_uniform);
}

mi::Sint32 Mdl_distiller_api_impl::start()
{
    m_dist_module.set();
    m_baker_module.set();
    return 0;
}

mi::Sint32 Mdl_distiller_api_impl::shutdown()
{
    m_baker_module.reset();
    m_dist_module.reset();
    return 0;
}

Baker_impl::Baker_impl(
    DB::Transaction* transaction,
    const BAKER::IBaker_code* baker_code,
    const char* pixel_type,
    bool is_uniform)
  : m_baker_module( false),
    m_transaction( transaction),
    m_baker_code( baker_code, mi::base::DUP_INTERFACE),
    m_pixel_type( pixel_type),
    m_is_uniform( is_uniform)
{
    m_transaction->pin();
}

Baker_impl::~Baker_impl()
{
    m_transaction->unpin();
}

const char* Baker_impl::get_pixel_type() const
{
    return m_pixel_type.c_str();
}

bool Baker_impl::is_uniform() const
{
    return m_is_uniform;
}

mi::Sint32 Baker_impl::bake_texture( mi::neuraylib::ICanvas* texture, mi::Uint32 samples) const
{
    if( !texture)
        return -1;
    if( !m_transaction->is_open())
        return -2;

    if( m_baker_module->bake_texture( m_transaction, m_baker_code.get(), texture, samples) != 0)
        return -3;
    return 0;
}

mi::Sint32 Baker_impl::bake_constant( mi::IData* constant, mi::Uint32 samples) const
{
    if( !constant)
        return -1;
    if( !m_transaction->is_open())
        return -2;

    if (m_pixel_type == "Boolean") {
        // boolean results cannot be accumulated
        samples = 1;
    }

    BAKER::Baker_module::Constant_result data;
    if( m_baker_module->bake_constant(
            m_transaction, m_baker_code.get(), data, samples, m_pixel_type.c_str()) != 0)
        return -3;

    if( m_pixel_type == "Rgb_fp") {

        mi::base::Handle<mi::IColor> constant_color(
            constant->get_interface<mi::IColor>());
        if( !constant_color)
            return -4;

        constant_color->set_value(0, 0, data.v.x);
        constant_color->set_value(1, 0, data.v.y);
        constant_color->set_value(2, 0, data.v.z);

    } else if( m_pixel_type == "Float32<3>") {

        mi::base::Handle<mi::IFloat32_3> constant_float32_3(
            constant->get_interface<mi::IFloat32_3>());
        if( !constant_float32_3)
            return -4;

        constant_float32_3->set_value( 0, 0, data.v.x);
        constant_float32_3->set_value( 1, 0, data.v.y);
        constant_float32_3->set_value( 2, 0, data.v.z);

    } else if( m_pixel_type == "Float32") {

        mi::base::Handle<mi::IFloat32> constant_float32(
            constant->get_interface<mi::IFloat32>());
        if( !constant_float32)
            return -4;

        constant_float32->set_value( data.f);

    } else if (m_pixel_type == "Boolean") {

        mi::base::Handle<mi::IBoolean> constant_boolean(
            constant->get_interface<mi::IBoolean>());
        if (!constant_boolean)
            return -4;

        constant_boolean->set_value(data.b);

    } else

        ASSERT( M_NEURAY_API, false);

    return 0;
}

} // namespace NEURAY

} // namespace MI
