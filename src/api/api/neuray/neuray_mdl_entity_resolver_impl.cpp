/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IMdl_entity_resolver implementation.
 **/

#include "pch.h"

#include "neuray_mdl_entity_resolver_impl.h"
#include "neuray_mdl_execution_context_impl.h"

#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_thread_context.h>
#include <mi/neuraylib/ireader.h>

#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <mdl/compiler/compilercore/compilercore_positions.h>

namespace MI {

namespace NEURAY {

mi::neuraylib::Uvtile_mode convert_mdl_udim_mode_to_api_uvtile_mode( mi::mdl::UDIM_mode mode)
{
    switch( mode) {
        case mi::mdl::NO_UDIM:   return mi::neuraylib::UVTILE_MODE_NONE;
        case mi::mdl::UM_MARI:   return mi::neuraylib::UVTILE_MODE_UDIM;
        case mi::mdl::UM_ZBRUSH: return mi::neuraylib::UVTILE_MODE_UVTILE0;
        case mi::mdl::UM_MUDBOX: return mi::neuraylib::UVTILE_MODE_UVTILE1;
    }

    ASSERT( M_NEURAY_API, false);
    return mi::neuraylib::UVTILE_MODE_NONE;
}

mi::mdl::UDIM_mode convert_mdl_udim_mode_to_api_uvtile_mode( mi::neuraylib::Uvtile_mode mode)
{
    switch( mode) {
        case mi::neuraylib::UVTILE_MODE_NONE:          return mi::mdl::NO_UDIM;
        case mi::neuraylib::UVTILE_MODE_UDIM:          return mi::mdl::UM_MARI;
        case mi::neuraylib::UVTILE_MODE_UVTILE0:       return mi::mdl::UM_ZBRUSH;
        case mi::neuraylib::UVTILE_MODE_UVTILE1:       return mi::mdl::UM_MUDBOX;
        case  mi::neuraylib::UVTILE_MODE_FORCE_32_BIT: ASSERT( M_NEURAY_API, false);
                                                       return mi::mdl::NO_UDIM;
    }

    ASSERT( M_NEURAY_API, false);
    return mi::mdl::NO_UDIM;
}

Mdl_entity_resolver_impl::Mdl_entity_resolver_impl(
    mi::mdl::IMDL* mdl, mi::mdl::IEntity_resolver* resolver)
  : m_mdl( mdl, mi::base::DUP_INTERFACE),
    m_resolver( resolver, mi::base::DUP_INTERFACE)
{
}

mi::neuraylib::IMdl_resolved_module* Mdl_entity_resolver_impl::resolve_module(
    const char* module_name,
    const char* owner_file_path,
    const char* owner_name,
    mi::Sint32 pos_line,
    mi::Sint32 pos_column,
    mi::neuraylib::IMdl_execution_context* context)
{
    std::unique_ptr<mi::mdl::Position_impl> position(
        new mi::mdl::Position_impl( pos_line, pos_column, pos_line, pos_column));

    mi::base::Handle<mi::mdl::IMDL_import_result> result( m_resolver->resolve_module(
        module_name, owner_file_path, owner_name, position.get()));

    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);
    const mi::mdl::Messages& messages = m_resolver->access_messages();
    MDL::convert_messages( messages, mdl_context);

    return result ? new Mdl_resolved_module_impl( m_mdl.get(), result.get()) : nullptr;
}

mi::neuraylib::IMdl_resolved_resource* Mdl_entity_resolver_impl::resolve_resource(
    const char* file_path,
    const char* owner_file_path,
    const char* owner_name,
    mi::Sint32 pos_line,
    mi::Sint32 pos_column,
    mi::neuraylib::IMdl_execution_context* context)
{
    std::unique_ptr<mi::mdl::Position_impl> position(
        new mi::mdl::Position_impl( pos_line, pos_column, pos_line, pos_column));

    mi::base::Handle<mi::mdl::IMDL_resource_set> result( m_resolver->resolve_resource_file_name(
        file_path, owner_file_path, owner_name, position.get()));


    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);
    const mi::mdl::Messages& messages = m_resolver->access_messages();
    MDL::convert_messages( messages, mdl_context);

    return result ? new Mdl_resolved_resource_impl( result.get()) : nullptr;
}

Mdl_resolved_module_impl::Mdl_resolved_module_impl(
    mi::mdl::IMDL* mdl, mi::mdl::IMDL_import_result* import_result)
  : m_mdl( mdl, mi::base::DUP_INTERFACE),
    m_import_result( import_result, mi::base::DUP_INTERFACE)
{
}

const char* Mdl_resolved_module_impl::get_module_name() const
{
    const char* s = m_import_result->get_absolute_name();
    return s && s[0] ? s : nullptr;
}

const char* Mdl_resolved_module_impl::get_filename() const
{
    const char* s = m_import_result->get_file_name();
    return s && s[0] ? s : nullptr;
}

mi::neuraylib::IReader* Mdl_resolved_module_impl::create_reader() const
{
    mi::base::Handle<mi::mdl::IThread_context> thread_context( m_mdl->create_thread_context());
    mi::base::Handle<mi::mdl::IInput_stream> stream( m_import_result->open( thread_context.get()));
    return stream ? MDL::get_reader( stream.get()) : nullptr;
}

Mdl_resolved_resource_impl::Mdl_resolved_resource_impl( mi::mdl::IMDL_resource_set* resource_set)
  : m_resource_set( resource_set, mi::base::DUP_INTERFACE)
{
}

mi::neuraylib::Uvtile_mode Mdl_resolved_resource_impl::get_uvtile_mode() const
{
    return convert_mdl_udim_mode_to_api_uvtile_mode( m_resource_set->get_udim_mode());
}

const char* Mdl_resolved_resource_impl::get_mdl_file_path_mask() const
{
    const char* s = m_resource_set->get_mdl_url_mask();
    return s && s[0] ? s : nullptr;
}

const char* Mdl_resolved_resource_impl::get_filename_mask() const
{
    const char* s = m_resource_set->get_filename_mask();
    return s && s[0] ? s : nullptr;
}

mi::Size Mdl_resolved_resource_impl::get_count() const
{
    return m_resource_set->get_count();
}

const char* Mdl_resolved_resource_impl::get_mdl_file_path( mi::Size i) const
{
    const char* s = m_resource_set->get_mdl_url( i);
    return s && s[0] ? s : nullptr;
}

const char* Mdl_resolved_resource_impl::get_filename( mi::Size i) const
{
    const char* s = m_resource_set->get_filename( i);
    return s && s[0] ? s : nullptr;
}

mi::neuraylib::IReader* Mdl_resolved_resource_impl::create_reader( mi::Size i) const
{
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader( m_resource_set->open_reader( i));
    return reader? MDL::get_reader( reader.get()) : nullptr;
}

mi::base::Uuid Mdl_resolved_resource_impl::get_resource_hash( mi::Size i) const
{
    unsigned char hash[16];
    bool success = m_resource_set->get_resource_hash( i, hash);
    return success ? MDL::convert_hash( hash) : mi::base::Uuid{0,0,0,0};
}

bool Mdl_resolved_resource_impl::get_uvtile_uv( mi::Size i, mi::Sint32& u, mi::Sint32& v) const
{
    return m_resource_set->get_udim_mapping( i, u, v);
}

Core_entity_resolver_impl::Core_entity_resolver_impl(
    mi::mdl::IMDL* mdl, mi::neuraylib::IMdl_entity_resolver* resolver)
  : m_mdl( mdl, mi::base::DUP_INTERFACE),
    m_resolver( resolver, mi::base::DUP_INTERFACE),
    m_messages( m_mdl->get_mdl_allocator(), /*owner_fname*/ nullptr)
{
}

namespace { bool is_non_null( const char* s) { return s && s[0]; } }

mi::mdl::IMDL_import_result* Core_entity_resolver_impl::resolve_module(
    const char* module_name,
    const char* owner_file_path,
    const char* owner_name,
    const mi::mdl::Position* pos)
{
    mi::Sint32 pos_line   = pos ? pos->get_start_line()   : 0;
    mi::Sint32 pos_column = pos ? pos->get_start_column() : 0;

    Mdl_execution_context_impl context;
    mi::base::Handle<mi::neuraylib::IMdl_resolved_module> result( m_resolver->resolve_module(
        module_name, owner_file_path, owner_name, pos_line, pos_column, &context));

    // Sanity check against incorrect implementations of mi::neuraylib::IMdl_resolved_module:
    // Module names have to be valid.
    if( result) {
        bool ok = true;
        ok &= is_non_null( result->get_module_name());
        if( !ok) {
            LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_MISC,
                "Rejecting incorrectly resolved module with invalid module name");
            return nullptr;
        }
    }

    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_context( &context, default_context);
    m_messages.clear();
    convert_messages( mdl_context, m_messages);

    return result ? new Core_mdl_import_result_impl( result.get()) : nullptr;
}

mi::mdl::IMDL_resource_set* Core_entity_resolver_impl::resolve_resource_file_name(
    const char* file_path,
    const char* owner_file_path,
    const char* owner_name,
    const mi::mdl::Position* pos)
{
    mi::Sint32 pos_line   = pos ? pos->get_start_line()   : 0;
    mi::Sint32 pos_column = pos ? pos->get_start_column() : 0;

    Mdl_execution_context_impl context;
    mi::base::Handle<mi::neuraylib::IMdl_resolved_resource> result( m_resolver->resolve_resource(
        file_path, owner_file_path, owner_name, pos_line, pos_column, &context));

    // Sanity check against incorrect implementations of mi::neuraylib::IMdl_resolved_resource:
    // The file path mask and file paths have to be valid.
    if( result) {
        bool ok = true;
        ok &= is_non_null( result->get_mdl_file_path_mask());
        for( mi::Size i = 0, n = result->get_count(); i < n; ++i)
            ok &= is_non_null( result->get_mdl_file_path( i));
        if( !ok) {
            LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_MISC,
                "Rejecting incorrectly resolved resource with invalid MDL file path mask and/or MDL "
                "file path");
            return nullptr;
        }
    }

    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_context( &context, default_context);
    m_messages.clear();
    convert_messages( mdl_context, m_messages);

    return result ? new Core_mdl_resource_set_impl( result.get()) : nullptr;
}

const mi::mdl::Messages& Core_entity_resolver_impl::access_messages() const
{
    return m_messages;
}

Core_mdl_import_result_impl::Core_mdl_import_result_impl(
    mi::neuraylib::IMdl_resolved_module* resolved_module)
  : m_resolved_module( resolved_module, mi::base::DUP_INTERFACE)
{
}

const char* Core_mdl_import_result_impl::get_absolute_name() const
{
    return m_resolved_module->get_module_name();
}

const char* Core_mdl_import_result_impl::get_file_name() const
{
    return m_resolved_module->get_filename();
}

mi::mdl::IInput_stream* Core_mdl_import_result_impl::open( mi::mdl::IThread_context* context) const
{
    mi::base::Handle<mi::neuraylib::IReader> reader( m_resolved_module->create_reader());
    if( !reader)
        return nullptr;


    const char* filename = m_resolved_module->get_filename();
    std::string filename_str = filename ? filename : "";

    if( MDL::is_mdle( filename_str)) {
        // Mimic the behavior of the builtin implementation of IMDL_import_result::open().
        filename_str += ":main.mdl";
        return MDL::get_mdle_input_stream( reader.get(), filename_str);
    } else
        return MDL::get_input_stream( reader.get(), filename_str);
}

Core_mdl_resource_set_impl::Core_mdl_resource_set_impl(
    mi::neuraylib::IMdl_resolved_resource* resolved_resource)
  : m_resolved_resource( resolved_resource, mi::base::DUP_INTERFACE)
{
}

const char* Core_mdl_resource_set_impl::get_mdl_url_mask() const
{
    return m_resolved_resource->get_mdl_file_path_mask();
}

const char* Core_mdl_resource_set_impl::get_filename_mask() const
{
    return m_resolved_resource->get_filename_mask();
}

size_t Core_mdl_resource_set_impl::get_count() const
{
    return m_resolved_resource->get_count();
}

const char* Core_mdl_resource_set_impl::get_mdl_url( size_t i) const
{
    return m_resolved_resource->get_mdl_file_path( i);
}

const char* Core_mdl_resource_set_impl::get_filename( size_t i) const
{
    return m_resolved_resource->get_filename( i);
}

bool Core_mdl_resource_set_impl::get_udim_mapping( size_t i, int &u, int &v) const
{
    return m_resolved_resource->get_uvtile_uv( i, u, v);
}

mi::mdl::IMDL_resource_reader* Core_mdl_resource_set_impl::open_reader( size_t i) const
{
    mi::base::Handle<mi::neuraylib::IReader> reader( m_resolved_resource->create_reader( i));
    if( !reader || !reader->supports_absolute_access())
        return nullptr;

    const char* file_path = m_resolved_resource->get_mdl_file_path( i);
    const char* filename  = m_resolved_resource->get_filename( i);
    mi::base::Uuid hash   = m_resolved_resource->get_resource_hash( i);
    return MDL::get_resource_reader(
        reader.get(), file_path ? file_path : "", filename ? filename : "", hash);
}

mi::mdl::UDIM_mode Core_mdl_resource_set_impl::get_udim_mode() const
{
    return convert_mdl_udim_mode_to_api_uvtile_mode( m_resolved_resource->get_uvtile_mode());
}

bool Core_mdl_resource_set_impl::get_resource_hash( size_t i, unsigned char hash[16]) const
{
    return MDL::convert_hash( m_resolved_resource->get_resource_hash( i), hash);
}

} // namespace NEURAY

} // namespace MI
