/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the Impexp_utilities implementation.
 **/

#include "pch.h"

#include "neuray_class_factory.h"
#include "neuray_impexp_utilities.h"
#include "neuray_recording_transaction.h"
#include "neuray_transaction_impl.h"
#include "neuray_uri.h"
#include "neuray_string_impl.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/iexport_result.h>
#include <mi/neuraylib/iimport_result.h>
#include <mi/neuraylib/iimpexp_base.h>
#include <mi/neuraylib/iimpexp_state.h>
#include <mi/neuraylib/istring.h>

#include <base/hal/hal/hal.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/system/main/access_module.h>
#include <base/lib/path/i_path.h>

#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>


#include <regex>

namespace MI {

namespace NEURAY {

std::string Impexp_utilities::s_shader = "${shader}";

mi::neuraylib::IImport_result* Impexp_utilities::create_import_result(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 message_number,
    mi::base::Message_severity message_severity,
    const char* message,
    DB::Tag rootgroup,
    DB::Tag camera_inst,
    DB::Tag options,
    const std::vector<std::string>& elements)
{
    ASSERT( M_NEURAY_API, transaction);
    ASSERT( M_NEURAY_API, message || message_number == 0);

    mi::neuraylib::IImport_result_ext* import_result_ext
        = transaction->create<mi::neuraylib::IImport_result_ext>( "Import_result_ext");

    // set message number, severity, and message
    if( message_number != 0)
        import_result_ext->message_push_back( message_number, message_severity, message);

    // set names of rootgroup, camera instance, and options
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    if( rootgroup.is_valid())
        import_result_ext->set_rootgroup( db_transaction->tag_to_name( rootgroup));
    if( camera_inst.is_valid())
        import_result_ext->set_camera_inst( db_transaction->tag_to_name( camera_inst));
    if( options.is_valid())
        import_result_ext->set_options( db_transaction->tag_to_name( options));

    // set names of imported elements
    std::vector<std::string>::const_iterator it = elements.begin();
    for( ; it != elements.end(); ++it)
        import_result_ext->element_push_back( it->c_str());

    return import_result_ext;
}

mi::neuraylib::IExport_result* Impexp_utilities::create_export_result(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 message_number,
    mi::base::Message_severity message_severity,
    const char* message)
{
    ASSERT( M_NEURAY_API, transaction);
    ASSERT( M_NEURAY_API, message || message_number == 0);

    mi::neuraylib::IExport_result_ext* export_result_ext
        = transaction->create<mi::neuraylib::IExport_result_ext>( "Export_result_ext");

    // set message number, severity, and message
    if( message_number != 0)
        export_result_ext->message_push_back( message_number, message_severity, message);

    return export_result_ext;
}

mi::neuraylib::IExport_result* Impexp_utilities::create_export_result(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 message_number,
    mi::base::Message_severity message_severity,
    const char* message,
    const char* argument)
{
    char buffer[1024];
    snprintf( &buffer[0], sizeof( buffer)-1, message, argument);
    return create_export_result( transaction, message_number, message_severity, &buffer[0]);
}

namespace {

/// Wraps a memory block identified by a pointer and a length as mi::neuraylib::IBuffer.
/// Does not copy the data.
class Buffer_wrapper
  : public mi::base::Interface_implement<mi::neuraylib::IBuffer>,
    public boost::noncopyable
{
public:
    Buffer_wrapper( const mi::Uint8* data, mi::Size data_size)
      : m_data( data), m_data_size( data_size) { }
    const mi::Uint8* get_data() const { return m_data; }
    mi::Size get_data_size() const { return m_data_size; }
private:
    const mi::Uint8* m_data;
    const mi::Size m_data_size;
};

} // anonymous namespace

mi::neuraylib::IReader* Impexp_utilities::create_reader( const char* data, mi::Size length)
{
    const mi::Uint8* d = reinterpret_cast<const mi::Uint8*>( data);
    mi::base::Handle<mi::neuraylib::IBuffer> buffer( new Buffer_wrapper( d, length));
    return new DISK::Memory_reader_impl( buffer.get());
}

mi::neuraylib::IReader* Impexp_utilities::create_reader(
    const std::string& uri, std::string& path)
{
    path = convert_uri_to_filename( uri);
    if( path.empty())
        return nullptr;

    DISK::File_reader_impl* file_reader_impl = new DISK::File_reader_impl();

    // handle ${shader} path
    if( is_shader_path( path)) {

        SYSTEM::Access_module<PATH::Path_module> path_module( false);
        const std::vector<std::string>& shader_paths
            = path_module->get_search_path( PATH::MDL);

        for( std::vector<std::string>::const_iterator it = shader_paths.begin();
            it != shader_paths.end(); ++it) {
            std::string test_path = resolve_shader_path( path, *it);
            if( file_reader_impl->open( test_path.c_str()))
                return file_reader_impl;
        }

        file_reader_impl->release();
        return nullptr;
    }

    // handle non-${shader} paths
    if( !file_reader_impl->open( path.c_str())) {
        file_reader_impl->release();
        return nullptr;
    }

    return file_reader_impl;
}

mi::neuraylib::IWriter* Impexp_utilities::create_writer(
    const std::string& uri, std::string& path)
{
    path = convert_uri_to_filename( uri);
    if( path.empty())
        return nullptr;

    DISK::File_writer_impl* file_writer_impl = new DISK::File_writer_impl();
    if( !file_writer_impl->open( path.c_str())) {
        file_writer_impl->release();
        return nullptr;
    }

    return file_writer_impl;
}

mi::neuraylib::IImport_result_ext* Impexp_utilities::create_import_result_ext(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IImport_result* import_result)
{
    mi::neuraylib::IImport_result_ext* import_result_ext
        = import_result->get_interface<mi::neuraylib::IImport_result_ext>();
    if( import_result_ext)
        return import_result_ext;

    import_result_ext
        = transaction->create<mi::neuraylib::IImport_result_ext>( "Import_result_ext");

    // set message numbers, severities, and messages
    import_result_ext->append_messages( import_result);

    // set names of rootgroup, camera instance, and options
    import_result_ext->set_rootgroup( import_result->get_rootgroup());
    import_result_ext->set_camera_inst( import_result->get_camera_inst());
    import_result_ext->set_options( import_result->get_options());

    // set names of imported elements
    import_result_ext->append_elements( import_result);

    return import_result_ext;
}

std::vector<std::string> Impexp_utilities::get_recorded_elements(
    const Transaction_impl* transaction, Recording_transaction* recording_transaction)
{
    const Class_factory* class_factory = transaction->get_class_factory();
    const std::vector<DB::Tag>& tags = recording_transaction->get_stored_tags();

    std::vector<std::string> names;
    for( std::vector<DB::Tag>::const_iterator it = tags.begin(); it != tags.end(); ++it) {
        const char* name = recording_transaction->tag_to_name( *it);
        if( name)
            names.push_back( name);

        // Skip DB elements that have not been registered with the API's class factory.
        SERIAL::Class_id class_id = class_factory->get_class_id( transaction, *it);
        if( !class_factory->is_class_registered( class_id))
            continue;

        // Implementation classes of resources should have been skipped. They are an internal
        // implementation detail and not visible in the API.
        ASSERT( M_NEURAY_API, class_id != DBIMAGE::ID_IMAGE_IMPL);
        ASSERT( M_NEURAY_API, class_id != LIGHTPROFILE::ID_LIGHTPROFILE_IMPL);
        ASSERT( M_NEURAY_API, class_id != BSDFM::ID_BSDF_MEASUREMENT_IMPL);
    }

    return names;
}

std::string Impexp_utilities::get_extension( const std::string& uri)
{
    std::string uri_string = HAL::Ospath::convert_to_forward_slashes( uri);

    std::string::size_type last_dot = uri_string.rfind( '.');
    if( last_dot == std::string::npos)
        return "";

    std::string::size_type last_slash = uri_string.rfind( '/');
    if( last_slash == std::string::npos)
        last_slash = 0;

    if( last_slash > last_dot)
        return "";

    return uri_string.substr( last_dot);
}

bool Impexp_utilities::is_shader_path( const std::string& path)
{
    if( path.find( s_shader) != 0)
        return false;
    if( path.size() == s_shader.size())
        return true;
    std::string norm_path = HAL::Ospath::convert_to_forward_slashes( path);
    if( norm_path[s_shader.size()] == '/')
        return true;
    return false;
}

std::string Impexp_utilities::resolve_shader_path(
    const std::string& path, const std::string& shader_path)
{
    if( !is_shader_path( path))
        return path;
    if( path.size() == s_shader.size())
        return "";
    return HAL::Ospath::normpath_v2( HAL::Ospath::join_v2(
        shader_path, path.substr( s_shader.size()+1)));
}

std::string Impexp_utilities::resolve_uri_against_state(
    const char* child, const mi::neuraylib::IImpexp_state* parent_state)
{
    ASSERT( M_NEURAY_API, child);

    if( !parent_state)
        return child;

    Uri child_uri( child);
    if( child_uri.is_absolute())
        return child;

    const char* parent = parent_state->get_uri();
    Uri parent_uri( parent ? parent : ".");
    if( !parent_uri.is_absolute())
        return child;

    Uri resolved_uri( parent, child);
    return resolved_uri.get_str();
}

std::string Impexp_utilities::convert_filename_to_uri( const std::string& filename)
{
#ifndef MI_PLATFORM_WINDOWS
    // absolute paths, case (3)
    if(( filename.length() >= 2) && ( filename[0] == '/') && ( filename[1] == '/'))
        return "//" + filename;

    // nothing to do
    return filename;
#else
    // use / to separate potential URI items and URI path components
    std::string uri =  HAL::Ospath::convert_to_forward_slashes( filename);

    // absolute paths, case (1)
    if(( uri.length() >= 2) && is_drive_letter( uri[0]) && ( uri[1] == ':'))
        return "/" + uri;

    // absolute paths, case (3)
    if(( uri.length() >= 2) && ( uri[0] == '/') && ( uri[1] == '/'))
        return "//" + uri;

    // relative paths and absolute paths, case (2)
    return uri;
#endif
}

std::string Impexp_utilities::convert_uri_to_filename( const std::string& uri)
{
    Uri uri_class( uri.c_str());

    const std::string& scheme = uri_class.get_scheme();
    if( !scheme.empty() && scheme != "file")
        return "";

    const std::string& authority = uri_class.get_authority();
    if( !authority.empty())
        return "";

    std::string path = uri_class.get_path();
    if( path.empty())
        return "";

#ifndef MI_PLATFORM_WINDOWS
    // nothing to do
    return path;
#else
    // use native path separator
    path = HAL::Ospath::convert_to_backward_slashes( path);

    // relative paths
    if( path[0] != '\\')
        return path;

    // absolute paths, case (1)
    if(( path.length() >= 3) && is_drive_letter( path[1]) && ( path[2] == ':')
        && (( path.length() == 3) || ( path[3] == '\\')))
        return path.substr( 1);

    // absolute paths, case (2)
    if(( path.length() >= 2) && is_drive_letter( path[1])
       && (( path.length() == 2) || ( path[2] == '\\'))) {
        path[0] = path[1];
        path[1] = ':';
        return path;
    }

    // absolute paths, cases (3) and (4)
    return path;
#endif
}

std::vector<DB::Tag> Impexp_utilities::convert_names_to_tags(
    mi::neuraylib::ITransaction* transaction, const mi::IArray* names)
{
    ASSERT( M_NEURAY_API, transaction);
    ASSERT( M_NEURAY_API, names);

    std::vector<DB::Tag> tags;
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    for( mi::Uint32 i = 0; i < names->get_length(); ++i) {

        mi::base::Handle<const mi::IString> name( names->get_element<mi::IString>( i));
        if( !name.is_valid_interface())
            return std::vector<DB::Tag>();
        const char* name_c_str = name->get_c_str();
        DB::Tag tag = db_transaction->name_to_tag( name_c_str);
        if( !tag)
            return std::vector<DB::Tag>();
        tags.push_back( tag);
    }

    return tags;
}

mi::IArray* Impexp_utilities::convert_tags_to_names(
    mi::neuraylib::ITransaction* transaction, const std::vector<DB::Tag>& tags)
{
    ASSERT( M_NEURAY_API, transaction);

    mi::IDynamic_array* names = transaction->create<mi::IDynamic_array>( "String[]");
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    for( mi::Size i = 0; i < tags.size(); ++i) {
        const char* name = db_transaction->tag_to_name( tags[i]);
        if( !name)
            continue;
        mi::base::Handle<mi::IString> string( transaction->create<mi::IString>( "String"));
        string->set_c_str( name);
        names->push_back( string.get());
    }

    return names;
}

bool Impexp_utilities::Impexp_less::operator()(
    const mi::neuraylib::IImpexp_base* lhs, const mi::neuraylib::IImpexp_base* rhs)
{
    if( lhs->get_priority() < rhs->get_priority())
        return true;
    if( lhs->get_priority() > rhs->get_priority())
        return false;

    if(( strcmp( lhs->get_author(), rhs->get_author()) == 0)
      && strcmp( lhs->get_name(), rhs->get_name()) == 0) {

        if( lhs->get_major_version() < rhs->get_major_version())
            return true;
        if( lhs->get_major_version() > rhs->get_major_version())
            return false;
        if( lhs->get_minor_version() < rhs->get_minor_version())
            return true;
        return false;
    }

    return false;
}

std::vector<DB::Tag> Impexp_utilities::get_export_elements(
    DB::Transaction* db_transaction,
    const std::vector<DB::Tag>& tags,
    bool recurse,
    DB::Tag_version* time_stamp,
    bool shortcuts_mdl)
{
    std::vector<DB::Tag> result;

    std::set<DB::Tag> tags_seen;
    for( std::vector<DB::Tag>::const_iterator it = tags.begin(); it != tags.end(); ++it)
        if( tags_seen.find( *it) == tags_seen.end()) {
            tags_seen.insert( *it);
            get_export_elements_internal(
                db_transaction, *it, recurse, time_stamp, shortcuts_mdl, result, tags_seen);
        }

    return result;
}

void Impexp_utilities::get_export_elements_internal(
    DB::Transaction* db_transaction,
    DB::Tag tag,
    bool recurse,
    DB::Tag_version* time_stamp,
    bool shortcuts_mdl,
    std::vector<DB::Tag>& result,
    std::set<DB::Tag>& tags_seen)
{
    SERIAL::Class_id class_id = db_transaction->get_class_id( tag);

    // recurse into references if requested (optimization: skip for MDL modules)
    if( recurse && (!shortcuts_mdl || class_id != MDL::ID_MDL_MODULE)) {

        // get references of tag (optimization: consider only MDL module for MDL definitions)
        DB::Tag_set references;
        if( shortcuts_mdl && class_id == MDL::ID_MDL_MATERIAL_DEFINITION) {
            DB::Access<MDL::Mdl_material_definition> element( tag, db_transaction);
            references.insert( element->get_module(db_transaction));
        } else if( shortcuts_mdl && class_id == MDL::ID_MDL_FUNCTION_DEFINITION) {
            DB::Access<MDL::Mdl_function_definition> element( tag, db_transaction);
            references.insert( element->get_module(db_transaction));
        } else {
            DB::Access<DB::Element_base> element( tag, db_transaction);
            element->get_references( &references);
        }

        // call recursively for all references not yet in tags_seen
        for( DB::Tag_set::const_iterator it = references.begin(); it != references.end(); ++it)
            if( tags_seen.find( *it) == tags_seen.end()) {
                tags_seen.insert( *it);
                get_export_elements_internal(
                    db_transaction, *it, recurse, time_stamp, shortcuts_mdl, result, tags_seen);
            }
    }

    // optimization: do not report tags of MDL definitions
    if( shortcuts_mdl && class_id == MDL::ID_MDL_MATERIAL_DEFINITION)
        return;
    if( shortcuts_mdl && class_id == MDL::ID_MDL_FUNCTION_DEFINITION)
        return;

    // report tag if no time stamp is given
    if( !time_stamp) {
        result.push_back( tag);
        return;
    }

    // or report tag if changed since time stamp
    DB::Tag_version tag_version = db_transaction->get_tag_version( tag);
    if(    (time_stamp->m_transaction_id < tag_version.m_transaction_id)
        || (    (time_stamp->m_transaction_id == tag_version.m_transaction_id)
             && (time_stamp->m_version < tag_version.m_version)))
        result.push_back( tag);
}

bool Impexp_utilities::is_drive_letter( char c)
{
    return ((c >= 'A') && (c <= 'Z')) || ((c >= 'a') && (c <= 'z'));
}

} // namespace NEURAY

} // namespace MI

