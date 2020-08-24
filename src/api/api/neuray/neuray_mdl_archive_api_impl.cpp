/***************************************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IMdl_archive_api implementation.
 **/

#include "pch.h"

#include "neuray_mdl_archive_api_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/istring.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_streams.h>

#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/shared_ptr.hpp>
#include <base/system/main/i_module_id.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_assert.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

namespace MI {

namespace NEURAY {

Mdl_archive_api_impl::Mdl_archive_api_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray)
{
}

Mdl_archive_api_impl::~Mdl_archive_api_impl()
{
    m_neuray = nullptr;
}

mi::Sint32 Mdl_archive_api_impl::create_archive(
    const char* directory, const char* archive, const mi::IArray* manifest_fields)
{
    if( !directory || !archive)
        return -1;

    std::string archive_str( archive);
    if( archive_str.size() < 5 || archive_str.substr( archive_str.size()-4) != ".mdr")
        return -2;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    if( mdl->uses_external_entity_resolver())
        return -5;

    // split archive into output_directory and package_name
    std::string output_directory;
    std::string package_name;
    size_t sep = archive_str.rfind( HAL::Ospath::sep());
    if( sep == std::string::npos) {
        output_directory = ".";
        package_name += archive_str;
    } else {
        output_directory = archive_str.substr( 0, sep);
        package_name += archive_str.substr( sep+1);
    }
    package_name = package_name.substr( 0, package_name.size()-4);
    boost::replace_all( package_name, ".", "::");

    // convert manifest_fields into manifest_entries
    mi::Size manifest_entries_count = manifest_fields ? manifest_fields->get_length() : 0;
    typedef boost::shared_ptr<mi::mdl::IArchive_tool::Key_value_entry[]>
        Manifest_entries_shared_ptr;
    Manifest_entries_shared_ptr manifest_entries;

    if( manifest_entries_count != 0) {

        manifest_entries = Manifest_entries_shared_ptr(
            new mi::mdl::IArchive_tool::Key_value_entry[manifest_entries_count]);

        for( mi::Size i = 0; i < manifest_entries_count; ++i) {

            mi::base::Handle<const mi::IStructure> field(
                manifest_fields->get_value<mi::IStructure>( i));
            if( !field)
                return -3;

            mi::base::Handle<const mi::IString> key( field->get_value<mi::IString>( "key"));
            if( !key)
                return -3;

            mi::base::Handle<const mi::IString> value( field->get_value<mi::IString>( "value"));
            if( !value)
                return -3;

            manifest_entries[i].key   = key->get_c_str();
            manifest_entries[i].value = value->get_c_str();
        }
    }

    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool( mdl->create_archive_tool());
    mi::mdl::Options& options = archive_tool->access_options();
    options.set_option( MDL_ARC_OPTION_OVERWRITE, "true");
    options.set_option( MDL_ARC_OPTION_IGNORE_EXTRA_FILES, "true");
    options.set_option( MDL_ARC_OPTION_COMPRESS_SUFFIXES, m_compression_extensions.c_str());
    mi::base::Handle<const mi::mdl::IArchive> the_archive(
        archive_tool->create_archive(
            directory,
            package_name.c_str(),
            output_directory.c_str(),
            manifest_entries_count > 0 ? manifest_entries.get() : nullptr,
            manifest_entries_count));

    const mi::mdl::Messages& messages = archive_tool->access_messages();
    MDL::report_messages( messages, /*out_messages*/ nullptr);
    if( messages.get_error_message_count() > 0)
        return -4;

    return 0;
}

mi::Sint32 Mdl_archive_api_impl::extract_archive( const char* archive, const char* directory)
{
    if( !archive || !directory)
        return -1;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool( mdl->create_archive_tool());
    mi::mdl::Options& options = archive_tool->access_options();
    options.set_option( MDL_ARC_OPTION_OVERWRITE, "true");
    options.set_option( MDL_ARC_OPTION_IGNORE_EXTRA_FILES, "true");
    archive_tool->extract_archive( archive, directory);

    const mi::mdl::Messages& messages = archive_tool->access_messages();
    MDL::report_messages( messages, /*out_messages*/ nullptr);
    if( messages.get_error_message_count() > 0)
        return -2;

    return 0;
}

const mi::neuraylib::IManifest* Mdl_archive_api_impl::get_manifest( const char* archive)
{
    if( !archive)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool( mdl->create_archive_tool());
    mi::base::Handle<const mi::mdl::IArchive_manifest> manifest(
        archive_tool->get_manifest( archive));

    const mi::mdl::Messages& messages = archive_tool->access_messages();
    MDL::report_messages( messages, /*out_messages*/ nullptr);
    if( !manifest || messages.get_error_message_count() > 0)
        return nullptr;

    return new Manifest_impl( manifest.get());
}

mi::neuraylib::IReader* Mdl_archive_api_impl::get_file( const char* archive, const char* filename)
{
    if( !archive || !filename)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool( mdl->create_archive_tool());
    mi::base::Handle<mi::mdl::IInput_stream> file(
        archive_tool->get_file_content( archive, filename));

    const mi::mdl::Messages& messages = archive_tool->access_messages();
    MDL::report_messages( messages, /*out_messages*/ nullptr);
    if( !file || messages.get_error_message_count() > 0)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL_resource_reader> file_random_access(
        file->get_interface<mi::mdl::IMDL_resource_reader>());
    ASSERT( M_NEURAY_API, file_random_access.get());
    return MDL::get_reader( file_random_access.get());
}

mi::neuraylib::IReader* Mdl_archive_api_impl::get_file(const char* filename)
{
    if (!filename)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());

    std::string fn(filename);
    auto p = fn.find(".mdr:");
    if (p == std::string::npos || p == fn.size() - 5)
        return nullptr;

    mi::base::Handle<mi::mdl::IInput_stream> file(
        archive_tool->get_file_content(fn.substr(0, p + 4).c_str(), fn.substr(p + 5).c_str()));

    const mi::mdl::Messages& messages = archive_tool->access_messages();
    MDL::report_messages(messages, /*out_messages*/ nullptr);
    if (!file || messages.get_error_message_count() > 0)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL_resource_reader> file_random_access(
        file->get_interface<mi::mdl::IMDL_resource_reader>());
    ASSERT(M_NEURAY_API, file_random_access.get());
    return MDL::get_reader(file_random_access.get());
}


mi::Sint32 Mdl_archive_api_impl::set_extensions_for_compression( const char* extensions)
{
    if( !extensions)
        return -1;

    m_compression_extensions = extensions;
    return 0;
}

const char* Mdl_archive_api_impl::get_extensions_for_compression() const
{
    return m_compression_extensions.c_str();
}

mi::Sint32 Mdl_archive_api_impl::start()
{
    m_mdlc_module.set();

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool( mdl->create_archive_tool());
    mi::mdl::Options& options = archive_tool->access_options();
    int index = options.get_option_index( MDL_ARC_OPTION_COMPRESS_SUFFIXES);
    const char* s = options.get_option_value( index);
    m_compression_extensions = s ? s : "";

    return 0;
}

mi::Sint32 Mdl_archive_api_impl::shutdown()
{
    m_mdlc_module.reset();
    return 0;
}

Manifest_impl::Manifest_impl( const mi::mdl::IArchive_manifest* manifest)
{
    mi::Size first_index = 0;
    const char* key = nullptr;
    const char* value = nullptr;
    const mi::mdl::IArchive_manifest_value* multi_value = nullptr;

    // mdl
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_MDL);
    mi::mdl::IMDL::MDL_version mdl = manifest->get_mdl_version();
    value = convert_mdl_version( mdl);
    m_fields.push_back( std::make_pair( key, value));
    m_index_count[key] = std::make_pair( m_fields.size()-1, 1);

    // version
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_VERSION);
    const mi::mdl::ISemantic_version* version = manifest->get_sema_version();
    std::string version_str = convert_sema_version( version);
    m_fields.push_back( std::make_pair( key, version_str));
    m_index_count[key] = std::make_pair( m_fields.size()-1, 1);

    // module
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_MODULE);
    mi::Size count = manifest->get_module_count();
    for( mi::Size i = 0; i < count; ++i) {
        // note: libMDL returns the module names WITHOUT the leasding ::
        value = manifest->get_module_name( i);
        m_fields.push_back( std::make_pair( key, std::string("::") + value));
    }
    m_index_count[key] = std::make_pair( m_fields.size()-count, count);

    // dependency
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_DEPENDENCY);
    first_index = m_fields.size();
    const mi::mdl::IArchive_manifest_dependency* dependency = manifest->get_first_dependency();
    while( dependency) {
        std::string dependency_str( dependency->get_archive_name());
        dependency_str += ' ';
        dependency_str += convert_sema_version( dependency->get_version());
        m_fields.push_back( std::make_pair( key, dependency_str));
        dependency = dependency->get_next();
    }
    m_index_count[key] = std::make_pair( first_index, m_fields.size()-first_index);

    // exports.*
    convert_exports( manifest, mi::mdl::IArchive_manifest::PK_EX_FUNCTION);
    convert_exports( manifest, mi::mdl::IArchive_manifest::PK_EX_MATERIAL);
    convert_exports( manifest, mi::mdl::IArchive_manifest::PK_EX_STRUCT);
    convert_exports( manifest, mi::mdl::IArchive_manifest::PK_EX_ENUM);
    convert_exports( manifest, mi::mdl::IArchive_manifest::PK_EX_CONST);
    convert_exports( manifest, mi::mdl::IArchive_manifest::PK_EX_ANNOTATION);

    // author
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_AUTHOR);
    first_index = m_fields.size();
    multi_value = manifest->get_opt_author();
    while( multi_value) {
        value = multi_value->get_value();
        m_fields.push_back( std::make_pair( key, value));
        multi_value = multi_value->get_next();
    }
    m_index_count[key] = std::make_pair( first_index, m_fields.size()-first_index);

    // contributor
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_CONTRIBUTOR);
    first_index = m_fields.size();
    multi_value = manifest->get_opt_contributor();
    while( multi_value) {
        value = multi_value->get_value();
        m_fields.push_back( std::make_pair( key, value));
        multi_value = multi_value->get_next();
    }
    m_index_count[key] = std::make_pair( first_index, m_fields.size()-first_index);

    // copyright_notice
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_COPYRIGHT_NOTICE);
    value = manifest->get_opt_copyrigth_notice();
    if( value) {
        m_fields.push_back( std::make_pair( key, value));
        m_index_count[key] = std::make_pair( m_fields.size()-1, 1);
    }

    // description
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_DESCRIPTION);
    value = manifest->get_opt_description();
    if( value) {
        m_fields.push_back( std::make_pair( key, value));
        m_index_count[key] = std::make_pair( m_fields.size()-1, 1);
    }

    // created
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_CREATED);
    value = manifest->get_opt_created();
    if( value) {
        m_fields.push_back( std::make_pair( key, value));
        m_index_count[key] = std::make_pair( m_fields.size()-1, 1);
    }

    // modified
    key = manifest->get_key( mi::mdl::IArchive_manifest::PK_MODIFIED);
    value = manifest->get_opt_modified();
    if( value) {
        m_fields.push_back( std::make_pair( key, value));
        m_index_count[key] = std::make_pair( m_fields.size()-1, 1);
    }

    // user-defined fields
    count = manifest->get_key_count();
    for( mi::Size i = mi::mdl::IArchive_manifest::PK_FIRST_USER_ID; i < count; ++i) {
        key = manifest->get_key( i);
        first_index = m_fields.size();
        multi_value = manifest->get_first_value( i);
        while( multi_value) {
            value = multi_value->get_value();
            m_fields.push_back( std::make_pair( key, value));
            multi_value = multi_value->get_next();
        }
        m_index_count[key] = std::make_pair( first_index, m_fields.size()-first_index);
    }
}

mi::Size Manifest_impl::get_number_of_fields() const
{
    return m_fields.size();
}

const char* Manifest_impl::get_key( mi::Size index) const
{
    return index < m_fields.size() ? m_fields[index].first.c_str() : nullptr;
}

const char* Manifest_impl::get_value( mi::Size index) const
{
    return index < m_fields.size() ? m_fields[index].second.c_str() : nullptr;
}

mi::Size Manifest_impl::get_number_of_fields( const char* key) const
{
    if( !key)
        return 0;

    Index_count_map::const_iterator it = m_index_count.find( key);
    return it == m_index_count.end() ? 0 : it->second.second;
}

const char* Manifest_impl::get_value( const char* key, mi::Size index) const
{
    if( !key)
        return nullptr;

    Index_count_map::const_iterator it = m_index_count.find( key);
    if( it == m_index_count.end())
        return nullptr;
    if( index >= it->second.second)
        return nullptr;

    return m_fields[it->second.first + index].second.c_str();
}

void Manifest_impl::convert_exports(
    const mi::mdl::IArchive_manifest* manifest, mi::mdl::IArchive_manifest::Predefined_key pkey)
{
    const char* key = manifest->get_key( pkey);

    mi::mdl::IArchive_manifest::Export_kind export_kind;
    switch( pkey) {
        case mi::mdl::IArchive_manifest::PK_EX_FUNCTION:
            export_kind =  mi::mdl::IArchive_manifest::EK_FUNCTION;   break;
        case mi::mdl::IArchive_manifest::PK_EX_MATERIAL:
            export_kind =  mi::mdl::IArchive_manifest::EK_MATERIAL;   break;
        case mi::mdl::IArchive_manifest::PK_EX_STRUCT:
            export_kind =  mi::mdl::IArchive_manifest::EK_STRUCT;     break;
        case mi::mdl::IArchive_manifest::PK_EX_ENUM:
            export_kind =  mi::mdl::IArchive_manifest::EK_ENUM;       break;
        case mi::mdl::IArchive_manifest::PK_EX_CONST:
            export_kind =  mi::mdl::IArchive_manifest::EK_CONST;      break;
        case mi::mdl::IArchive_manifest::PK_EX_ANNOTATION:
            export_kind =  mi::mdl::IArchive_manifest::EK_ANNOTATION; break;
        default:
            ASSERT( M_NEURAY_API, false);
            return;
    }

    mi::Size first_index = m_fields.size();
    for( mi::Size i = 0, n = manifest->get_module_count(); i < n; ++i) {
        const mi::mdl::IArchive_manifest_export* multi_value
            = manifest->get_first_export( i, export_kind);
        const char* module_name = manifest->get_module_name( i);

        while( multi_value) {
            const char* export_name = multi_value->get_export_name();
            std::string qualified_name = std::string( "::") + module_name + "::" + export_name;
            m_fields.push_back( std::make_pair( key, qualified_name));
            multi_value = multi_value->get_next();
        }
    }

    m_index_count[key] = std::make_pair( first_index, m_fields.size()-first_index);
}

const char* Manifest_impl::convert_mdl_version( mi::mdl::IMDL::MDL_version version)
{
    switch( version) {
        case mi::mdl::IMDL::MDL_VERSION_1_0: return "1.0";
        case mi::mdl::IMDL::MDL_VERSION_1_1: return "1.1";
        case mi::mdl::IMDL::MDL_VERSION_1_2: return "1.2";
        case mi::mdl::IMDL::MDL_VERSION_1_3: return "1.3";
        case mi::mdl::IMDL::MDL_VERSION_1_4: return "1.4";
        case mi::mdl::IMDL::MDL_VERSION_1_5: return "1.5";
        case mi::mdl::IMDL::MDL_VERSION_1_6: return "1.6";
        case mi::mdl::IMDL::MDL_VERSION_1_7: return "1.7";
    }

    ASSERT( M_NEURAY_API, false);
    return "unknown";
}

std::string Manifest_impl::convert_sema_version( const mi::mdl::ISemantic_version* version)
{
    std::ostringstream result;

    result << version->get_major() << '.';
    result << version->get_minor() << '.';
    result << version->get_patch();
    const char* pre_release = version->get_prerelease();
    if (pre_release && pre_release[0] != '\0') {
        result << '-' << pre_release;
    }

    return result.str();
}

} // namespace NEURAY

} // namespace MI
