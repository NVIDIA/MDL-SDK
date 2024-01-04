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

/** \file
 ** \brief
 **/


#include "pch.h"

#if defined(__GNUC__) && (__GNUC__ <= 7)
// GCC 7 does not support <filesystem>
int main() { return 0; }
#else

// Enable this to install an external entity resolver for most tests.
//
#define EXTERNAL_ENTITY_RESOLVER

// Enable this define to change the external entity resolver such that it returns resources without
// filenames.
//
// #define RESOLVE_RESOURCES_WITHOUT_FILENAMES

// Enable this define to skip/adapt tests if the context option "resolve_resources" defaults to false.
// Note: this define does \em not change the value of the context option itself.
//
// #define RESOLVE_RESOURCES_FALSE

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/config.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include <mi/neuraylib/annotation_wrapper.h>
#include <mi/neuraylib/argument_editor.h>
#include <mi/neuraylib/definition_wrapper.h>
#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/ibsdf_measurement.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/ilightprofile.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/imdl_archive_api.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_distiller_api.h>
#include <mi/neuraylib/imdl_entity_resolver.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdl_module_builder.h>
#include <mi/neuraylib/imdl_module_transformer.h>
#include <mi/neuraylib/imdle_api.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/ispectrum.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/itransaction.h>

#include <mi/neuraylib/definition_wrapper.h>
#include <mi/neuraylib/imdl_compiler.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <boost/algorithm/string/replace.hpp>

#include "test_shared.h"



// There used to be several variants of this unit test. To avoid race conditions, each variant used a
// separate subdirectory for all files it creates (possibly with further subdirectories).
#define DIR_PREFIX "output_test_imdl_module"

// True name of the ::test_mdl module. This allows to test various problematic names without
// changing every other line of the entire test.
#define TEST_MDL      "test_mdl%24"
#define TEST_MDL_FILE "test_mdl$"

#define TEST_NO_1_RUS u8"\u0422\u0435\u0441\u0442\u041d\u043e\u043c\u0435\u0440\u041e\u0434\u0438\u043d\u0451"



namespace fs = std::filesystem;

mi::Sint32 result = 0;

const char* native_module = "mdl 1.3; export color f(color c) [[ native() ]] { return c; }";

std::string mdle_path = MI::TEST::mi_src_path( "prod/lib/neuray") + "/test.mdle";
std::string mdle_module_db_name;

const std::string unicode_id_package     = u8"H\u00f6lzer";
const std::string unicode_id_eiche       = u8"Eiche";
const std::string unicode_id_foehre      = u8"F\u00f6hre";
const std::string unicode_id_buche       = u8"Buche";
const std::string unicode_id_mdl_package = u8"mdl";
const std::string unicode_id_keyword     = u8"keyword";

char const* keyword_src = u8"mdl 1.8;\nexport material mat() = material();\n\n";
char const* eiche_src   = u8"mdl 1.8;\nexport material eichen_material() = material();\n";
char const* foehre_src  = u8"mdl 1.8;\nexport material kiefern_material() = material();\n";
char const* buche_src   = u8"mdl 1.8;\n\
import ::'H\u00f6lzer'::Eiche::eichen_material;\n\
import 'F\u00f6hre'::kiefern_material;\n\
import 'mdl'::keyword::mat;\n\
export material eiche() = ::'H\\U000000f6lzer'::'\\x45iche'::eichen_material();\n\
export material kiefer() = 'F\\u00f6hre'::kiefern_material();\n\
export material buche() = material();\n\
export material mat() = 'mdl'::keyword::mat();\n";

bool ends_with( const char* s, const char* suffix)
{
    if( !s)
        return false;
    size_t n_s = strlen( s);
    size_t n_suffix = strlen( suffix);
    if( n_s < n_suffix)
        return false;
    return strcmp( s + n_s-n_suffix, suffix) == 0;
}

bool compare_files( const std::string& filename1, const std::string& filename2)
{
    std::ifstream file1( filename1);
    std::ifstream file2( filename2);

    if( !file1 || !file2)
        return false;

    std::string s1, s2;
    while( !file1.eof() && !file2.eof()) {
        std::getline( file1, s1);
        std::getline( file2, s2);
        if( s1 != s2)
            return false;
    }

    return ! (file1.eof() ^ file2.eof());
}

std::vector<std::pair<std::string,std::string>> encoding {
    { "%",  "%25" }, // The simple encoder needs this pair first.
    { "(",  "%28" },
    { ")",  "%29" },
    { "<",  "%3C" },
    { ">",  "%3E" },
    { ",",  "%2C" },
    { ":",  "%3A" },
    { "#",  "%23" },
    { "?",  "%3F" },
    { "@",  "%40" },
    { "$",  "%24" }
};

// Poor man's encoding function.
std::string encode( std::string s)
{
    for( const auto& e: encoding)
        boost::replace_all( s, e.first, e.second);
    return s;
}

// Poor man's decoding function.
std::string decode( std::string s)
{
    for( const auto& e: encoding)
        boost::replace_all( s, e.second, e.first);
    return s;
}

/// Converts the initializer list of string literals into a dynamic array of IString elements.
mi::IArray* create_istring_array( mi::neuraylib::IFactory* factory, std::initializer_list<const char*> l)
{
    mi::IDynamic_array* result = factory->create<mi::IDynamic_array>( "String[]");
    mi::Size n = 0;

    for( auto it = l.begin(); it != l.end(); ++it) {
        mi::base::Handle<mi::IString> s( factory->create<mi::IString>());
        s->set_c_str( *it);
        result->set_length( ++n);
        result->set_element( n-1, s.get());
    }

    return result;
}

// Wrapper for mi::neuraylib::IMdl_resolved_module.
//
// Forwards all calls. Used by My_resolver.
class My_module : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_module>
{
public:
    My_module( mi::neuraylib::IMdl_resolved_module* impl) : m_impl( impl, mi::base::DUP_INTERFACE) { }
    const char* get_module_name() const { return m_impl->get_module_name(); }
    const char* get_filename() const { return m_impl->get_filename(); }
    mi::neuraylib::IReader* create_reader() const { return m_impl->create_reader(); }

private:
    mi::base::Handle<IMdl_resolved_module> m_impl;
};

// Wrapper for mi::neuraylib::IMdl_resolved_resource.
//
// Forwards all calls unless RESOLVE_RESOURCES_WITHOUT_FILENAMES is set. Used by My_resolver.
class My_resource_element : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource_element>
{
public:
    My_resource_element(const mi::neuraylib::IMdl_resolved_resource_element *impl) : m_impl(impl, mi::base::DUP_INTERFACE) { }
    mi::Size get_frame_number() const final { return m_impl->get_frame_number(); }
    mi::Size get_count() const final { return m_impl->get_count(); }
    const char* get_mdl_file_path( mi::Size i) const final { return m_impl->get_mdl_file_path( i); }
#ifdef RESOLVE_RESOURCES_WITHOUT_FILENAMES
    const char* get_filename( mi::Size i) const final { return nullptr; }
#else
    const char* get_filename( mi::Size i) const final { return m_impl->get_filename( i); }
#endif
    mi::neuraylib::IReader* create_reader( mi::Size i) const final { return m_impl->create_reader( i); }
    mi::base::Uuid get_resource_hash( mi::Size i) const final { return m_impl->get_resource_hash( i); }
    bool get_uvtile_uv( mi::Size i, mi::Sint32 & u, mi::Sint32 & v) const { return m_impl->get_uvtile_uv( i, u, v); }

private:
    mi::base::Handle<const mi::neuraylib::IMdl_resolved_resource_element> m_impl;
};

// Wrapper for mi::neuraylib::IMdl_resolved_resource.
//
// Forwards all calls unless RESOLVE_RESOURCES_WITHOUT_FILENAMES is set. Used by My_resolver.
class My_resource : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource>
{
public:
    My_resource( mi::neuraylib::IMdl_resolved_resource* impl) : m_impl( impl, mi::base::DUP_INTERFACE) { }
    mi::Size get_count() const final { return m_impl->get_count(); }
    bool has_sequence_marker() const final { return m_impl->has_sequence_marker(); }
    mi::neuraylib::Uvtile_mode get_uvtile_mode() const final { return m_impl->get_uvtile_mode(); }
    const char* get_mdl_file_path_mask() const final { return m_impl->get_mdl_file_path_mask(); }
#ifdef RESOLVE_RESOURCES_WITHOUT_FILENAMES
    const char* get_filename_mask() const final { return nullptr; }
#else
    const char* get_filename_mask() const final { return m_impl->get_filename_mask(); }
#endif
    const mi::neuraylib::IMdl_resolved_resource_element* get_element( mi::Size i) const final
    {
        mi::base::Handle<const mi::neuraylib::IMdl_resolved_resource_element> elem( m_impl->get_element( i));
        return elem ? new My_resource_element( elem.get()) : nullptr;
    }

private:
    mi::base::Handle<mi::neuraylib::IMdl_resolved_resource> m_impl;
};

#define SAFE_STR(x) ((x)?(x):"null")

// define a custom IInterface to test context user data
class IMdl_execution_context_user_data : public
    mi::base::Interface_declare<0x1f0d182a,0xf013,0x4b40,0xa3,0xd3,0xfb,0xd5,0x4c,0xe2,0x76,0x20>
{
public:
    // some user defined methods
    virtual mi::Size get_number() const = 0;
};

class Mdl_execution_context_user_data : public
    mi::base::Interface_implement<IMdl_execution_context_user_data>
{
public:
    mi::Size get_number() const final { return 42; }
};

// Wrapper for mi::neuraylib::IMdl_entity_resolver.
//
// Forwards all calls and wraps the results.
class My_resolver : public mi::base::Interface_implement<mi::neuraylib::IMdl_entity_resolver>
{
public:
    My_resolver( mi::neuraylib::IMdl_entity_resolver* impl)
      : m_impl( impl, mi::base::DUP_INTERFACE),
        m_enable_user_data_check( false) { }

    void enable_user_data_check( bool value) { m_enable_user_data_check = value; }

    void check_user_data( mi::neuraylib::IMdl_execution_context* context)
    {
        if( !m_enable_user_data_check)
            return;

        mi::base::Handle<const IMdl_execution_context_user_data> user_data(
            context->get_option<IMdl_execution_context_user_data>( "user_data", result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( user_data);

        mi::Size number = user_data->get_number();
        MI_CHECK_EQUAL( number, 42);
    }

    mi::neuraylib::IMdl_resolved_module* resolve_module(
        const char* module_name,
        const char* owner_file_path,
        const char* owner_name,
        mi::Sint32 pos_line,
        mi::Sint32 pos_column,
        mi::neuraylib::IMdl_execution_context* context)
    {
        check_user_data( context);

        // std::cerr << "module " << module_name << " " << SAFE_STR( owner_file_path) << " " << SAFE_STR( owner_name) << std::endl;
        mi::base::Handle<mi::neuraylib::IMdl_resolved_module> result( m_impl->resolve_module(
            module_name, owner_file_path, owner_name, pos_line, pos_column, context));
        if( !result)
            return nullptr;
        return new My_module( result.get());
    }

    mi::neuraylib::IMdl_resolved_resource* resolve_resource(
        const char* file_path,
        const char* owner_file_path,
        const char* owner_name,
        mi::Sint32 pos_line,
        mi::Sint32 pos_column,
        mi::neuraylib::IMdl_execution_context* context)
    {
        check_user_data( context);

        // std::cerr << "resource " << file_path << " " << SAFE_STR( owner_file_path) << " " << SAFE_STR( owner_name) << std::endl;
        mi::base::Handle<mi::neuraylib::IMdl_resolved_resource> result( m_impl->resolve_resource(
            file_path, owner_file_path, owner_name, pos_line, pos_column, context));
        if( !result)
            return nullptr;
        return new My_resource( result.get());
    }

private:
    mi::base::Handle<IMdl_entity_resolver> m_impl;
    bool m_enable_user_data_check;
};

void install_external_resolver(
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    bool enable_user_data_check = false)
{
#ifdef EXTERNAL_ENTITY_RESOLVER
    // For testing purposes we use the internal resolver wrapped twice and install it as external
    // resolver.
    //
    // Note that such an resolver needs to be uninstalled and re-installed on search path changes to
    // take effect. MDR creation fails with an external resolver.
    //
    // A limitation of that test setup is that the inner (original builtin) resolver does not have
    // access to the module cache which is only known to the outer (wrapped) resolver. Therefore,
    // imports of modules that only exist in the database will fail, e.g., string-based modules,
    // or modules created by the module builder.

    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> resolver(
        mdl_configuration->get_entity_resolver());
    // Wrap the resolver returned by the API, such that we can easily intercept all calls (and
    // e.g. modify them if RESOLVE_RESOURCES_WITHOUT_FILENAMES is defined).
    My_resolver* my_resolver = new My_resolver( resolver.get());
    my_resolver->enable_user_data_check( enable_user_data_check);
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> modified_resolver( my_resolver);
    mdl_configuration->set_entity_resolver( modified_resolver.get());
#endif
}

void uninstall_external_resolver( mi::neuraylib::IMdl_configuration* mdl_configuration)
{
#ifdef EXTERNAL_ENTITY_RESOLVER
    mdl_configuration->set_entity_resolver( nullptr);
#endif
}

// Demo implementation of IMdle_serialization_callback swapping upper and lower-case characters
// (except for the file extension). Adds "FOO" prefix to test handling of string length changes.
class Mdle_serialization_callback
  : public mi::base::Interface_implement<mi::neuraylib::IMdle_serialization_callback>
{
public:
    Mdle_serialization_callback( mi::neuraylib::IFactory* factory)
      : m_factory( factory, mi::base::DUP_INTERFACE) { }

    const mi::IString* get_serialized_filename( const char* filename) const
    {
        std::string s = filename;
        // ".mdle" suffix => 5
        for( size_t i = 0, n = s.size(); i+5 < n; ++i) {
            if( islower( s[i]))
                s[i] = static_cast<char>( std::toupper( s[i]));
            else if( isupper( s[i]))
                s[i] = static_cast<char>( std::tolower( s[i]));
        }

        s = "FOO" + s;
        mi::IString* result = m_factory->create<mi::IString>();
        result->set_c_str( s.c_str());
        return result;
    }

private:
    mi::base::Handle<mi::neuraylib::IFactory> m_factory;
};

// Demo implementation of IMdle_deserialization_callback swapping upper and lower-case characters
// (except for the file extension). Removes "FOO" prefix to test handling of string length changes.
// Replaces "/" by "/./" and "\" by "\.\ to test normalization of the callback result.
class Mdle_deserialization_callback
  : public mi::base::Interface_implement<mi::neuraylib::IMdle_deserialization_callback>
{
public:
    Mdle_deserialization_callback( mi::neuraylib::IFactory* factory)
      : m_factory( factory, mi::base::DUP_INTERFACE) { }

    const mi::IString* get_deserialized_filename( const char* filename) const
    {
        std::string s = filename;
        if( s.substr( 0, 3) != "FOO")
            return nullptr;

        s = s.substr( 3);
        // ".mdle" suffix => 5
        for( size_t i = 0, n = s.size(); i+5 < n; ++i) {
            if( islower( s[i]))
                s[i] = static_cast<char>( std::toupper( s[i]));
            else if( isupper( s[i]))
                s[i] = static_cast<char>( std::tolower( s[i]));
        }

        boost::replace_all( s, "/", "/./");
        boost::replace_all( s, "\"", "\\.\"");

        mi::IString* result = m_factory->create<mi::IString>();
        result->set_c_str( s.c_str());
        return result;
    }

private:
    mi::base::Handle<mi::neuraylib::IFactory> m_factory;
};

void create_unicode_module()
{
    std::string dirname = std::string( DIR_PREFIX) + "/" + TEST_NO_1_RUS;
    std::string filename = dirname + "/1_module.mdl";

    const char* src =
        "mdl 1.6;\n"
        "import::state::*;\n"
        "import::df::*;\n"
        "export color fd_test(float a = 1.0) {\n"
        "    return color(state::texture_coordinate(0)) * color(a);\n"
        "}\n"
        "\n"
        "export material md_test(float a = 1.0) = material(\n"
        "    surface: material_surface(\n"
        "        scattering : df::diffuse_reflection_bsdf(tint : color(a)))\n"
        ");\n";

    fs::remove_all( fs::u8path( dirname));
    if( fs::create_directory( fs::u8path( dirname))) {
        std::ofstream( fs::u8path( filename)) << src << std::endl;
    }
}

void create_unicode_id_package()
{
    std::string dirname = std::string( DIR_PREFIX) + "/" + unicode_id_package;
    std::string mdl_dirname = std::string( DIR_PREFIX) + "/" + unicode_id_package + "/" + unicode_id_mdl_package;

    std::string eiche_filename = dirname + "/" + unicode_id_eiche + ".mdl";
    std::string buche_filename = dirname + "/" + unicode_id_buche + ".mdl";
    std::string foehre_filename = dirname + "/" + unicode_id_foehre + ".mdl";
    std::string keyword_filename = mdl_dirname + "/" + unicode_id_keyword + ".mdl";

    fs::remove_all( fs::u8path( dirname));
    if( fs::create_directory( fs::u8path( dirname))) {
        std::ofstream( fs::u8path( eiche_filename)) << eiche_src << std::endl;
        std::ofstream( fs::u8path( foehre_filename)) << foehre_src << std::endl;
        std::ofstream( fs::u8path( buche_filename)) << buche_src << std::endl;
    }

    fs::remove_all( fs::u8path( mdl_dirname));
    if( fs::create_directory( fs::u8path( mdl_dirname))) {
        std::ofstream( fs::u8path( keyword_filename)) << keyword_src << std::endl;
    }
}

void check_mdl_factory( mi::neuraylib::IMdl_factory* mdl_factory)
{
    MI_CHECK( mdl_factory->is_valid_mdl_identifier( "foo"));
    MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "_foo"));   // starts with underscore
    MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "42foo"));  // starts with digit
    MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "module")); // keyword
    MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "new"));    // reserved for future use
}

void check_decoding(
    mi::neuraylib::IMdl_factory* mdl_factory, const char* input, const char* expected_result)
{
    mi::base::Handle<const mi::IString> s1( mdl_factory->decode_name( input));
    MI_CHECK_EQUAL_CSTR( s1->get_c_str(), expected_result);
}

void check_encoding_module(
    mi::neuraylib::IMdl_factory* mdl_factory, const char* input, const char* expected_result)
{
    mi::base::Handle<const mi::IString> s( mdl_factory->encode_module_name( input));
    MI_CHECK_EQUAL_CSTR( s->get_c_str(), expected_result);
}

void check_encoding_type(
    mi::neuraylib::IMdl_factory* mdl_factory, const char* input, const char* expected_result)
{
    mi::base::Handle<const mi::IString> s( mdl_factory->encode_type_name( input));
    MI_CHECK_EQUAL_CSTR( s->get_c_str(), expected_result);
}

void check_encoding_function(
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const char* name,
    std::initializer_list<const char*> parameter_types,
    const char* expected_result)
{
    mi::base::Handle<const mi::IArray> array( create_istring_array( factory, parameter_types));
    mi::base::Handle<const mi::IString> s( mdl_factory->encode_function_definition_name( name, array.get()));
    MI_CHECK_EQUAL_CSTR( s->get_c_str(), expected_result);
}

void check_encoding_decoding( mi::neuraylib::IFactory* factory, mi::neuraylib::IMdl_factory* mdl_factory)
{
    check_decoding( mdl_factory, "%28%29%3C%3E%2C%3A%24%25", "()<>,:$%");
    check_decoding( mdl_factory, "()<>,:$%",                 "()<>,:$%");

    check_encoding_module( mdl_factory, "::foo_()<>,:$%", "::foo_%28%29%3C%3E%2C%3A%24%25");

    check_encoding_function( factory, mdl_factory, "::foo::bar",   { "int",  "float" }, "::foo::bar(int,float)" );
    check_encoding_function( factory, mdl_factory, "::foo$::bar$", { "int$", "float" }, "::foo%24::bar$(int$,float)" );

    check_encoding_module( mdl_factory, "int",         "int");
    check_encoding_module( mdl_factory, "::foo$::bar", "::foo%24::bar");
}


// import the module in the given transaction
void import_mdl_module(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const char* module_name,
    mi::Sint32 expected_result)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Pass a custom interface as user data to the context in order to receive it in callbacks.
    mi::base::Handle<IMdl_execution_context_user_data> user_data(
        new Mdl_execution_context_user_data());
    context->set_option( "user_data", user_data.get());

    result = mdl_impexp_api->load_module( transaction, module_name, context.get());
    if( expected_result >= 0) {
        MI_CHECK_CTX( context.get());
    }
    MI_CHECK_EQUAL( result, expected_result);
}

// check import of modules (limited test since there is only one global scope)
void check_mdl_import(
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
    install_external_resolver( mdl_configuration, /*enable_user_data_check*/ true);

    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::lib::neuray::" TEST_MDL, 0);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::neuray::" TEST_MDL, 0);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::" TEST_MDL, 0);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::test_mdl2", 0);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::test_mdl3", 0);
    std::string path = MI::TEST::mi_src_path( "prod/lib/neuray/") + TEST_MDL_FILE ".mdl";
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, path.c_str(), -1);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::base", 1);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::state", 1);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::test_archives", 0);
    import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::test_compatibility", 0);

    {
        //  check some properties of ::lib::neuray::test_mdl<.mdl
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::lib::neuray::" TEST_MDL));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL_CSTR( "::lib::neuray::" TEST_MDL, c_module->get_mdl_name());
        MI_CHECK_EQUAL_CSTR( "" TEST_MDL, c_module->get_mdl_simple_name());
        MI_CHECK_EQUAL( 2, c_module->get_mdl_package_component_count());
        MI_CHECK_EQUAL_CSTR( "lib", c_module->get_mdl_package_component_name( 0));
        MI_CHECK_EQUAL_CSTR( "neuray", c_module->get_mdl_package_component_name( 1));
    }
    {
        //  check some properties of ::neuray::test_mdl<.mdl
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::neuray::" TEST_MDL));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL_CSTR( "::neuray::" TEST_MDL, c_module->get_mdl_name());
        MI_CHECK_EQUAL_CSTR( "" TEST_MDL, c_module->get_mdl_simple_name());
        MI_CHECK_EQUAL( 1, c_module->get_mdl_package_component_count());
        MI_CHECK_EQUAL_CSTR( "neuray", c_module->get_mdl_package_component_name( 0));
    }
    {
        // test.mdle
        import_mdl_module( transaction, mdl_impexp_api, mdl_factory, mdle_path.c_str(), 0);
        mi::base::Handle<const mi::IString> s( mdl_factory->get_db_module_name( mdle_path.c_str()));
        MI_CHECK( s);
        mdle_module_db_name = s->get_c_str();
    }
    {
        // prepare for mdl reload test
        fs::copy_file(
            MI::TEST::mi_src_path( "prod/lib/neuray") + "/test_mdl_reload_orig.mdl",
            DIR_PREFIX "/test_mdl_reload.mdl");
        import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::test_mdl_reload", 0);
        import_mdl_module( transaction, mdl_impexp_api, mdl_factory, "::test_mdl_reload_import", 0);

        const char* module_source = "mdl 1.0; export material some_material() = material();";
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdl_impexp_api->load_module_from_string(
            transaction, "::test_mdl_reload_from_string", module_source, context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 0);
    }
    {
        // prepare module with unicode file name
        create_unicode_module();
        std::string unicode_name = "::" TEST_NO_1_RUS "::1_module";
        import_mdl_module( transaction, mdl_impexp_api, mdl_factory, unicode_name.c_str(), 0);
    }

    // import_mdl_module() used above sets up the context for the user data check, but the ad-hoc
    // context used below (and possibly later in other places) is not prepared for that.
    uninstall_external_resolver( mdl_configuration);
    install_external_resolver( mdl_configuration, /*enable_user_data_check*/ false);

    {
        // prepare package with unicode package/module names
        create_unicode_id_package();
        std::string unicode_id_name = std::string( "::") + unicode_id_package + "::" + unicode_id_buche;
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdl_impexp_api->load_module( transaction, unicode_id_name.c_str(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 0);
    }
    {
        // check IMdl_impexp_api::get_mdl_module_name()
        mi::base::Handle<const mi::IString> mdl_module_name;
        std::string expected_mdl_module_name;

        mdl_module_name = mdl_impexp_api->get_mdl_module_name( 0);
        MI_CHECK( !mdl_module_name);

        mdl_module_name = mdl_impexp_api->get_mdl_module_name( "");
        MI_CHECK( !mdl_module_name);

        mdl_module_name = mdl_impexp_api->get_mdl_module_name( mdle_path.c_str());
        MI_CHECK( !mdl_module_name);

        std::string mdl_path = MI::TEST::mi_src_path( "prod/lib/neuray/") + TEST_MDL_FILE ".mdl";

        expected_mdl_module_name = "::neuray::" TEST_MDL;
        mdl_module_name = mdl_impexp_api->get_mdl_module_name( mdl_path.c_str());
        MI_CHECK( mdl_module_name);
        MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

        expected_mdl_module_name = "::neuray::" TEST_MDL;
        mdl_module_name = mdl_impexp_api->get_mdl_module_name(
            mdl_path.c_str(), mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_FIRST);
        MI_CHECK( mdl_module_name);
        MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

        expected_mdl_module_name = "::lib::neuray::" TEST_MDL;
        mdl_module_name = mdl_impexp_api->get_mdl_module_name(
            mdl_path.c_str(), mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_SHORTEST);
        MI_CHECK( mdl_module_name);
        MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

        expected_mdl_module_name = "::" TEST_MDL;
        mdl_module_name = mdl_impexp_api->get_mdl_module_name(
            mdl_path.c_str(), mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_LONGEST);
        MI_CHECK( mdl_module_name);
        MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());
    }
}


void do_create_function_call(
    mi::neuraylib::ITransaction* transaction,
    const char* definition_name,
    const char* call_name)
{
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>( definition_name));
    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( 0, &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 0, transaction->store( fc.get(), call_name));
}

void check_itransaction_methods( mi::neuraylib::ITransaction* transaction)
{
    // modules and definitions

    // check that ITransaction::access() works
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
    MI_CHECK( c_module);
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_0()"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_0()"));
    MI_CHECK( c_md);

    // check that ITransaction::edit() works
    mi::base::Handle<mi::neuraylib::IModule> m_module(
        transaction->edit<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
    MI_CHECK( m_module);
    // check that ITransaction::edit() is prohibited
    mi::base::Handle<mi::neuraylib::IFunction_definition> m_fd(
        transaction->edit<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_0()"));
    MI_CHECK_EQUAL( m_fd, 0);
    mi::base::Handle<mi::neuraylib::IFunction_definition> m_md(
        transaction->edit<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_0()"));
    MI_CHECK_EQUAL( m_md, 0);

    // check that ITransaction::copy() is prohibited
    MI_CHECK_EQUAL( -6, transaction->copy( "mdl::" TEST_MDL, "foo"));
    MI_CHECK_EQUAL( -6, transaction->copy( "mdl::" TEST_MDL "::fd_0()", "foo"));
    MI_CHECK_EQUAL( -6, transaction->copy( "mdl::" TEST_MDL "::md_0()", "foo"));

    {
        // ... also on defaults
        mi::base::Handle<const mi::neuraylib::IFunction_definition> m_md_default_call(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_default_call(material)"));

        mi::base::Handle<const mi::neuraylib::IExpression_list> md_defaults(
            m_md_default_call->get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression_call> md_default_call(
            md_defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
        const char* call_name = md_default_call->get_call();
        MI_CHECK_EQUAL( -6, transaction->copy( call_name, "foo"));

        mi::base::Handle<mi::neuraylib::IImage> dummy(
            transaction->create<mi::neuraylib::IImage>("Image"));
        transaction->store( dummy.get(), "dummy_image");
        MI_CHECK_EQUAL( -9, transaction->copy( "dummy_image", call_name));
    }

    // function calls and material instances

    do_create_function_call( transaction, "mdl::" TEST_MDL "::fd_1(int)", "mdl::" TEST_MDL "::fc_1");
    do_create_function_call(
        transaction, "mdl::" TEST_MDL "::md_1(color)", "mdl::" TEST_MDL "::mi_1");

    // check that ITransaction::access() works
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1"));
    MI_CHECK( c_fc);
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_1"));
    MI_CHECK( c_mi);

    // check that ITransaction::edit() works
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1"));
    MI_CHECK( m_fc);
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_1"));
    MI_CHECK( m_mi);

    // check that ITransaction::copy() works
    MI_CHECK_EQUAL( 0, transaction->copy( "mdl::" TEST_MDL "::fc_1", "mdl::" TEST_MDL "::fc_1_copy"));
    MI_CHECK_EQUAL( 0, transaction->copy( "mdl::" TEST_MDL "::mi_1", "mdl::" TEST_MDL "::mi_1_copy"));
}

void create_function_calls_and_material_instances(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // instantiate various functions without arguments or using the defaults
    const char* functions[] = {
        "mdl::" TEST_MDL "::fd_0()", "mdl::" TEST_MDL "::fc_0",
        "mdl::" TEST_MDL "::fd_1(int)", "mdl::" TEST_MDL "::fc_1",
        "mdl::" TEST_MDL "::fd_1_44(int)", "mdl::" TEST_MDL "::fc_1_44",
        "mdl::" TEST_MDL "::fd_deferred(int[N])",
            "mdl::" TEST_MDL "::fc_deferred_returns_int_array",
        "mdl::" TEST_MDL "::fd_deferred_struct(::" TEST_MDL "::foo_struct[N])",
            "mdl::" TEST_MDL "::fc_deferred_returns_struct_array",
        "mdl::" TEST_MDL "::fd_ret_bool()", "mdl::" TEST_MDL "::fc_ret_bool",
        "mdl::" TEST_MDL "::fd_ret_int()", "mdl::" TEST_MDL "::fc_ret_int",
        "mdl::" TEST_MDL "::fd_ret_float()", "mdl::" TEST_MDL "::fc_ret_float",
        "mdl::" TEST_MDL "::fd_ret_string()", "mdl::" TEST_MDL "::fc_ret_string",
        "mdl::" TEST_MDL "::fd_ret_enum()", "mdl::" TEST_MDL "::fc_ret_enum",
        "mdl::" TEST_MDL "::fd_ret_enum2()", "mdl::" TEST_MDL "::fc_ret_enum2",
        "mdl::" TEST_MDL "::fd_return_uniform()", "mdl::" TEST_MDL "::fc_return_uniform",
        "mdl::" TEST_MDL "::fd_return_auto()", "mdl::" TEST_MDL "::fc_return_auto",
        "mdl::" TEST_MDL "::fd_return_varying()", "mdl::" TEST_MDL "::fc_return_varying",
        "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying(int,int,int)",
            "mdl::" TEST_MDL "::fc_parameters_uniform_auto_varying",
        "mdl::" TEST_MDL "::fd_uniform()", "mdl::" TEST_MDL "::fc_uniform",
        "mdl::" TEST_MDL "::fd_jit(float)", "mdl::" TEST_MDL "::fc_jit",
        "mdl::" TEST_MDL "::normal()", "mdl::" TEST_MDL "::fc_normal",
        "mdl::" TEST_MDL "::fd_parameter_references(float,float,float)",
            "mdl::" TEST_MDL "::fc_parameter_references",
        "mdl::" TEST_NO_1_RUS "::1_module::fd_test(float)",
            "mdl::" TEST_NO_1_RUS "::1_module::fc_test"
    };

    for( mi::Size i = 0; i < sizeof( functions) / sizeof( const char*); i+=2)
        do_create_function_call( transaction, functions[i], functions[i+1]);

    // instantiate various materials without arguments or using the defaults
    const char* materials[] = {
        "mdl::" TEST_MDL "::md_0()", "mdl::" TEST_MDL "::mi_0",
        "mdl::" TEST_MDL "::md_1(color)", "mdl::" TEST_MDL "::mi_1",
        "mdl::" TEST_MDL "::md_tmp_ms(color)", "mdl::" TEST_MDL "::mi_tmp_ms",
        "mdl::" TEST_MDL "::md_tmp_bsdf(color)", "mdl::" TEST_MDL "::mi_tmp_bsdf",
        "mdl::" TEST_MDL "::md_float3_arg(float3)", "mdl::" TEST_MDL "::mi_float3_arg",
        "mdl::" TEST_MDL "::md_ternary_operator_argument(bool)",
            "mdl::" TEST_MDL "::mi_ternary_operator_argument",
        "mdl::" TEST_MDL "::md_ternary_operator_default(color)",
            "mdl::" TEST_MDL "::mi_ternary_operator_default",
        "mdl::" TEST_MDL "::md_ternary_operator_body(color)",
            "mdl::" TEST_MDL "::mi_ternary_operator_body",
        "mdl::" TEST_MDL "::md_resource_sharing(texture_2d,texture_2d)",
            "mdl::" TEST_MDL "::mi_resource_sharing",
        "mdl::" TEST_MDL "::md_folding()", "mdl::" TEST_MDL "::mi_folding",
        "mdl::" TEST_MDL "::md_folding2(bool,::" TEST_MDL "::Enum,int)",
            "mdl::" TEST_MDL "::mi_folding2",
        "mdl::" TEST_MDL "::md_folding_cutout_opacity(float,float)",
            "mdl::" TEST_MDL "::mi_folding_cutout_opacity",
        "mdl::" TEST_MDL "::md_folding_cutout_opacity2(material)",
            "mdl::" TEST_MDL "::mi_folding_cutout_opacity2",
        "mdl::" TEST_MDL "::md_folding_transparent_layers(float)",
            "mdl::" TEST_MDL "::mi_folding_transparent_layers",
        "mdl::" TEST_MDL "::md_folding_transparent_layers2(color,float,bool)",
            "mdl::" TEST_MDL "::mi_folding_transparent_layers2",
        "mdl::" TEST_MDL "::md_folding_transparent_layers3(float,float)",
            "mdl::" TEST_MDL "::mi_folding_transparent_layers3",
        "mdl::" TEST_MDL "::md_parameters_uniform_auto_varying(int,int,int)",
            "mdl::" TEST_MDL "::mi_parameters_uniform_auto_varying",
        "mdl::" TEST_MDL "::md_uses_non_exported_function()",
            "mdl::" TEST_MDL "::mi_uses_non_exported_function",
        "mdl::" TEST_MDL "::md_uses_non_exported_material()",
            "mdl::" TEST_MDL "::mi_uses_non_exported_material",
        "mdl::" TEST_MDL "::md_jit(texture_2d,string,light_profile)",
            "mdl::" TEST_MDL "::mi_jit",
        "mdl::" TEST_MDL "::md_texture(texture_2d)", "mdl::" TEST_MDL "::mi_texture",
        "mdl::" TEST_MDL "::md_light_profile(light_profile)",
            "mdl::" TEST_MDL "::mi_light_profile",
        "mdl::" TEST_MDL "::md_bsdf_measurement(bsdf_measurement)",
            "mdl::" TEST_MDL "::mi_bsdf_measurement",
        "mdl::" TEST_MDL "::md_baking()", "mdl::" TEST_MDL "::mi_baking",
        "mdl::" TEST_MDL "::md_class_baking(float,::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::mi_class_baking",
        "mdl::" TEST_NO_1_RUS "::1_module::md_test(float)",
            "mdl::" TEST_NO_1_RUS "::1_module::mi_test",
    };

    for( mi::Size i = 0; i < sizeof( materials) / sizeof( const char*); i+=2)
        do_create_function_call( transaction, materials[i], materials[i+1]);

    // test instantiation with parameters

    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_1(int)"));
    mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_1(color)"));
    mi::base::Handle<const mi::neuraylib::IFunction_call> fc;
    mi::base::Handle<const mi::neuraylib::IFunction_call> mi;

    // with empty argument list
    fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( fc);
    mi = md->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( mi);

    // with wrong argument name
    mi::base::Handle<mi::neuraylib::IValue> arg_int_value( vf->create_int());
    mi::base::Handle<mi::neuraylib::IExpression> arg_int(
        ef->create_constant( arg_int_value.get()));
    MI_CHECK_EQUAL( 0, args->add_expression( "wrong_argument_name", arg_int.get()));
    fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -1, result);
    MI_CHECK( !fc);
    mi = md->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -1, result);
    MI_CHECK( !mi);
    args = ef->create_expression_list();

    // with wrong argument type
    mi::base::Handle<mi::neuraylib::IValue> arg_float_value( vf->create_float());
    mi::base::Handle<mi::neuraylib::IExpression> arg_float(
        ef->create_constant( arg_float_value.get()));
    MI_CHECK_EQUAL( 0, args->add_expression( "param0", arg_float.get()));
    fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -2, result);
    MI_CHECK( !fc);
    args = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, args->add_expression( "tint", arg_float.get()));
    mi = md->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -2, result);
    MI_CHECK( !mi);
}

void check_imodule(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // check the "builtins" module

    mi::base::Handle<const mi::neuraylib::IModule> b_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::%3Cbuiltins%3E"));
    MI_CHECK( b_module);
    MI_CHECK_EQUAL_CSTR( "::%3Cbuiltins%3E", b_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "%3Cbuiltins%3E", b_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, b_module->get_mdl_package_component_count());
    MI_CHECK( b_module->is_standard_module());
    MI_CHECK( !b_module->is_mdle_module());

    // check another standard module

    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::state"));
    MI_CHECK( c_module);
    MI_CHECK_EQUAL_CSTR( "::state", c_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "state", c_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, c_module->get_mdl_package_component_count());
    MI_CHECK( c_module->is_standard_module());
    MI_CHECK( !c_module->is_mdle_module());

    // check regular module

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
    MI_CHECK( c_module);
#ifndef MI_PLATFORM_WINDOWS
    std::string filename = MI::TEST::mi_src_path( "prod/lib/neuray") + "/" TEST_MDL_FILE ".mdl";
#else
    std::string filename = MI::TEST::mi_src_path( "prod\\lib\\neuray") + "\\" TEST_MDL_FILE ".mdl";
#endif
    MI_CHECK_EQUAL( filename, c_module->get_filename());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "" TEST_MDL, c_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, c_module->get_mdl_package_component_count());
    MI_CHECK( !c_module->is_standard_module());
    MI_CHECK( !c_module->is_mdle_module());

    {
        static const char* const imports[] = {
            "mdl::%3Cbuiltins%3E",
            "mdl::df",
            "mdl::anno",
            "mdl::state",
            "mdl::tex",
            "mdl::base",
            "mdl::math",
        };

        mi::Size count = sizeof( imports) / sizeof( const char*);
        MI_CHECK_EQUAL( count, c_module->get_import_count());
        for( mi::Size i = 0; i < count; ++i)
            MI_CHECK_EQUAL_CSTR( imports[i], c_module->get_import( i));
        MI_CHECK( !c_module->get_import( count));
    }
    {
        static const char* const materials[] = {
            "mdl::" TEST_MDL "::md_0()",
            "mdl::" TEST_MDL "::md_1(color)",
            "mdl::" TEST_MDL "::md_1_green(color)",
            "mdl::" TEST_MDL "::md_1_no_default(color)",
            "mdl::" TEST_MDL "::md_2_index(color,color)",
            "mdl::" TEST_MDL "::md_2_index_nested(color,color[N])",
            "mdl::" TEST_MDL "::md_tmp_ms(color)",
            "mdl::" TEST_MDL "::md_tmp_bsdf(color)",
            "mdl::" TEST_MDL "::md_float3_arg(float3)",
            "mdl::" TEST_MDL "::md_with_annotations(color,float)",
            "mdl::" TEST_MDL "::md_enum(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::md_thin_walled(material)",
            "mdl::" TEST_MDL "::md_deferred(int[N])",
            "mdl::" TEST_MDL "::md_deferred_2(int[N],int[N])",
            "mdl::" TEST_MDL "::md_default_call(material)",
            "mdl::" TEST_MDL "::md_ternary_operator_argument(bool)",
            "mdl::" TEST_MDL "::md_ternary_operator_default(color)",
            "mdl::" TEST_MDL "::md_ternary_operator_body(color)",
            "mdl::" TEST_MDL "::md_resource_sharing(texture_2d,texture_2d)",
            "mdl::" TEST_MDL "::md_folding()",
            "mdl::" TEST_MDL "::md_folding2(bool,::" TEST_MDL "::Enum,int)",
            "mdl::" TEST_MDL "::md_folding_cutout_opacity(float,float)",
            "mdl::" TEST_MDL "::md_folding_cutout_opacity2(material)",
            "mdl::" TEST_MDL "::md_folding_transparent_layers(float)",
            "mdl::" TEST_MDL "::md_folding_transparent_layers2(color,float,bool)",
            "mdl::" TEST_MDL "::md_folding_transparent_layers3(float,float)",
            "mdl::" TEST_MDL "::md_parameters_uniform_auto_varying(int,int,int)",
            "mdl::" TEST_MDL "::md_parameter_uniform(color)",
            "mdl::" TEST_MDL "::md_uses_non_exported_function()",
            "mdl::" TEST_MDL "::md_uses_non_exported_material()",
            "mdl::" TEST_MDL "::md_jit(texture_2d,string,light_profile)",
            "mdl::" TEST_MDL "::md_reexport(float3)",
            "mdl::" TEST_MDL "::md_wrap(float,color,color)",
            "mdl::" TEST_MDL "::md_texture(texture_2d)",
            "mdl::" TEST_MDL "::md_light_profile(light_profile)",
            "mdl::" TEST_MDL "::md_bsdf_measurement(bsdf_measurement)",
            "mdl::" TEST_MDL "::md_baking()",
            "mdl::" TEST_MDL "::md_class_baking(float,::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::md_named_temporaries(float)"
        };

        mi::Size count = sizeof( materials) / sizeof( const char*);
        MI_CHECK_EQUAL( count, c_module->get_material_count());
        for( mi::Size i = 0; i < count; ++i)
            MI_CHECK_EQUAL_CSTR( materials[i], c_module->get_material( i));
        MI_CHECK( !c_module->get_material( count));
    }
    {
        static const char* const functions[] = {
            "mdl::" TEST_MDL "::normal()",
            "mdl::" TEST_MDL "::fd_0()",
            "mdl::" TEST_MDL "::fd_1(int)",
            "mdl::" TEST_MDL "::fd_1_44(int)",
            "mdl::" TEST_MDL "::fd_1_no_default(int)",
            "mdl::" TEST_MDL "::fd_2_index(int,float)",
            "mdl::" TEST_MDL "::fd_2_index_nested(int,int[N])",
            "mdl::" TEST_MDL "::fd_with_annotations(color,float)",
            "mdl::" TEST_MDL "::Enum(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::int(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::fd_enum(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::fd_remove(int)",
            "mdl::" TEST_MDL "::fd_overloaded(int)",
            "mdl::" TEST_MDL "::fd_overloaded(float)",
            "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
            "mdl::" TEST_MDL "::fd_overloaded(int[3])",
            "mdl::" TEST_MDL "::fd_overloaded(int[N])",
            "mdl::" TEST_MDL "::foo_struct(::" TEST_MDL "::foo_struct)",
            "mdl::" TEST_MDL "::foo_struct()",          // default constructor
            "mdl::" TEST_MDL "::foo_struct(int,float)", // elemental constructor
            "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)", // field access op.
            "mdl::" TEST_MDL "::foo_struct.param_float(::" TEST_MDL "::foo_struct)", // field access op.
            "mdl::" TEST_MDL "::fd_immediate(int[42])",
            "mdl::" TEST_MDL "::fd_deferred(int[N])",
            "mdl::" TEST_MDL "::fd_deferred_2(int[N],int[N])",
            "mdl::" TEST_MDL "::fd_deferred_struct(::" TEST_MDL "::foo_struct[N])",
            "mdl::" TEST_MDL "::fd_matrix(float3x2)",
            "mdl::" TEST_MDL "::fd_default_call(int)",
            "mdl::" TEST_MDL "::Enum2(::" TEST_MDL "::Enum2)",
            "mdl::" TEST_MDL "::int(::" TEST_MDL "::Enum2)",
            "mdl::" TEST_MDL "::fd_ret_bool()",
            "mdl::" TEST_MDL "::fd_ret_int()",
            "mdl::" TEST_MDL "::fd_ret_float()",
            "mdl::" TEST_MDL "::fd_ret_string()",
            "mdl::" TEST_MDL "::fd_ret_enum()",
            "mdl::" TEST_MDL "::fd_ret_enum2()",
            "mdl::" TEST_MDL "::fd_add(int,int)",
            "mdl::" TEST_MDL "::fd_return_uniform()",
            "mdl::" TEST_MDL "::fd_return_auto()",
            "mdl::" TEST_MDL "::fd_return_varying()",
            "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying(int,int,int)",
            "mdl::" TEST_MDL "::fd_uniform()",
            "mdl::" TEST_MDL "::fd_varying()",
            "mdl::" TEST_MDL "::fd_auto_uniform()",
            "mdl::" TEST_MDL "::fd_auto_varying()",
            "mdl::" TEST_MDL "::fd_non_exported()",     // local function
            "mdl::" TEST_MDL "::lookup(int)",
            "mdl::" TEST_MDL "::fd_jit(float)",
            "mdl::" TEST_MDL "::color_weight(texture_2d,light_profile)",  // local function
            "mdl::" TEST_MDL "::color_weight(string)",  // local function
            "mdl::" TEST_MDL "::fd_wrap_x(float)",
            "mdl::" TEST_MDL "::fd_wrap_rhs(color,color,color)",
            "mdl::" TEST_MDL "::fd_wrap_r(color)",
            "mdl::" TEST_MDL "::fd_cycle(color)",
            "mdl::" TEST_MDL "::sub_struct(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct()",
            "mdl::" TEST_MDL "::sub_struct(bool,bool,color,float)",
            "mdl::" TEST_MDL "::sub_struct.a(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct.b(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct.c(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct.d(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::top_struct(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct()",
            "mdl::" TEST_MDL "::top_struct(float,double,::" TEST_MDL "::sub_struct,int,"
                "::" TEST_MDL "::sub_struct,float)",
            "mdl::" TEST_MDL "::top_struct.a(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.x(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.b(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.c(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.d(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.e(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::fd_identity(int)",
            "mdl::" TEST_MDL "::fd_named_temporaries(int)",
            "mdl::" TEST_MDL "::fd_parameter_references(float,float,float)",
            "mdl::" TEST_MDL "::fd_global_scope_reference_test(int)"
        };

#if 0
        for( mi::Size i = 0, n = c_module->get_function_count(); i < n; ++i)
            printf("\"%s\",\n", c_module->get_function( i));
#endif

        mi::Size count = sizeof( functions) / sizeof( const char*);
        MI_CHECK_EQUAL( count, c_module->get_function_count());
        for( mi::Size i = 0; i < count; ++i)
            MI_CHECK_EQUAL_CSTR( functions[i], c_module->get_function( i));
        MI_CHECK( !c_module->get_function( count));
    }

    // check module in archive

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_archives");
    MI_CHECK( c_module);

#ifndef MI_PLATFORM_WINDOWS
    filename = MI::TEST::mi_src_path( "prod/lib/neuray") + "/test_archives.mdr";
#else
    filename = MI::TEST::mi_src_path( "prod\\lib\\neuray") + "\\test_archives.mdr";
#endif
    MI_CHECK_EQUAL( filename, c_module->get_filename());
    MI_CHECK_EQUAL_CSTR( "::test_archives", c_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "test_archives", c_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, c_module->get_mdl_package_component_count());
    MI_CHECK( !c_module->is_standard_module());
    MI_CHECK( !c_module->is_mdle_module());

    // check MDLE module

    mi::base::Handle<const mi::neuraylib::IModule> m_module(
        transaction->access<mi::neuraylib::IModule>( mdle_module_db_name.c_str()));
    MI_CHECK_EQUAL( 0, m_module->get_mdl_package_component_count());
    MI_CHECK( !m_module->is_standard_module());
    MI_CHECK( m_module->is_mdle_module());
    MI_CHECK_EQUAL_CSTR( m_module->get_mdl_name()+2, m_module->get_mdl_simple_name());

    {
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        std::string fd_name
            = mdle_module_db_name + "::main(texture_2d,light_profile,bsdf_measurement)";

        Mdle_serialization_callback serialization_cb( factory);
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                fd_name.c_str(),
                /*argument_types*/ nullptr,
                /*return_type*/ nullptr,
                &serialization_cb,
                context.get()));
        MI_CHECK_CTX( context.get());

        // "::FOO" should occur once for the module name
        std::string s = sfn->get_function_name();
        size_t first_foo  = s.find( "::FOO");
        MI_CHECK_NOT_EQUAL( first_foo, std::string::npos);
        size_t left_paren = s.find( "(", first_foo);
        MI_CHECK_NOT_EQUAL( left_paren, std::string::npos);

        Mdle_deserialization_callback deserialization_cb( factory);
        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), &deserialization_cb, context.get()));

        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), fd_name.c_str());

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                &deserialization_cb,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), fd_name.c_str());

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), &deserialization_cb, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), mdle_module_db_name.c_str());
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }

    // check get_function_overloads() (no expression list, retrieve all overloads)

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
    mi::base::Handle<const mi::IArray> result;
    mi::base::Handle<const mi::IString> element;
    MI_CHECK( !c_module->get_function_overloads( 0));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::non-existing");
    MI_CHECK_EQUAL( 0, result->get_length());
    result = c_module->get_function_overloads( "non-existing");
    MI_CHECK_EQUAL( 0, result->get_length());

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded");
    MI_CHECK_EQUAL( 6, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());
    element = result->get_element<mi::IString>( 1);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());
    element = result->get_element<mi::IString>( 2);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());
    element = result->get_element<mi::IString>( 3);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());
    element = result->get_element<mi::IString>( 4);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());
    element = result->get_element<mi::IString>( 5);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    result = c_module->get_function_overloads( "fd_overloaded");
    MI_CHECK_EQUAL( 6, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());
    element = result->get_element<mi::IString>( 1);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());
    element = result->get_element<mi::IString>( 2);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());
    element = result->get_element<mi::IString>( 3);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());
    element = result->get_element<mi::IString>( 4);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());
    element = result->get_element<mi::IString>( 5);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::md_1");
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", element->get_c_str());

    result = c_module->get_function_overloads( "md_1");
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", element->get_c_str());

    // check get_function_overloads (expression list, retrieve exact match overload)
    mi::base::Handle<mi::neuraylib::IValue> value_int( vf->create_int( 42));
    mi::base::Handle<mi::neuraylib::IExpression> expr_int(
        ef->create_constant( value_int.get()));
    mi::base::Handle<mi::neuraylib::IValue> value_color( vf->create_color( 1.0, 1.0, 1.0));
    mi::base::Handle<mi::neuraylib::IExpression> expr_color(
        ef->create_constant( value_color.get()));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arguments->add_expression( "param0", expr_int.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "param0", expr_color.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "tint", expr_color.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::md_1", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", element->get_c_str());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "tint", expr_int.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::md_1", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    // check get_function_overloads() (signature, retrieve best-matching overloads)
    mi::base::Handle<mi::IArray> parameter_types;
    MI_CHECK( !c_module->get_function_overloads( 0, parameter_types.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::non-existing", parameter_types.get());
    MI_CHECK_EQUAL( 0, result->get_length());
    result = c_module->get_function_overloads( "non-existing", parameter_types.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 6, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());
    element = result->get_element<mi::IString>( 1);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());
    element = result->get_element<mi::IString>( 2);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());
    element = result->get_element<mi::IString>( 3);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());
    element = result->get_element<mi::IString>( 4);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());
    // the overload with "int[N]" is not in the result set because "int[3]" is more specific.

    parameter_types = create_istring_array( factory, { "int" });
    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());

    parameter_types = create_istring_array( factory, { "float" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());

    parameter_types = create_istring_array( factory, { "::" TEST_MDL "::Enum" });
    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());

    parameter_types = create_istring_array( factory, { "::state::coordinate_space" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());

    parameter_types = create_istring_array( factory, { "int[3]" });
    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());

    parameter_types = create_istring_array( factory, { "int[6]" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    // the deferred symbol size symbol is meaningless
    parameter_types = create_istring_array( factory, { "int[XnonexistantX]" });
    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    // the deferred symbol size symbol is meaningless, even empty is allowed
    parameter_types = create_istring_array( factory, { "int[]" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    parameter_types = create_istring_array( factory, { "int" });
    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::foo_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::foo_struct(int,float)", element->get_c_str());

    parameter_types = create_istring_array( factory, { });
    result = c_module->get_function_overloads( "foo_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::foo_struct()", element->get_c_str());

    parameter_types = create_istring_array( factory, { });
    result = b_module->get_function_overloads( "mdl::int", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::int(int)", element->get_c_str()); // Best match due to default argument

    parameter_types = create_istring_array( factory, { "float" });
    result = b_module->get_function_overloads( "int", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::int(float)", element->get_c_str());

    c_module = transaction->access<mi::neuraylib::IModule>("mdl::test_mdl2");

    // check if reexported structs can be resolved using the new name
    parameter_types = create_istring_array( factory, { "::test_mdl2::foo_struct" });
    result = c_module->get_function_overloads( "mdl::test_mdl2::fd_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::test_mdl2::fd_reexported_struct(::" TEST_MDL "::foo_struct)", element->get_c_str());

    // check if reexported structs can be resolved using the original name
    parameter_types = create_istring_array( factory, { "::" TEST_MDL "::foo_struct" });
    result = c_module->get_function_overloads( "fd_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::test_mdl2::fd_reexported_struct(::" TEST_MDL "::foo_struct)", element->get_c_str());

    // same as above but this time 'foo_struct' is not re-exported
    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl3");

    parameter_types = create_istring_array( factory, { "::test_mdl3::foo_struct" });
    result = c_module->get_function_overloads( "mdl::test_mdl3::fd_using_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::test_mdl3::fd_using_reexported_struct(::" TEST_MDL "::foo_struct)", element->get_c_str());

    parameter_types = create_istring_array( factory, { "::" TEST_MDL "::foo_struct" });
    result = c_module->get_function_overloads( "fd_using_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::test_mdl3::fd_using_reexported_struct(::" TEST_MDL "::foo_struct)", element->get_c_str());

    // check ::%3Cbuiltins%3E module

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::%3Cbuiltins%3E");

    // array length operator without arguments
    arguments = ef->create_expression_list();
    result = c_module->get_function_overloads( "mdl::operator_len", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    // array length operator with arguments
    mi::base::Handle<const mi::neuraylib::IType> type_int( tf->create_int());
    mi::base::Handle<const mi::neuraylib::IType_array> type_int2(
        tf->create_immediate_sized_array( type_int.get(), 2));
    mi::base::Handle<const mi::neuraylib::IValue> value( vf->create_array( type_int2.get()));
    mi::base::Handle<const mi::neuraylib::IExpression> expr( ef->create_constant( value.get()));
    MI_CHECK_EQUAL( 0, arguments->add_expression( "a", expr.get()));
    result = c_module->get_function_overloads( "operator_len", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
         "mdl::operator_len(%3C0%3E[])", element->get_c_str());

    // array index operator without arguments
    arguments = ef->create_expression_list();
    result = c_module->get_function_overloads( "mdl::operator[]", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    // array index operator with arguments
    MI_CHECK_EQUAL( 0, arguments->add_expression( "a", expr.get()));
    value = vf->create_int( 0);
    expr = ef->create_constant( value.get());
    MI_CHECK_EQUAL( 0, arguments->add_expression( "i", expr.get()));
    result = c_module->get_function_overloads( "operator[]", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
         "mdl::operator[](%3C0%3E[],int)", element->get_c_str());

    // operator with encoded characters
    result = c_module->get_function_overloads(
         "mdl::operator%3C",
        static_cast<const mi::neuraylib::IExpression_list*>( nullptr));
    MI_CHECK_EQUAL( 3, result->get_length());

    // check types

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
    mi::base::Handle<const mi::neuraylib::IType_list> types( c_module->get_types());
    MI_CHECK_EQUAL( 5, types->get_size());
    mi::base::Handle<const mi::neuraylib::IType_enum> type0(
        types->get_type<mi::neuraylib::IType_enum>( zero_size));
    MI_CHECK( type0);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", types->get_name( 0));
    mi::base::Handle<const mi::neuraylib::IType_struct> type1(
        types->get_type<mi::neuraylib::IType_struct>( 1));
    MI_CHECK( type1);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", types->get_name( 1));
    // just test the API binding, extensive test is in io/scene/mdl_elements/test_misc.cpp
    mi::base::Handle<const mi::IString> module_name( tf->get_mdl_module_name( type1.get()));
    MI_CHECK_EQUAL_CSTR( module_name->get_c_str(), "::" TEST_MDL);
    // just test the API binding, extensive test is in io/scene/mdl_elements/test_types.cpp
    mi::base::Handle<const mi::IString> mdl_type_name( tf->get_mdl_type_name( type1.get()));
    MI_CHECK_EQUAL_CSTR( mdl_type_name->get_c_str(), "::" TEST_MDL "::foo_struct");
    mi::base::Handle<const mi::neuraylib::IType_struct> type2(
        tf->create_from_mdl_type_name<mi::neuraylib::IType_struct>( mdl_type_name->get_c_str()));
    MI_CHECK_EQUAL( tf->compare( type1.get(), type2.get()), 0);

    // check constants

    mi::base::Handle<const mi::neuraylib::IValue_list> constants( c_module->get_constants());
    MI_CHECK_EQUAL( 1, constants->get_size());
    mi::base::Handle<const mi::neuraylib::IValue_int> constant0(
        constants->get_value<mi::neuraylib::IValue_int>( zero_size));
    MI_CHECK_EQUAL( 42, constant0->get_value());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::some_constant", constants->get_name( 0));

    // check MDL versions

    MI_CHECK_EQUAL( mi::neuraylib::MDL_VERSION_1_7, c_module->get_mdl_version());
}

void check_mdl_reload(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create material instances
        const char* materials[] = {
            "mdl::test_mdl_reload::md_1(color)", "mdl::test_mdl_reload::mi_1",
            "mdl::test_mdl_reload::md_2(float)", "mdl::test_mdl_reload::mi_2",
            "mdl::test_mdl_reload::md_2(float)", "mdl::test_mdl_reload::mi_2_2",
            "mdl::test_mdl_reload::md_3()",      "mdl::test_mdl_reload::mi_3",
            "mdl::test_mdl_reload::md_4(float)", "mdl::test_mdl_reload::mi_4"
        };

        for (mi::Size i = 0; i < sizeof(materials) / sizeof(const char*); i += 2)
            do_create_function_call(
                transaction, materials[i], materials[i + 1]);

        // create function calls
        do_create_function_call(
            transaction, "mdl::test_mdl_reload::fd_1(float)", "mdl::test_mdl_reload::fd_1_inst");

        mi::neuraylib::Argument_editor ae(transaction, "mdl::test_mdl_reload::mi_2_2", mdl_factory);
        MI_CHECK(ae.is_valid());
        MI_CHECK_EQUAL(ae.set_call("f", "mdl::test_mdl_reload::fd_1_inst"), 0);

       // create compiled materials
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> minst(
            transaction->access<mi::neuraylib::IMaterial_instance>("mdl::test_mdl_reload::mi_1"));
        MI_CHECK(minst);
        mi::base::Handle<mi::neuraylib::ICompiled_material> cinst(
            minst->create_compiled_material(mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, nullptr));
        MI_CHECK(cinst);
        transaction->store(cinst.get(), "mdl::test_mdl_reload::mi_1_cmp");
    }

    {
        // modify file on disk and reload
        fs::copy_file(
            MI::TEST::mi_src_path("prod/lib/neuray") + "/test_mdl_reload_1.mdl",
            DIR_PREFIX "/test_mdl_reload.mdl", fs::copy_options::overwrite_existing);
        mi::base::Handle<mi::neuraylib::IModule> module(
            transaction->edit<mi::neuraylib::IModule>("mdl::test_mdl_reload"));
        result = module->reload(false, context.get());
        MI_CHECK_EQUAL(0, result);
        MI_CHECK_EQUAL(3, module->get_material_count());
    }

    {
        // access some definitions
        mi::base::Handle<const mi::neuraylib::IFunction_definition> mdef1(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::md_1(color,color)"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> mdef2(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::md_2(float)"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> mdef3(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::md_3()"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fdef1(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload::fd_1(float)"));

        // check old definitions
        MI_CHECK_EQUAL(true,  mdef1->is_valid(context.get()));
        MI_CHECK_EQUAL(true,  mdef2->is_valid(context.get()));
        MI_CHECK_EQUAL(false, mdef3->is_valid(context.get()));
        MI_CHECK_EQUAL(nullptr, mdef3->create_function_call(nullptr, &result));
        MI_CHECK_EQUAL(-9, result);
        MI_CHECK_EQUAL(true, fdef1->is_valid(context.get()));

        // check old instances
        mi::base::Handle<const mi::neuraylib::IFunction_call> inst1(
            transaction->access<mi::neuraylib::IFunction_call>("mdl::test_mdl_reload::mi_1"));
        MI_CHECK_EQUAL(false, inst1->is_valid(context.get()));

        // ensure we cannot create a compiled material from this invalid instance.
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> inst1_mi(
            transaction->access<mi::neuraylib::IMaterial_instance>("mdl::test_mdl_reload::mi_1"));
        mi::base::Handle<mi::neuraylib::ICompiled_material> cinst(
            inst1_mi->create_compiled_material(mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, nullptr));
        MI_CHECK(!cinst);

        mi::base::Handle<const mi::neuraylib::IFunction_call> inst2(
            transaction->access<mi::neuraylib::IFunction_call>("mdl::test_mdl_reload::mi_2"));
        // this instance is still valid
        MI_CHECK_EQUAL(true, inst2->is_valid(context.get()));

        mi::base::Handle<const mi::neuraylib::IFunction_call> inst3(
            transaction->access<mi::neuraylib::IFunction_call>("mdl::test_mdl_reload::mi_3"));
        MI_CHECK_EQUAL(false, inst3->is_valid(context.get()));

        mi::base::Handle<const mi::neuraylib::IFunction_call> call1(
            transaction->access<mi::neuraylib::IFunction_call>("mdl::test_mdl_reload::fd_1_inst"));
        // interface has changed.
        MI_CHECK_EQUAL(false, call1->is_valid(context.get()));

        mi::base::Handle<mi::neuraylib::IFunction_call> inst2a(
            transaction->edit<mi::neuraylib::IFunction_call>("mdl::test_mdl_reload::mi_2_2"));
        // this instance has an invalid attachment
        MI_CHECK_EQUAL(false, inst2a->is_valid(context.get()));

        // try to repair. this is not possible, the return type of the attached function has changed.
        MI_CHECK_EQUAL(-1, inst2a->repair(mi::neuraylib::MDL_REPAIR_INVALID_ARGUMENTS, context.get()));

        // try to repair by removing the invalid argument.
        MI_CHECK_EQUAL(0, inst2a->repair(mi::neuraylib::MDL_REMOVE_INVALID_ARGUMENTS, context.get()));

        // the instance is valid again.
        MI_CHECK_EQUAL(true, inst2a->is_valid(context.get()));

        mi::base::Handle<mi::neuraylib::IFunction_call> inst4(
            transaction->edit<mi::neuraylib::IFunction_call>("mdl::test_mdl_reload::mi_4"));
        // repair.
        MI_CHECK_EQUAL(0, inst2a->repair(mi::neuraylib::MDL_REPAIR_DEFAULT, context.get()));
    }

    {
        // check module that imports mdl::test_mdl_reload
        mi::base::Handle<mi::neuraylib::IModule> module1(
            transaction->edit<mi::neuraylib::IModule>("mdl::test_mdl_reload_import"));

        // the module should be invalid
        MI_CHECK_EQUAL(false, module1->is_valid(context.get()));

        // so should its definitions
        mi::base::Handle<const mi::neuraylib::IFunction_definition> mdef11(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload_import::md_1(color)"));
        MI_CHECK_EQUAL(false, mdef11->is_valid(context.get()));

        mi::base::Handle<const mi::neuraylib::IFunction_definition> mdef21(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_mdl_reload_import::md_2(float)"));
        MI_CHECK_EQUAL(false, mdef21->is_valid(context.get()));

        // reload
        MI_CHECK_EQUAL(0, module1->reload(/*recursive=*/true, context.get()));

        MI_CHECK_EQUAL(true, module1->is_valid(context.get()));
        // a default has changed. therefore, this definition has been replaced.
        MI_CHECK_EQUAL(false, mdef11->is_valid(context.get()));
        mdef11 = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::test_mdl_reload_import::md_1(color)");
        MI_CHECK_EQUAL(true, mdef11->is_valid(context.get()));
        MI_CHECK_EQUAL(true, mdef21->is_valid(context.get()));
    }

    {
        // modify file on disk again and try to reload
        fs::copy_file(
            MI::TEST::mi_src_path("prod/lib/neuray") + "/test_mdl_reload_2.mdl",
            DIR_PREFIX "/test_mdl_reload.mdl",
            fs::copy_options::overwrite_existing);
        mi::base::Handle<mi::neuraylib::IModule> module(
            transaction->edit<mi::neuraylib::IModule>("mdl::test_mdl_reload"));
        result = module->reload(false, context.get());
        MI_CHECK_EQUAL(-1, result);
        MI_CHECK_EQUAL( 1, context->get_error_messages_count());
    }

    {
        // check, that reloading base.mdl is prohibited
        mi::base::Handle<mi::neuraylib::IModule> base_module(
            transaction->edit<mi::neuraylib::IModule>("mdl::base"));
        if (base_module) {
            result = base_module->reload(/*recursive=*/false, context.get());
            MI_CHECK_EQUAL(-1, result);
            MI_CHECK_EQUAL(1, context->get_error_messages_count());
        }
    }

    {
        // check, that reloading distilling_support.mdl is prohibited
        mi::base::Handle<mi::neuraylib::IModule> ds_module(
            transaction->edit<mi::neuraylib::IModule>("mdl::nvidia::distilling_support"));
        if (ds_module) {
            result = ds_module->reload(/*recursive=*/false, context.get());
            MI_CHECK_EQUAL(-1, result);
            MI_CHECK_EQUAL(1, context->get_error_messages_count());
        }
    }

    // check reload from string
    {
        mi::base::Handle<mi::neuraylib::IModule> s_module(
            transaction->edit<mi::neuraylib::IModule>("mdl::test_mdl_reload_from_string"));
        MI_CHECK(s_module.is_valid_interface());
        // null-source string should fail
        result = s_module->reload_from_string(/*module_source=*/nullptr, /*recursive=*/false, context.get());
        MI_CHECK_EQUAL(-1, result);
        MI_CHECK_EQUAL(1,  context->get_error_messages_count());

        // empty-source string should fail, too.
        result = s_module->reload_from_string(/*module_source=*/"", /*recursive=*/false, context.get());
        MI_CHECK_EQUAL(-1, result);
        MI_CHECK_EQUAL(1, context->get_error_messages_count());

        const char* module_source = "mdl 1.0; export float some_function() { return 1.0; }";
        result = s_module->reload_from_string(module_source, /*recursive=*/false, context.get());
        MI_CHECK_EQUAL(0, result);
        MI_CHECK_EQUAL(0, context->get_error_messages_count());

        mi::Size material_count = s_module->get_material_count();
        MI_CHECK_EQUAL(0, material_count);

        mi::Size function_count = s_module->get_function_count();
        MI_CHECK_EQUAL(1, function_count);

        const char* filename = s_module->get_filename();
        MI_CHECK_EQUAL(0, filename);
    }

    // check recursive reloads, test_mdl_reload_4/5.mdl import test_mdl_reload_3.mdl
    {
        {
            // create the three modules
            std::ofstream module_3( DIR_PREFIX "/test_mdl_reload_3.mdl");
            module_3 << "mdl 1.3;";
            std::ofstream module_4( DIR_PREFIX "/test_mdl_reload_4.mdl");
            module_4 << "mdl 1.3; \
                         import ::test_mdl_reload_3::*;";
            std::ofstream module_5( DIR_PREFIX "/test_mdl_reload_5.mdl");
            module_5 << "mdl 1.3; \
                         import ::test_mdl_reload_3::*;";
        }

        // and load them
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::test_mdl_reload_3"));
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::test_mdl_reload_4"));
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::test_mdl_reload_5"));

        mi::base::Handle<mi::neuraylib::IModule> module;
        mi::base::Handle<const mi::neuraylib::IModule> c_module;

        {
            // reload test_mdl_reload_4.mdl recursively, check that all modules remain valid
            module = transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_4");
            MI_CHECK_EQUAL( 0, module->reload( true, context.get()));
            MI_CHECK( module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_3");
            MI_CHECK( c_module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_5");
            MI_CHECK( c_module->is_valid( context.get()));
        }
        {
            // change test_mdl_reload_3.mdl in a compatible way
            std::ofstream module_3( DIR_PREFIX "/test_mdl_reload_3.mdl");
            module_3 << "mdl 1.3; \
                         export int f() { return 42; }";
        }
        {
            // reload test_mdl_reload_4.mdl recursively, check that all modules but test_mdl_reload_5.mdl are valid
            module = transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_4");
            MI_CHECK_EQUAL( 0, module->reload( true, context.get()));
            MI_CHECK( module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_3");
            MI_CHECK( c_module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_5");
            MI_CHECK( !c_module->is_valid( context.get()));
        }
        {
            // reload also test_mdl_reload_5.mdl recursively, check that all modules are now valid
            module = transaction->edit<mi::neuraylib::IModule>( "mdl::test_mdl_reload_5");
            MI_CHECK_EQUAL( 0, module->reload( true, context.get()));
            MI_CHECK( module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_3");
            MI_CHECK( c_module->is_valid( context.get()));
            c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_reload_4");
            MI_CHECK( c_module->is_valid( context.get()));
        }
    }
}

void check_mdl_reload_compiled_materials(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api)
{
    // access compiled material that is no longer valid
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cinst(
        transaction->access<mi::neuraylib::ICompiled_material>("mdl::test_mdl_reload::mi_1_cmp"));
    MI_CHECK(cinst);

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    MI_CHECK_EQUAL(false, cinst->is_valid(context.get()));
    mi::base::Handle<mi::neuraylib::IMdl_backend> backend(
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE));

    mi::base::Handle<const mi::neuraylib::ITarget_code> tc(
        backend->translate_material_df(transaction, cinst.get(), "surface.scattering", "my_scatter", nullptr));
    MI_CHECK(!tc.is_valid_interface());

}

void check_ifunction_definition(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> t;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults;
    mi::base::Handle<const mi::neuraylib::IExpression> body;

    // check a function definition with no arguments

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_0()"));
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_fd->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_0()", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_0", c_fd->get_mdl_simple_name());
    MI_CHECK( !c_fd->get_prototype());
    t = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    body = c_fd->get_body();
    MI_CHECK( body);
    MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, body->get_kind());

    MI_CHECK_EQUAL( 0, c_fd->get_parameter_count());
    MI_CHECK( !c_fd->get_parameter_name( 0));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( "invalid"));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( 0));

    types = c_fd->get_parameter_types();
    MI_CHECK( !types->get_type( zero_size));
    MI_CHECK( !types->get_type( zero_string));
    MI_CHECK( !types->get_type( "invalid"));

    defaults = c_fd->get_defaults();
    MI_CHECK( !defaults->get_expression( zero_size));
    MI_CHECK( !defaults->get_expression( zero_string));
    MI_CHECK( !defaults->get_expression( "invalid"));

    // check a function definition with arguments

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_1(int)");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_fd->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_1(int)", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_1", c_fd->get_mdl_simple_name());
    MI_CHECK( !c_fd->get_prototype());
    t = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    body = c_fd->get_body();
    MI_CHECK( body);
    MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_PARAMETER, body->get_kind());

    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( 0));
    MI_CHECK( !c_fd->get_parameter_name( 1));

    MI_CHECK_EQUAL( 0, c_fd->get_parameter_index( "param0"));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( 0));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( "invalid"));

    types = c_fd->get_parameter_types();
    t = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    MI_CHECK( !types->get_type( 1));
    t = types->get_type( "param0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    MI_CHECK( !types->get_type( zero_string));
    MI_CHECK( !types->get_type( "invalid"));

    MI_CHECK_EQUAL_CSTR( "int", c_fd->get_mdl_parameter_type_name( 0));
    MI_CHECK( !c_fd->get_mdl_parameter_type_name( 1));

    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_default;
    mi::base::Handle<const mi::neuraylib::IValue_int> c_value_int;

    defaults = c_fd->get_defaults();
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
    t = c_default->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    c_value_int = c_default->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( c_value_int->get_value(), 42);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( 1);
    MI_CHECK( !c_default);

    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0");
    t = c_default->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    c_value_int = c_default->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( c_value_int->get_value(), 42);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_string);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "invalid");
    MI_CHECK( !c_default);

    // check a function definition with arguments without defaults

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_1_no_default(int)");
    MI_CHECK( c_fd);

    defaults = c_fd->get_defaults();
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( 1);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0");
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_string);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "invalid");
    MI_CHECK( !c_default);

    // check is_uniform() property

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_uniform()");
    MI_CHECK( c_fd->is_uniform());
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_varying()");
    MI_CHECK( !c_fd->is_uniform());
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_auto_uniform()");
    MI_CHECK( c_fd->is_uniform());
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_auto_varying()");
    MI_CHECK( !c_fd->is_uniform());

    // check a function definition which is a variant

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_1_44(int)");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_1_44(int)", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_1_44", c_fd->get_mdl_simple_name());
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_1(int)", c_fd->get_prototype());

    // check MDL versions

    mi::neuraylib::Mdl_version since, removed;

    // removed in MDL 1.1
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::df::measured_edf$1.0(light_profile,bool,float3x3,string)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_1_1);

    // removed in MDL 1.2
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::df::measured_edf$1.1(light_profile,float,bool,float3x3,string)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_1);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_1_2);

    // not yet removed
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::df::measured_edf(light_profile,float,bool,float3x3,float3,string)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_2);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // not from stdlib
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_0()");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_7);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: cast operator
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator_cast(%3C0%3E)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_5);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: DAG intrinsic
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator_len(%3C0%3E[])");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: operator
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator[](%3C0%3E[],int)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);
}

void check_imaterial_definition(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IValue> v_color( vf->create_color( 1.0f, 1.0f, 1.0f));
    mi::base::Handle<const mi::neuraylib::IType> t_color( v_color->get_type());

    mi::base::Handle<const mi::neuraylib::IType> t;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults;

    // check a material definition with no arguments

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_0()"));
    MI_CHECK( c_md);
    MI_CHECK_EQUAL( c_md->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        c_md->get_interface<mi::neuraylib::IFunction_definition>());
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( c_fd->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);
    mi::base::Handle<const mi::neuraylib::IScene_element> c_se(
        c_md->get_interface<mi::neuraylib::IScene_element>());
    MI_CHECK( c_se);
    MI_CHECK_EQUAL( c_se->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_0()");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( c_fd->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_md->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_md->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::md_0()", c_md->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "md_0", c_md->get_mdl_simple_name());
    MI_CHECK( !c_md->get_prototype());

    MI_CHECK_EQUAL( 0, c_md->get_parameter_count());
    MI_CHECK( !c_md->get_parameter_name( 0));
    MI_CHECK_EQUAL( minus_one_size, c_md->get_parameter_index( 0));
    MI_CHECK_EQUAL( minus_one_size, c_md->get_parameter_index( "invalid"));

    types = c_md->get_parameter_types();
    MI_CHECK( !types->get_type( zero_size));
    MI_CHECK( !types->get_type( zero_string));
    MI_CHECK( !types->get_type( "invalid"));

    defaults = c_md->get_defaults();
    MI_CHECK( !defaults->get_expression( zero_size));
    MI_CHECK( !defaults->get_expression( zero_string));
    MI_CHECK( !defaults->get_expression( "invalid"));

    // check a material definition with arguments with defaults

    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_1(color)");
    MI_CHECK( c_md);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_md->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_md->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::md_1(color)", c_md->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "md_1", c_md->get_mdl_simple_name());
    MI_CHECK( !c_md->get_prototype());

    MI_CHECK_EQUAL( 1, c_md->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "tint", c_md->get_parameter_name( 0));
    MI_CHECK( !c_md->get_parameter_name( 1));
    MI_CHECK_EQUAL( 0, c_md->get_parameter_index( "tint"));
    MI_CHECK_EQUAL( minus_one_size, c_md->get_parameter_index( 0));
    MI_CHECK_EQUAL( minus_one_size, c_md->get_parameter_index( "invalid"));

    types = c_md->get_parameter_types();
    t = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, t->get_kind());
    MI_CHECK( !types->get_type( 1));
    t = types->get_type( "tint");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, t->get_kind());
    MI_CHECK( !types->get_type( zero_string));
    MI_CHECK( !types->get_type( "invalid"));

    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_default;
    mi::base::Handle<const mi::neuraylib::IValue_color> c_value_color;

    defaults = c_md->get_defaults();
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
    t = c_default->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, t->get_kind());
    c_value_color = c_default->get_value<mi::neuraylib::IValue_color>();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, t->get_kind());
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( 1);
    MI_CHECK( !c_default);

    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "tint");
    t = c_default->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, t->get_kind());
    c_value_color = c_default->get_value<mi::neuraylib::IValue_color>();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, t->get_kind());
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_string);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "invalid");
    MI_CHECK( !c_default);

    // check a material definition with arguments without defaults

    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_1_no_default(color)");
    MI_CHECK( c_md);

    defaults = c_md->get_defaults();
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( 1);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "tint");
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_string);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "invalid");
    MI_CHECK( !c_default);

    // check a material definition which is a variant

    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_1_green(color)");
    MI_CHECK( c_md);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_md->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_md->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::md_1_green(color)", c_md->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "md_1_green", c_md->get_mdl_simple_name());
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", c_md->get_prototype());

    // check MDL versions

    mi::neuraylib::Mdl_version since, removed;

    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_0()");
    c_md->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_7);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

}

void check_ifunction_call(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> t;

    // check a function call with no arguments

    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_0"));
    MI_CHECK( c_fc);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_0()", c_fc->get_function_definition());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_0()", c_fc->get_mdl_function_definition());
    t = c_fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    MI_CHECK_EQUAL( 0, c_fc->get_parameter_count());
    MI_CHECK( !c_fc->get_parameter_name( 0));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( "invalid"));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( 0));

    // check a function call with one argument

    c_fc = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1");
    MI_CHECK( c_fc);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_1(int)", c_fc->get_function_definition());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_1(int)", c_fc->get_mdl_function_definition());
    t = c_fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    MI_CHECK_EQUAL( 1, c_fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "param0", c_fc->get_parameter_name( 0));
    MI_CHECK( !c_fc->get_parameter_name( 1));

    MI_CHECK_EQUAL( 0, c_fc->get_parameter_index( "param0"));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( 0));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( "invalid"));

    // check serialized names

    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::" TEST_MDL "::fd_1(int)",
            /*argument_types*/ nullptr,
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::" TEST_MDL "::fd_1(int)");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "fd_1(int)");

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        c_fc->get_parameter_types());

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::" TEST_MDL "::fd_1(int)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
    MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::" TEST_MDL "::fd_1(int)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
    MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::" TEST_MDL);
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL( result, 1);

    c_fc = 0;

    // check get_argument()/set_argument() (constants only)

    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1"));
    MI_CHECK( m_fc);

    mi::base::Handle<mi::neuraylib::IValue_int> m_value_int( vf->create_int( -42));
    mi::base::Handle<mi::neuraylib::IExpression_constant> m_expression(
        ef->create_constant( m_value_int.get()));
    MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_expression.get()));

    mi::base::Handle<const mi::neuraylib::IExpression_list> c_expression_list( m_fc->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_expression(
        c_expression_list->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
    mi::base::Handle<const mi::neuraylib::IValue_int> c_value_int(
        c_expression->get_value<mi::neuraylib::IValue_int>());
    MI_CHECK_EQUAL( -42, c_value_int->get_value());

    // check error codes of set_argument()

    MI_CHECK_EQUAL( -1, m_fc->set_argument( zero_string, m_expression.get()));
    MI_CHECK_EQUAL( -1, m_fc->set_argument( "param0", 0));
    MI_CHECK_EQUAL( -2, m_fc->set_argument( 1, m_expression.get()));
    MI_CHECK_EQUAL( -2, m_fc->set_argument( "invalid", m_expression.get()));
    mi::base::Handle<mi::neuraylib::IValue> m_value_string( vf->create_string( "foo"));
    m_expression = ef->create_constant( m_value_string.get());
    MI_CHECK_EQUAL( -3, m_fc->set_argument( "param0", m_expression.get()));

    // check set_arguments()/access_arguments()

    m_value_int->set_value( -43);
    m_expression = ef->create_constant( m_value_int.get());
    mi::base::Handle<mi::neuraylib::IExpression_list> m_expression_list(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "param0", m_expression.get()));
    MI_CHECK_EQUAL( 0, m_fc->set_arguments( m_expression_list.get()));

    c_expression_list = m_fc->get_arguments();
    c_expression = c_expression_list->get_expression<mi::neuraylib::IExpression_constant>(
        "param0");
    c_value_int = c_expression->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( -43, c_value_int->get_value());

    // check error codes of set_arguments()

    MI_CHECK_EQUAL( -1, m_fc->set_arguments( 0));

    m_expression_list = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "param0", m_expression.get()));
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "invalid", m_expression.get()));
    MI_CHECK_EQUAL( -2, m_fc->set_arguments( m_expression_list.get()));

    m_expression = ef->create_constant( m_value_string.get());
    m_expression_list = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "param0", m_expression.get()));
    MI_CHECK_EQUAL( -3, m_fc->set_arguments( m_expression_list.get()));

    // check reset_argument()

    MI_CHECK_EQUAL( 0, m_fc->reset_argument( "param0"));

    c_expression_list = m_fc->get_arguments();
    c_expression = c_expression_list->get_expression<mi::neuraylib::IExpression_constant>( "param0");
    c_value_int = c_expression->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( 42, c_value_int->get_value());

    // check error codes of reset_argument()

    MI_CHECK_EQUAL( -1, m_fc->reset_argument( zero_string));
    MI_CHECK_EQUAL( -2, m_fc->reset_argument( "invalid"));
    MI_CHECK_EQUAL( -2, m_fc->reset_argument( 1));

    // check parameter references

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_parameter_references(float,float,float)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());
    m_fc = transaction->edit<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::fc_parameter_references");
    MI_CHECK( m_fc);


    mi::base::Handle<const mi::neuraylib::IExpression> toplevel_reference( defaults->get_expression( "b"));
    MI_CHECK_EQUAL( -6, m_fc->set_argument( "b", toplevel_reference.get()));
    MI_CHECK_EQUAL(  0, m_fc->reset_argument( "b"));

    mi::base::Handle<const mi::neuraylib::IExpression> nested_reference( defaults->get_expression( "c"));
    MI_CHECK_EQUAL( -6, m_fc->set_argument( "c", nested_reference.get()));
    MI_CHECK_EQUAL(  0, m_fc->reset_argument( "c"));

    {
        // check non-template-like function from builtins module
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::operator+(float,float)"));
        MI_CHECK_EQUAL_CSTR( "operator+(float,float)", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fd->get_parameter_count());

        mi::base::Handle<mi::neuraylib::IExpression_call> expr_x(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
        mi::base::Handle<mi::neuraylib::IValue_float> value_1( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expr_y(
            ef->create_constant( value_1.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "x", expr_x.get());
        arguments->add_expression( "y", expr_y.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "x", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "y", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fc->get_parameter_types());
        MI_CHECK_EQUAL( 2, parameter_types->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type( parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_type->get_kind());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type( parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter1_type->get_kind());

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator+(float,float)",
                /*argument_types*/ nullptr,
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::operator+(float,float)");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "operator+(float,float)");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator+(float,float)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator+(float,float)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check field access function

        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)",
                /*argument_types*/ nullptr,
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::" TEST_MDL);
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "foo_struct.param_int(::" TEST_MDL "::foo_struct)");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK_EQUAL( 1, argument_types->get_size());
        MI_CHECK_EQUAL_CSTR( "s", argument_types->get_name( zero_size));
        mi::base::Handle<const mi::neuraylib::IType_struct> arg0(
            argument_types->get_type<mi::neuraylib::IType_struct>( zero_size));
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", arg0->get_symbol());

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        MI_CHECK_EQUAL( 1, argument_types2->get_size());
        MI_CHECK_EQUAL_CSTR( "s", argument_types2->get_name( zero_size));
        mi::base::Handle<const mi::neuraylib::IType_struct> arg02(
            argument_types2->get_type<mi::neuraylib::IType_struct>( zero_size));
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", arg02->get_symbol());

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::" TEST_MDL);
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::" TEST_MDL);
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array constructor call
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::T[](...)"));
        MI_CHECK_EQUAL_CSTR( "T[](...)", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
        MI_CHECK_EQUAL( 0, fd->get_parameter_count());

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_0(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
        mi::base::Handle<mi::neuraylib::IValue_float> value_1( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_1(
            ef->create_constant( value_1.get()));

        // reversed order to check correct handling of named arguments (was positional arguments
        // some time ago)
        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "value1", expression_1.get());
        arguments->add_expression( "value0", expression_0.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "value0", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "value1", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fc->get_parameter_types());
        MI_CHECK_EQUAL( 2, parameter_types->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type( parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_type->get_kind());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type( parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter1_type->get_kind());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_arguments( fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression> arg0( c_arguments->get_expression( "value0"));
        MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_CALL);
        mi::base::Handle<const mi::neuraylib::IExpression> arg1( c_arguments->get_expression( "value1"));
        MI_CHECK_EQUAL( arg1->get_kind(), mi::neuraylib::IExpression::EK_CONSTANT);

        MI_CHECK_EQUAL( 0, fc->set_argument( zero_size, expression_1.get()));
        MI_CHECK_EQUAL( 0, fc->set_argument( 1, expression_0.get()));

        mi::base::Handle<mi::neuraylib::IValue_double> value_2( vf->create_double( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_2(
            ef->create_constant( value_2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( zero_size, expression_2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( 1, expression_2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::T[](...)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::T[](...)<float,2>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "T[](...)<float,2>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::T[](...)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::T[](...)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array length operator
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator_len(%3C0%3E[])"));
        MI_CHECK_EQUAL_CSTR(  "operator_len(%3C0%3E[])", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // really int
        MI_CHECK_EQUAL( 1, fd->get_parameter_count());
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fd->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type( parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_int_array"));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ALIAS, ret_type->get_kind());
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, ret_type->get_all_type_modifiers());
        ret_type = ret_type->skip_all_type_aliases();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind());
        MI_CHECK_EQUAL( 1, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        parameter_types = fc->get_parameter_types();
        parameter0_type = parameter_types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, parameter0_type->get_kind());

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a2(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_struct_array"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator_len(%3C0%3E[])",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::operator_len(%3C0%3E[])<int[N]>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "operator_len(%3C0%3E[])<int[N]>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator_len(%3C0%3E[])");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        // IType_factor::compare() does not work due to different array size symbols
        MI_CHECK_EQUAL( 1, argument_types->get_size());
        MI_CHECK_EQUAL_CSTR( "a", argument_types->get_name( zero_size));
        mi::base::Handle<const mi::neuraylib::IType_array> arg0(
            argument_types->get_type<mi::neuraylib::IType_array>( zero_size));
        MI_CHECK( !arg0->is_immediate_sized());
        mi::base::Handle<const mi::neuraylib::IType> arg0_element_type(
            arg0->get_element_type());
        MI_CHECK_EQUAL( arg0_element_type->get_kind(), mi::neuraylib::IType::TK_INT);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator_len(%3C0%3E[])");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        // IType_factor::compare() does not work due to different array size symbols
        MI_CHECK_EQUAL( 1, argument_types2->get_size());
        MI_CHECK_EQUAL_CSTR( "a", argument_types2->get_name( zero_size));
        mi::base::Handle<const mi::neuraylib::IType_array> arg02(
            argument_types2->get_type<mi::neuraylib::IType_array>( zero_size));
        MI_CHECK( !arg02->is_immediate_sized());
        mi::base::Handle<const mi::neuraylib::IType> arg0_element_type2(
            arg02->get_element_type());
        MI_CHECK_EQUAL( arg0_element_type2->get_kind(), mi::neuraylib::IType::TK_INT);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array index operator with deferred array
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator[](%3C0%3E[],int)"));
        MI_CHECK_EQUAL_CSTR(  "operator[](%3C0%3E[],int)", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
        MI_CHECK_EQUAL( 2, fd->get_parameter_count());
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fd->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type( parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type( parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_struct_array"));
        mi::base::Handle<mi::neuraylib::IValue_int> value_i( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i(
            ef->create_constant( value_i.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());
        arguments->add_expression( "i", expression_i.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_STRUCT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", fc->get_parameter_name( 1));
        parameter_types = fc->get_parameter_types();
        parameter0_type = parameter_types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, parameter0_type->get_kind());
        parameter1_type = parameter_types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a2(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_int_array"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> value_i2( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i2(
            ef->create_constant( value_i2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "i", expression_i2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator[](%3C0%3E[],int)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator[](%3C0%3E[],int)<::" TEST_MDL "::foo_struct[N]>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator[](%3C0%3E[],int)<::" TEST_MDL "::foo_struct[N]>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        // IType_factor::compare() does not work due to different array size symbols
        MI_CHECK_EQUAL( 2, argument_types->get_size());
        MI_CHECK_EQUAL_CSTR( "a", argument_types->get_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", argument_types->get_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_array> arg0(
            argument_types->get_type<mi::neuraylib::IType_array>( zero_size));
        MI_CHECK( !arg0->is_immediate_sized());
        mi::base::Handle<const mi::neuraylib::IType> arg0_element_type(
            arg0->get_element_type());
        MI_CHECK_EQUAL( arg0_element_type->get_kind(), mi::neuraylib::IType::TK_STRUCT);
        mi::base::Handle<const mi::neuraylib::IType> arg1( argument_types->get_type( 1));
        MI_CHECK_EQUAL( arg1->get_kind(), mi::neuraylib::IType::TK_INT);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        // IType_factor::compare() does not work due to different array size symbols
        MI_CHECK_EQUAL( 2, argument_types2->get_size());
        MI_CHECK_EQUAL_CSTR( "a", argument_types2->get_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", argument_types2->get_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_array> arg02(
            argument_types2->get_type<mi::neuraylib::IType_array>( zero_size));
        MI_CHECK( !arg02->is_immediate_sized());
        mi::base::Handle<const mi::neuraylib::IType> arg0_element_type2(
            arg02->get_element_type());
        MI_CHECK_EQUAL( arg0_element_type2->get_kind(), mi::neuraylib::IType::TK_STRUCT);
        mi::base::Handle<const mi::neuraylib::IType> arg12( argument_types2->get_type( 1));
        MI_CHECK_EQUAL( arg12->get_kind(), mi::neuraylib::IType::TK_INT);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array index operator with vector
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_float( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_float3(
            tf->create_vector( type_float.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a( vf->create_vector( type_float3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a(
            ef->create_constant( value_a.get()));
        mi::base::Handle<mi::neuraylib::IValue_int> value_i( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i(
            ef->create_constant( value_i.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());
        arguments->add_expression( "i", expression_i.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator[](%3C0%3E[],int)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IType> ret_type( fc->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType_vector> parameter0_type(
            parameter_types->get_type<mi::neuraylib::IType_vector>( zero_size));
        mi::base::Handle<const mi::neuraylib::IType> parameter0_elem_type(
            parameter0_type->get_element_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_elem_type->get_kind());
        MI_CHECK_EQUAL( 3, parameter0_type->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
            parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<const mi::neuraylib::IType_atomic> type_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_int3(
            tf->create_vector( type_int.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a2( vf->create_vector( type_int3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a2(
            ef->create_constant( value_a2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> value_i2( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i2(
            ef->create_constant( value_i2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "i", expression_i2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator[](%3C0%3E[],int)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator[](%3C0%3E[],int)<float3>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator[](%3C0%3E[],int)<float3>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array index operator with array
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_float( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_float3(
            tf->create_vector( type_float.get(), 3));
        mi::base::Handle<const mi::neuraylib::IType_matrix> type_matrix3x3(
            tf->create_matrix( type_float3.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a( vf->create_matrix( type_matrix3x3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a(
            ef->create_constant( value_a.get()));
        mi::base::Handle<mi::neuraylib::IValue_int> value_i( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i(
            ef->create_constant( value_i.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());
        arguments->add_expression( "i", expression_i.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator[](%3C0%3E[],int)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IType> ret_type( fc->get_return_type());
        MI_CHECK_EQUAL( 0, tf->compare( type_float3.get(), ret_type.get()));
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType_matrix> parameter0_type(
            parameter_types->get_type<mi::neuraylib::IType_matrix>( zero_size));
        mi::base::Handle<const mi::neuraylib::IType_vector> parameter0_vector_type(
            parameter0_type->get_element_type());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_elem_type(
            parameter0_vector_type->get_element_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_elem_type->get_kind());
        MI_CHECK_EQUAL( 3, parameter0_vector_type->get_size());
        MI_CHECK_EQUAL( 3, parameter0_type->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
            parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<const mi::neuraylib::IType_atomic> type_double( tf->create_double());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_double3(
            tf->create_vector( type_double.get(), 3));
        mi::base::Handle<const mi::neuraylib::IType_matrix> type_double3x3(
            tf->create_matrix( type_double3.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a2( vf->create_matrix( type_double3x3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a2(
            ef->create_constant( value_a2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> value_i2( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i2(
            ef->create_constant( value_i2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "i", expression_i2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator[](%3C0%3E[],int)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator[](%3C0%3E[],int)<float3x3>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator[](%3C0%3E[],int)<float3x3>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check cast operator
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator_cast(%3C0%3E)"));
        MI_CHECK_EQUAL_CSTR(  "operator_cast(%3C0%3E)", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
        MI_CHECK_EQUAL( 1, fd->get_parameter_count());
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fd->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type( parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum"));
        mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast_return(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum2"));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "cast", expression_cast.get());
        arguments->add_expression( "cast_return", expression_cast_return.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ENUM, ret_type->get_kind());
        MI_CHECK_EQUAL( 1, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "cast", fc->get_parameter_name( zero_size));
        parameter_types = fc->get_parameter_types();
        MI_CHECK_EQUAL( 1, parameter_types->get_size());
        parameter0_type = parameter_types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ENUM, parameter0_type->get_kind());

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast2(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum"));
        MI_CHECK_EQUAL( 0, fc->set_argument( "cast", expression_cast2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast3(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum2"));
        MI_CHECK_EQUAL( 0, fc->set_argument( "cast", expression_cast3.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast4(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "cast", expression_cast4.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> return_type(
            fc->get_return_type());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator_cast(%3C0%3E)",
                orig_argument_types.get(),
                return_type.get(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator_cast(%3C0%3E)<::" TEST_MDL "::Enum,::" TEST_MDL "::Enum2>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator_cast(%3C0%3E)<::" TEST_MDL "::Enum,::" TEST_MDL "::Enum2>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator_cast(%3C0%3E)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_tpes( dfn->get_argument_types());
        // IType_factor::compare() does not work since the parameter type list of the function call
        // has only one argument
        MI_CHECK_EQUAL( 2, argument_tpes->get_size());
        MI_CHECK_EQUAL_CSTR( "cast", argument_tpes->get_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "cast_return", argument_tpes->get_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_enum> arg0(
            argument_tpes->get_type<mi::neuraylib::IType_enum>( zero_size));
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", arg0->get_symbol());
        mi::base::Handle<const mi::neuraylib::IType_enum> arg1(
            argument_tpes->get_type<mi::neuraylib::IType_enum>( 1));
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum2", arg1->get_symbol());

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator_cast(%3C0%3E)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        // IType_factor::compare() does not work since the parameter type list of the function call
        // has only one argument
        MI_CHECK_EQUAL( 2, argument_types2->get_size());
        MI_CHECK_EQUAL_CSTR( "cast", argument_types2->get_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "cast_return", argument_types2->get_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_enum> arg02(
            argument_types2->get_type<mi::neuraylib::IType_enum>( zero_size));
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", arg02->get_symbol());
        mi::base::Handle<const mi::neuraylib::IType_enum> arg12(
            argument_types2->get_type<mi::neuraylib::IType_enum>( 1));
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum2", arg12->get_symbol());

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check ternary operator
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)"));
        MI_CHECK_EQUAL_CSTR(  "operator%3F(bool,%3C0%3E,%3C0%3E)", fd->get_mdl_name());
        MI_CHECK_EQUAL_CSTR(  "operator%3F", fd->get_mdl_simple_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
        MI_CHECK_EQUAL( 3, fd->get_parameter_count());
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types( fd->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type( parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_BOOL, parameter0_type->get_kind());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type( parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // dummy
        mi::base::Handle<const mi::neuraylib::IType> parameter2_type( parameter_types->get_type( 2));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter2_type->get_kind()); // dummy

        mi::base::Handle<mi::neuraylib::IValue_bool> value_cond( vf->create_bool( true));
        mi::base::Handle<mi::neuraylib::IValue_float> value_true( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IValue_float> value_false( vf->create_float( 43.0));

        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_cond(
            ef->create_constant( value_cond.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_true(
            ef->create_constant( value_true.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_false(
            ef->create_constant( value_false.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "cond", expression_cond.get());
        arguments->add_expression( "true_exp", expression_true.get());
        arguments->add_expression( "false_exp", expression_false.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 3, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "cond", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "true_exp", fc->get_parameter_name( 1));
        MI_CHECK_EQUAL_CSTR( "false_exp", fc->get_parameter_name( 2));
        parameter_types = fc->get_parameter_types();
        parameter0_type = parameter_types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_BOOL, parameter0_type->get_kind());
        parameter1_type = parameter_types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter1_type->get_kind());
        parameter2_type = parameter_types->get_type( 2);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter2_type->get_kind());

        value_cond->set_value( false);
        MI_CHECK_EQUAL( 0, fc->set_argument( "cond", expression_cond.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "cond", expression_true.get()));

        MI_CHECK_EQUAL( 0, fc->set_argument( "true_exp", expression_false.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "true_exp", expression_cond.get()));

        MI_CHECK_EQUAL( 0, fc->set_argument( "false_exp", expression_true.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "false_exp", expression_cond.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)<float>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator%3F(bool,%3C0%3E,%3C0%3E)<float>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
        MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check that deserialization loads modules if they have not yet been loaded

        // check that the module has not yet been loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl4"));
        MI_CHECK( !module);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, "mdl::test_mdl4::fd_1(int)", /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::test_mdl4::fd_1(int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
        MI_CHECK_EQUAL( 1, argument_types->get_size());
        MI_CHECK_EQUAL_CSTR( "param0", argument_types2->get_name( zero_size));
        mi::base::Handle<const mi::neuraylib::IType> arg0(
            argument_types2->get_type<mi::neuraylib::IType>( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, arg0->get_kind());

        // check that the module got loaded as side effect of deserialization
        module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl4");
        MI_CHECK( module);
    }
}

void check_imaterial_instance(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IValue> v_color( vf->create_color( 1.0f, 1.0f, 1.0f));
    mi::base::Handle<const mi::neuraylib::IType> t_color( v_color->get_type());
    mi::base::Handle<const mi::neuraylib::IType> t;

    // check a material instance with no arguments

    mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_0"));
    MI_CHECK( c_mi);
    MI_CHECK_EQUAL( c_mi->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);

    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
        c_mi->get_interface<mi::neuraylib::IFunction_call>());
    MI_CHECK( c_fc);
    MI_CHECK_EQUAL( c_fc->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);
    mi::base::Handle<const mi::neuraylib::IScene_element> c_se(
        c_mi->get_interface<mi::neuraylib::IScene_element>());
    MI_CHECK( c_se);
    MI_CHECK_EQUAL( c_se->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);

    c_fc = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_0");
    MI_CHECK( c_fc);
    MI_CHECK_EQUAL( c_fc->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_0()", c_mi->get_function_definition());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::md_0()", c_mi->get_mdl_function_definition());

    MI_CHECK_EQUAL( 0, c_mi->get_parameter_count());
    MI_CHECK( !c_mi->get_parameter_name( 0));

    MI_CHECK_EQUAL( minus_one_size, c_mi->get_parameter_index( "invalid"));
    MI_CHECK_EQUAL( minus_one_size, c_mi->get_parameter_index( 0));

    // check a material instance with one argument

    c_mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_1");
    MI_CHECK( c_mi);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", c_mi->get_function_definition());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::md_1(color)", c_mi->get_mdl_function_definition());

    MI_CHECK_EQUAL( 1, c_mi->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "tint", c_mi->get_parameter_name( 0));
    MI_CHECK( !c_mi->get_parameter_name( 1));

    MI_CHECK_EQUAL( 0, c_mi->get_parameter_index( "tint"));
    MI_CHECK_EQUAL( minus_one_size, c_mi->get_parameter_index( "invalid"));
    MI_CHECK_EQUAL( minus_one_size, c_mi->get_parameter_index( 0));

    // check serialized names

    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::" TEST_MDL "::md_1(color)",
            /*argument_types*/ nullptr,
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::" TEST_MDL "::md_1(color)");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "md_1(color)");

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        c_mi->get_parameter_types());

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::" TEST_MDL "::md_1(color)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
    MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::" TEST_MDL "::md_1(color)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
    MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::" TEST_MDL);
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module( transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK_EQUAL( result, 1);

    c_mi = 0;

    // check get_argument()/set_argument() (constants only)

    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_1"));
    MI_CHECK( m_mi);

    mi::base::Handle<mi::neuraylib::IValue_color> m_value_color(
        vf->create_color( 0.5f,  0.5f, 0.5f));
    mi::base::Handle<mi::neuraylib::IExpression_constant> m_expression(
        ef->create_constant( m_value_color.get()));
    MI_CHECK_EQUAL( 0, m_mi->set_argument( "tint", m_expression.get()));

    mi::base::Handle<const mi::neuraylib::IExpression_list> c_expression_list(
        m_mi->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_expression(
        c_expression_list->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
    mi::base::Handle<const mi::neuraylib::IValue_color> c_value_color(
        c_expression->get_value<mi::neuraylib::IValue_color>());
    MI_CHECK_EQUAL( 0, vf->compare( m_value_color.get(), c_value_color.get()));

    // check error codes of set_argument()

    MI_CHECK_EQUAL( -1, m_mi->set_argument( zero_string, m_expression.get()));
    MI_CHECK_EQUAL( -1, m_mi->set_argument( "tint", 0));
    MI_CHECK_EQUAL( -2, m_mi->set_argument( 1, m_expression.get()));
    MI_CHECK_EQUAL( -2, m_mi->set_argument( "invalid", m_expression.get()));
    mi::base::Handle<mi::neuraylib::IValue> m_value_string( vf->create_string( "foo"));
    m_expression = ef->create_constant( m_value_string.get());
    MI_CHECK_EQUAL( -3, m_mi->set_argument( "tint", m_expression.get()));

    // check set_arguments()/access_arguments()

    m_value_color = vf->create_color( 0.6f,  0.6f, 0.6f);
    m_expression = ef->create_constant( m_value_color.get());
    mi::base::Handle<mi::neuraylib::IExpression_list> m_expression_list(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "tint", m_expression.get()));
    MI_CHECK_EQUAL( 0, m_mi->set_arguments( m_expression_list.get()));

    c_expression_list = m_mi->get_arguments();
    c_expression = c_expression_list->get_expression<mi::neuraylib::IExpression_constant>( "tint");
    c_value_color = c_expression->get_value<mi::neuraylib::IValue_color>();
    MI_CHECK_EQUAL( 0, vf->compare( m_value_color.get(), c_value_color.get()));

    // check error codes of set_arguments()

    MI_CHECK_EQUAL( -1, m_mi->set_arguments( 0));

    m_expression_list = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "tint", m_expression.get()));
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "invalid", m_expression.get()));
    MI_CHECK_EQUAL( -2, m_mi->set_arguments( m_expression_list.get()));

    m_expression = ef->create_constant( m_value_string.get());
    m_expression_list = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "tint", m_expression.get()));
    MI_CHECK_EQUAL( -3, m_mi->set_arguments( m_expression_list.get()));

    // check reset_argument()

    MI_CHECK_EQUAL( 0, m_mi->reset_argument( "tint"));

    c_expression_list = m_mi->get_arguments();
    c_expression = c_expression_list->get_expression<mi::neuraylib::IExpression_constant>( "tint");
    c_value_color = c_expression->get_value<mi::neuraylib::IValue_color>();
    m_value_color = vf->create_color( 1.0f,  1.0f, 1.0f);
    MI_CHECK_EQUAL( 0, vf->compare( m_value_color.get(), c_value_color.get()));

    // check error codes of reset_argument()

    MI_CHECK_EQUAL( -1, m_mi->reset_argument( zero_string));
    MI_CHECK_EQUAL( -2, m_mi->reset_argument( "invalid"));
    MI_CHECK_EQUAL( -2, m_mi->reset_argument( 1));

    // change "tint" argument to a value different from the default (used by hash comparison in
    // class/instance compilation checks in check_icompiled_material())

    m_value_color = vf->create_color( 0.6f,  0.6f, 0.6f);
    m_expression = ef->create_constant( m_value_color.get());
    MI_CHECK_EQUAL( 0, m_mi->set_argument( "tint", m_expression.get()));

    // check uniform parameters vs auto return type on a varying function call

    mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_parameter_uniform(color)"));
    m_mi = md->create_function_call( 0, &result);
    MI_CHECK_EQUAL( -8, result);

    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( md->get_defaults());
    mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->clone( defaults.get()));
    m_mi = md->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -8, result);
}

mi::neuraylib::IAnnotation* create_string_annotation(
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    const char* value)
{
    mi::base::Handle<mi::neuraylib::IValue_string> arg_value( vf->create_string( value));
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_2_int_annotation(
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name0,
    const char* parameter_name1,
    mi::Sint32 value0,
    mi::Sint32 value1)
{
    mi::base::Handle<mi::neuraylib::IValue_int> arg_value0( vf->create_int( value0));
    mi::base::Handle<mi::neuraylib::IValue_int> arg_value1( vf->create_int( value1));
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr0(
        ef->create_constant( arg_value0.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr1(
        ef->create_constant( arg_value1.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name0, arg_expr0.get()));
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name1, arg_expr1.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_float2_annotation(
    mi::neuraylib::IType_factory* tf,
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    mi::Float32 value0,
    mi::Float32 value1)
{
    mi::base::Handle<const mi::neuraylib::IType_float> element_type( tf->create_float());
    mi::base::Handle<const mi::neuraylib::IType_vector> arg_type(
        tf->create_vector( element_type.get(), 2));
    mi::base::Handle<mi::neuraylib::IValue_vector> arg_value(
        vf->create_vector( arg_type.get()));
    mi::base::Handle<mi::neuraylib::IValue_float> arg_value0(
        arg_value->get_value<mi::neuraylib::IValue_float>( 0));
    arg_value0->set_value( value0);
    mi::base::Handle<mi::neuraylib::IValue_float> arg_value1(
        arg_value->get_value<mi::neuraylib::IValue_float>( 1));
    arg_value1->set_value( value1);

    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_Enum_annotation(
    mi::neuraylib::IType_factory* tf,
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    mi::Size index)
{
    mi::base::Handle<const mi::neuraylib::IType_enum> arg_type(
        tf->create_enum( "::" TEST_MDL "::Enum"));
    mi::base::Handle<mi::neuraylib::IValue_enum> arg_value(
        vf->create_enum( arg_type.get()));
    arg_value->set_index( index);
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_foo_struct_annotation(
    mi::neuraylib::IType_factory* tf,
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    int value0,
    float value1)
{
    mi::base::Handle<const mi::neuraylib::IType_struct> arg_type(
        tf->create_struct( "::" TEST_MDL "::foo_struct"));
    mi::base::Handle<mi::neuraylib::IValue_struct> arg_value(
        vf->create_struct( arg_type.get()));
    mi::base::Handle<mi::neuraylib::IValue_int> arg_value0(
        arg_value->get_value<mi::neuraylib::IValue_int>( 0));
    arg_value0->set_value( value0);
    mi::base::Handle<mi::neuraylib::IValue_float> arg_value1(
        arg_value->get_value<mi::neuraylib::IValue_float>( 1));
    arg_value1->set_value( value1);

    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

void check_variants(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create module with multiple variants with color/enum arguments and annotations

    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            transaction,
            "mdl::variants",
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));

    {
        // create md_1_white with different annotations
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation( create_string_annotation( vf.get(),
            ef.get(), "::anno::description(string)", "description", "variant description annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation.get()));

        result = module_builder->add_variant(
            "md_1_white",
            "mdl::" TEST_MDL "::md_1(color)",
            /*defaults*/ nullptr,
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 1, c_module->get_material_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::md_1_white(color)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_md->get_defaults());

        // check the default for tint of "variants::md_1_white(color)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_tint_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
        MI_CHECK( c_tint_expr);
        mi::base::Handle<const mi::neuraylib::IValue_color> c_tint_value(
            c_tint_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<mi::neuraylib::IValue_color> m_tint_value(
            vf->create_color( 1.0, 1.0, 1.0));
        MI_CHECK_EQUAL( 0, vf->compare( c_tint_value.get(), m_tint_value.get()));

        // check the annotation of "variants::md_1_white(color)"
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_md->get_annotations());
        MI_CHECK( c_annotations);
        MI_CHECK_EQUAL( 1, c_annotations->get_size());
        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation(
            c_annotations->get_annotation( 0));
        MI_CHECK( c_annotation);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", c_annotation->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_args(
            c_annotation->get_arguments());
        MI_CHECK( c_annotation_args);
        MI_CHECK_EQUAL( 1, c_annotation_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_arg(
            c_annotation_args->get_expression<mi::neuraylib::IExpression_constant>(
                "description"));
        MI_CHECK( c_annotation_arg);
        mi::base::Handle<const mi::neuraylib::IValue_string> c_annotation_value(
            c_annotation_arg->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK( c_annotation_value);
        MI_CHECK_EQUAL_CSTR( "variant description annotation", c_annotation_value->get_value());
    }

    // re-create module builder to test modifying modules already existing in the DB
    module_builder = mdl_factory->create_module_builder(
        transaction,
        "mdl::variants",
        mi::neuraylib::MDL_VERSION_1_0,
        mi::neuraylib::MDL_VERSION_LATEST,
        context.get());

    {
        // create md_enum_one with a new default for "param0" and new annotation block

        // create a "two" "param0" argument
        mi::base::Handle<const mi::neuraylib::IType_enum> param0_type(
            tf->create_enum( "::" TEST_MDL "::Enum"));
        mi::base::Handle<mi::neuraylib::IValue_enum> m_param0_value(
            vf->create_enum( param0_type.get(), 1));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
            ef->create_constant( m_param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "param0", m_param0_expr.get()));

        // create two ::anno::contributor annotations
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_a( create_string_annotation(
            vf.get(), ef.get(), "::anno::author(string)", "name", "variant author annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_b( create_string_annotation( vf.get(),
            ef.get(), "::anno::contributor(string)", "name", "variant contributor annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_a.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_b.get()));

        result = module_builder->add_variant(
            "md_enum_one",
            "mdl::" TEST_MDL "::md_enum(::" TEST_MDL "::Enum)",
            defaults.get(),
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 2, c_module->get_material_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::md_enum_one(::" TEST_MDL "::Enum)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_md->get_defaults());

        // check the default for param0 of "variants::md_enum_one(::" TEST_MDL "::Enum)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_param0_expr);
        mi::base::Handle<const mi::neuraylib::IValue_enum> c_param0_value(
            c_param0_expr->get_value<mi::neuraylib::IValue_enum>());
        MI_CHECK_EQUAL( 0, vf->compare( c_param0_value.get(), m_param0_value.get()));

        // check the annotations of "variants::md_enum_one(::" TEST_MDL "::Enum)"
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_md->get_annotations());
        MI_CHECK( c_annotations);
        MI_CHECK_EQUAL( 2, c_annotations->get_size());
        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_a(
            c_annotations->get_annotation( 0));
        MI_CHECK( c_annotation_a);
        MI_CHECK_EQUAL_CSTR( "::anno::author(string)", c_annotation_a->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_a_args(
            c_annotation_a->get_arguments());
        MI_CHECK( c_annotation_a_args);
        MI_CHECK_EQUAL( 1, c_annotation_a_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_a_arg(
            c_annotation_a_args->get_expression<mi::neuraylib::IExpression_constant>( "name"));
        MI_CHECK( c_annotation_a_arg);
        mi::base::Handle<const mi::neuraylib::IValue_string> c_annotation_a_value(
            c_annotation_a_arg->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK( c_annotation_a_value);
        MI_CHECK_EQUAL_CSTR( "variant author annotation", c_annotation_a_value->get_value());
        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_b(
            c_annotations->get_annotation( 1));
        MI_CHECK( c_annotation_b);
        MI_CHECK_EQUAL_CSTR( "::anno::contributor(string)", c_annotation_b->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_b_args(
            c_annotation_b->get_arguments());
        MI_CHECK( c_annotation_b_args);
        MI_CHECK_EQUAL( 1, c_annotation_b_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_b_arg(
            c_annotation_b_args->get_expression<mi::neuraylib::IExpression_constant>( "name"));
        MI_CHECK( c_annotation_b_arg);
        mi::base::Handle<const mi::neuraylib::IValue_string> c_annotation_b_value(
            c_annotation_b_arg->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK( c_annotation_b_value);
        MI_CHECK_EQUAL_CSTR( "variant contributor annotation", c_annotation_b_value->get_value());
    }
    {
        // create fd_1_43 with a new default for "param0" and new annotations

        // create a 43 "param0" argument
        mi::base::Handle<mi::neuraylib::IValue_int> m_variant2_param0_value( vf->create_int( 43));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_variant2_param0_expr(
            ef->create_constant( m_variant2_param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults( ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "param0", m_variant2_param0_expr.get()));

        // create a ::" TEST_MDL "::anno_2_int, a ::" TEST_MDL "::anno_float2, a ::" TEST_MDL "::anno_Enum, and
        // a ::" TEST_MDL "::anno_foo_struct annotation
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_a( create_2_int_annotation(
            vf.get(), ef.get(), "::" TEST_MDL "::anno_2_int(int,int)", "max", "min", 20, 10));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_b( create_float2_annotation(
            tf.get(), vf.get(), ef.get(), "::" TEST_MDL "::anno_float2(float2)",
            "param0", 42.0f, 43.0f));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_c( create_Enum_annotation(
            tf.get(), vf.get(), ef.get(), "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)",
            "param0", 1));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_d( create_foo_struct_annotation(
            tf.get(), vf.get(), ef.get(), "::" TEST_MDL "::anno_struct(::" TEST_MDL "::foo_struct)",
            "param0", 42, 43.0f));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_a.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_b.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_c.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_d.get()));

        {
            // check annotation wrapper
            mi::neuraylib::Annotation_wrapper anno_wrapper( m_annotations.get());
            MI_CHECK_EQUAL( 4, anno_wrapper.get_annotation_count());
            MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_2_int(int,int)", anno_wrapper.get_annotation_name( 0));
            MI_CHECK_EQUAL( 2, anno_wrapper.get_annotation_param_count( 0));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_INT,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 0, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "max", anno_wrapper.get_annotation_param_name( 0, 0));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_INT,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 0, 1))->get_kind());
            MI_CHECK_EQUAL_CSTR( "min", anno_wrapper.get_annotation_param_name( 0, 1));

            mi::Sint32 test_value = 0;
            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value<mi::Sint32>( 0, 0, test_value));
            MI_CHECK_EQUAL( 20, test_value);
            test_value = 0;
            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value_by_name<mi::Sint32>(
                "::" TEST_MDL "::anno_2_int(int,int)", 0, test_value ) );
            MI_CHECK_EQUAL( 20, test_value );

            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value<mi::Sint32>( 0, 1, test_value));
            MI_CHECK_EQUAL( 10, test_value);
            test_value = 0;
            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value_by_name<mi::Sint32>(
                "::" TEST_MDL "::anno_2_int(int,int)", 1, test_value ) );
            MI_CHECK_EQUAL( 10, test_value );

            MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_float2(float2)", anno_wrapper.get_annotation_name( 1));
            MI_CHECK_EQUAL( 1, anno_wrapper.get_annotation_param_count( 1));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_VECTOR,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 1, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "param0", anno_wrapper.get_annotation_param_name( 1, 0));

            mi::Size test_index = anno_wrapper.get_annotation_index(
                "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)" );
            MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)",
                anno_wrapper.get_annotation_name( mi::Size(test_index) ));
            MI_CHECK_EQUAL( 1, anno_wrapper.get_annotation_param_count( 2));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_ENUM,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 2, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "param0", anno_wrapper.get_annotation_param_name( 2, 0));

            MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_struct(::" TEST_MDL "::foo_struct)",
                anno_wrapper.get_annotation_name( 3));
            MI_CHECK_EQUAL( 1, anno_wrapper.get_annotation_param_count( 3));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_STRUCT,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 3, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "param0", anno_wrapper.get_annotation_param_name( 3, 0));

            // test behavior of invalid usage of the annotation wrapper (quite, no crash)
            mi::Float32 test_value_float = 0.0f;
            mi::neuraylib::Annotation_wrapper anno_wrapper_invalid( nullptr);
            MI_CHECK_EQUAL( 0, anno_wrapper_invalid.get_annotation_count());
            MI_CHECK_EQUAL( nullptr, anno_wrapper_invalid.get_annotation_name( 0));
            MI_CHECK_EQUAL( static_cast<mi::Size>(-1), anno_wrapper_invalid.get_annotation_index(
                "::anno::foo(int)"));
            MI_CHECK_EQUAL( 0, anno_wrapper_invalid.get_annotation_param_count( 0));
            MI_CHECK_EQUAL( nullptr, anno_wrapper_invalid.get_annotation_param_name( 0, 0));
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IType>(
                anno_wrapper_invalid.get_annotation_param_type( 0, 0)).get());
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IValue>(
                anno_wrapper_invalid.get_annotation_param_value( 0, 0)).get());
            MI_CHECK_EQUAL( -3, anno_wrapper_invalid.get_annotation_param_value<mi::Float32>(
                0, 0, test_value_float));

            MI_CHECK_EQUAL( nullptr, anno_wrapper.get_annotation_param_name( 0, 2));
            MI_CHECK_EQUAL(static_cast<mi::Size>(-1), anno_wrapper.get_annotation_index(
                "::anno::foo(int)"));
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IType>(
                anno_wrapper.get_annotation_param_type( 0, 2)).get());
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IValue>(
                anno_wrapper.get_annotation_param_value( 0, 2)).get());
            MI_CHECK_EQUAL( -3, anno_wrapper.get_annotation_param_value<mi::Float32>(
                0, 2, test_value_float));
            MI_CHECK_EQUAL( -1, anno_wrapper.get_annotation_param_value<mi::Float32>(
                0, 1, test_value_float ) );
            MI_CHECK_EQUAL( -3, anno_wrapper.get_annotation_param_value_by_name<mi::Float32>(
                "::anno::foo(int)", 1, test_value_float ) );
            MI_CHECK_EQUAL( -1, anno_wrapper.get_annotation_param_value_by_name<mi::Float32>(
                "::" TEST_MDL "::anno_2_int(int,int)", 1, test_value_float ) );
        }

        result = module_builder->add_variant(
            "fd_1_43",
            "mdl::" TEST_MDL "::fd_1(int)",
            defaults.get(),
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 1, c_module->get_function_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::fd_1_43(int)"));
        MI_CHECK( c_fd);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_fd->get_defaults());

        // check the default for param0 of "variants::fd_1_43"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_param0_expr);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_param0_value(
            c_param0_expr->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK_EQUAL( 0, vf->compare( c_param0_value.get(), m_variant2_param0_value.get()));

        // check the annotations of "variants::fd_1_43"
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_fd->get_annotations());
        MI_CHECK( c_annotations);
        MI_CHECK_EQUAL( 4, c_annotations->get_size());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_a(
            c_annotations->get_annotation( 0));
        MI_CHECK( c_annotation_a);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_2_int(int,int)", c_annotation_a->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_a_args(
            c_annotation_a->get_arguments());
        MI_CHECK( c_annotation_a_args);
        MI_CHECK_EQUAL( 2, c_annotation_a_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_a_arg(
            c_annotation_a_args->get_expression<mi::neuraylib::IExpression_constant>( "min"));
        MI_CHECK( c_annotation_a_arg);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_annotation_a_value(
            c_annotation_a_arg->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK( c_annotation_a_value);
        MI_CHECK_EQUAL( 10, c_annotation_a_value->get_value());
        c_annotation_a_arg
            = c_annotation_a_args->get_expression<mi::neuraylib::IExpression_constant>( "max");
        MI_CHECK( c_annotation_a_arg);
        c_annotation_a_value
            = c_annotation_a_arg->get_value<mi::neuraylib::IValue_int>();
        MI_CHECK( c_annotation_a_value);
        MI_CHECK_EQUAL( 20, c_annotation_a_value->get_value());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_b(
            c_annotations->get_annotation( 1));
        MI_CHECK( c_annotation_b);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_float2(float2)", c_annotation_b->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_b_args(
            c_annotation_b->get_arguments());
        MI_CHECK( c_annotation_b_args);
        MI_CHECK_EQUAL( 1, c_annotation_b_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_b_arg(
            c_annotation_b_args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_annotation_b_arg);
        mi::base::Handle<const mi::neuraylib::IValue_vector> c_annotation_b_value(
            c_annotation_b_arg->get_value<mi::neuraylib::IValue_vector>());
        MI_CHECK( c_annotation_b_value);
        mi::base::Handle<const mi::neuraylib::IValue_float> c_annotation_b_value_element(
            c_annotation_b_value->get_value<mi::neuraylib::IValue_float>( 0));
        MI_CHECK_EQUAL( 42.0f, c_annotation_b_value_element->get_value());
        c_annotation_b_value_element
            = c_annotation_b_value->get_value<mi::neuraylib::IValue_float>( 1);
        MI_CHECK_EQUAL( 43.0f, c_annotation_b_value_element->get_value());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_c(
            c_annotations->get_annotation( 2));
        MI_CHECK( c_annotation_c);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)", c_annotation_c->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_c_args(
            c_annotation_c->get_arguments());
        MI_CHECK( c_annotation_c_args);
        MI_CHECK_EQUAL( 1, c_annotation_c_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_c_arg(
            c_annotation_c_args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_annotation_c_arg);
        mi::base::Handle<const mi::neuraylib::IValue_enum> c_annotation_c_value(
            c_annotation_c_arg->get_value<mi::neuraylib::IValue_enum>());
        MI_CHECK( c_annotation_c_value);
        MI_CHECK_EQUAL( 1, c_annotation_c_value->get_index());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_d(
            c_annotations->get_annotation( 3));
        MI_CHECK( c_annotation_d);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_struct(::" TEST_MDL "::foo_struct)", c_annotation_d->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_d_args(
            c_annotation_d->get_arguments());
        MI_CHECK( c_annotation_d_args);
        MI_CHECK_EQUAL( 1, c_annotation_d_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_d_arg(
            c_annotation_d_args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_annotation_d_arg);
        mi::base::Handle<const mi::neuraylib::IValue_struct> c_annotation_d_value(
            c_annotation_d_arg->get_value<mi::neuraylib::IValue_struct>());
        MI_CHECK( c_annotation_d_value);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_annotation_d_value0(
            c_annotation_d_value->get_value<mi::neuraylib::IValue_int>( 0));
        MI_CHECK_EQUAL( 42, c_annotation_d_value0->get_value());
        mi::base::Handle<const mi::neuraylib::IValue_float> c_annotation_d_value1(
            c_annotation_d_value->get_value<mi::neuraylib::IValue_float>( 1));
        MI_CHECK_EQUAL( 43.0f, c_annotation_d_value1->get_value());
    }
    {
        // create fd_1_42_plus_42 with a new default for "param0"

        // create a 42+42 "param0" argument
        mi::base::Handle<mi::neuraylib::IValue_int> m_variant_lhs_value( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_variant_lhs_expr(
            ef->create_constant( m_variant_lhs_value.get()));
        mi::base::Handle<mi::neuraylib::IValue_int> m_variant_rhs_value( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_variant_rhs_expr(
            ef->create_constant( m_variant_rhs_value.get()));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd_variant_plus(
            transaction->access<mi::neuraylib::IFunction_definition>( "mdl::operator+(int,int)"));
        const char* lhs_name = fd_variant_plus->get_parameter_name( static_cast<mi::Size>( 0));
        const char* rhs_name = fd_variant_plus->get_parameter_name( 1);
        mi::base::Handle<mi::neuraylib::IExpression_list> m_variant_plus_args(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, m_variant_plus_args->add_expression( lhs_name, m_variant_lhs_expr.get()));
        MI_CHECK_EQUAL( 0, m_variant_plus_args->add_expression( rhs_name, m_variant_lhs_expr.get()));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc_variant_plus(
            fd_variant_plus->create_function_call( m_variant_plus_args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc_variant_plus.get(), "variant_plus"));
        mi::base::Handle<mi::neuraylib::IExpression_call> m_variant_param0_expr(
            ef->create_call( "variant_plus"));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults( ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "param0", m_variant_param0_expr.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());

        result = module_builder->add_variant(
            "fd_1_42_plus_42",
            "mdl::" TEST_MDL "::fd_1(int)",
            defaults.get(),
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 2, c_module->get_function_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::fd_1_42_plus_42(int)"));
        MI_CHECK( c_fd);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_fd->get_defaults());

#if 1
        // check the default for param0 of "variants::fd_1_42_plus_42" (constant folding)
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_param0_expr);

        mi::base::Handle<const mi::neuraylib::IValue_int> c_param0_value(
            c_param0_expr->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK_EQUAL( 84, c_param0_value->get_value());
#else
        // check the default for param0 of "variants::fd_1_42_plus_42" (no constant
        // folding)
        mi::base::Handle<const mi::neuraylib::IExpression_call> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
        MI_CHECK( c_param0_expr);

        const char* fc_name = c_param0_expr->get_call();
        mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc_plus(
           transaction->access<mi::neuraylib::IFunction_call>( fc_name));
        const char* fd_name = c_fc_plus->get_function_definition();
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd_plus(
           transaction->access<mi::neuraylib::IFunction_definition>( fd_name));
        MI_CHECK_EQUAL( mi::neuraylib::IFunction_definition::DS_PLUS, c_fd_plus->get_semantic());

        mi::base::Handle<const mi::neuraylib::IExpression_list> c_fc_plus_args(
            c_fc_plus->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_fc_plus_lhs_expr(
            c_fc_plus_args->get_expression<mi::neuraylib::IExpression_constant>( zero_size));
        MI_CHECK( c_fc_plus_lhs_expr);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_fc_plus_lhs_value(
            c_fc_plus_lhs_expr->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK_EQUAL( 0, vf->compare( c_fc_plus_lhs_value.get(), m_variant_lhs_value.get()));
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_fc_plus_rhs_expr(
            c_fc_plus_args->get_expression<mi::neuraylib::IExpression_constant>( 1));
        MI_CHECK( c_fc_plus_rhs_expr);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_fc_plus_rhs_value(
            c_fc_plus_rhs_expr->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK_EQUAL( 0, vf->compare( c_fc_plus_rhs_value.get(), m_variant_rhs_value.get()));
#endif

        // check that the annotation of "" TEST_MDL "::fd_1" is not copied
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_fd->get_annotations());
        MI_CHECK( !c_annotations);
    }

    std::string prototype_name = "mdl::" TEST_MDL "::md_resource_sharing(texture_2d,texture_2d)";

    {
        // create md_resource_sharing_new_defaults with a new default for "tex0"
        // (using a resource with MDL file path works)

        // create a "tex0" argument
        mi::base::Handle<mi::neuraylib::IValue> tex0_value(
            mdl_factory->create_texture(
                transaction,
                "/mdl_elements/resources/test.png",
                mi::neuraylib::IType_texture::TS_2D,
                2.2f,
                /*selector*/ nullptr,
                /*shared*/ false,
                context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( tex0_value);
        mi::base::Handle<mi::neuraylib::IExpression_constant> tex0_expr(
            ef->create_constant( tex0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "tex0", tex0_expr.get()));

        result = module_builder->add_variant(
            "md_resource_sharing_new_defaults",
            prototype_name.c_str(),
            defaults.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // create md_resource_sharing_new_defaults2 with a new default for "tex0"
        // (using an invalid resource works)

        // create a "tex0" argument
        mi::base::Handle<const mi::neuraylib::IType_texture> tex0_type(
            tf->create_texture( mi::neuraylib::IType_texture::TS_2D));
        mi::base::Handle<mi::neuraylib::IValue_texture> tex0_value(
            vf->create_texture( tex0_type.get(), 0));
        MI_CHECK( tex0_value);
        mi::base::Handle<mi::neuraylib::IExpression_constant> tex0_expr(
            ef->create_constant( tex0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "tex0", tex0_expr.get()));

        result = module_builder->add_variant(
            "md_resource_sharing_new_defaults2",
            prototype_name.c_str(),
            defaults.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // create md_resource_sharing_new_defaults3 with a new default for "tex0"
        // (using a resource without MDL file path, but filename inside the search paths work also)

        // create dummy image/texture
        mi::base::Handle<mi::neuraylib::IImage> image(
            transaction->create<mi::neuraylib::IImage>( "Image"));
        result = transaction->store( image.get(), "dummy_image");
        MI_CHECK_EQUAL( result, 0);
        mi::base::Handle<mi::neuraylib::ITexture> texture(
            transaction->create<mi::neuraylib::ITexture>( "Texture"));
        texture->set_image( "dummy_image");
        result = transaction->store( texture.get(), "dummy_texture");
        MI_CHECK_EQUAL( result, 0);

        // create a "tex0" argument
        mi::base::Handle<const mi::neuraylib::IType_texture> tex0_type(
            tf->create_texture( mi::neuraylib::IType_texture::TS_2D));
        mi::base::Handle<mi::neuraylib::IValue_texture> tex0_value(
            vf->create_texture( tex0_type.get(), "dummy_texture"));
        MI_CHECK( tex0_value);
        mi::base::Handle<mi::neuraylib::IExpression_constant> tex0_expr(
            ef->create_constant( tex0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "tex0", tex0_expr.get()));

        result = module_builder->add_variant(
            "md_resource_sharing_new_defaults3",
            prototype_name.c_str(),
            defaults.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        // remove again to support import/export later (does not work with memory-based image above)
        result = module_builder->remove_entity(
            "md_resource_sharing_new_defaults3",
            0,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
}

void check_variants2(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // Create module with multiple variants with resource parameters (same defaults as before).
    //
    // As variants the AST representation will contain tags that are visible to the exporter. The
    // definitions from ::" TEST_MDL " use resources from files, the definitions from ::test_archives
    // use resources from an archive.

    const char* variants[] = {
        "md_texture",
        "md_light_profile",
        "md_bsdf_measurement",
        "fd_in_archive_texture_weak_relative",
        "fd_in_archive_texture_strict_relative",
        "fd_in_archive_texture_absolute",
    };

    const char* prototypes[] = {
        "mdl::" TEST_MDL "::md_texture(texture_2d)",
        "mdl::" TEST_MDL "::md_light_profile(light_profile)",
        "mdl::" TEST_MDL "::md_bsdf_measurement(bsdf_measurement)",
        "mdl::test_archives::fd_in_archive_texture_weak_relative(texture_2d)",
        "mdl::test_archives::fd_in_archive_texture_strict_relative(texture_2d)",
        "mdl::test_archives::fd_in_archive_texture_absolute(texture_2d)"
    };

    mi::Size n =       sizeof( prototypes) / sizeof( prototypes[0]);
    MI_CHECK_EQUAL( n, sizeof( variants  ) / sizeof( variants[0]  ));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            transaction,
            "mdl::resources",
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));

    for( mi::Size i = 0; i < n; ++i) {
        result = module_builder->add_variant(
            variants[i],
            prototypes[i],
            /*defaults*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( "mdl::resources"));
    MI_CHECK( module);
    MI_CHECK_EQUAL( 5, module->get_import_count());
    MI_CHECK_EQUAL( 3, module->get_material_count());
    MI_CHECK_EQUAL( 3, module->get_function_count());
}

void check_materials(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create module "::new_materials" with material "md_wrap", based on "::" TEST_MDL "::md_wrap(float,color,color)",
        // with new parameters "sqrt_x", "r", and "g".

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> sqrt_x_type( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType> r_type( tf->create_color());
        mi::base::Handle<const mi::neuraylib::IType> g_type( tf->create_color());
        g_type = tf->create_alias( g_type.get(), mi::neuraylib::IType::MK_UNIFORM, /*symbol*/ nullptr);
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "sqrt_x", sqrt_x_type.get());
        parameters->add_type( "r", r_type.get());
        parameters->add_type( "g", g_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> x_x_expr(
            ef->create_parameter( sqrt_x_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> x_args(
            ef->create_expression_list());
        x_args->add_expression( "x", x_x_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> x_direct_call(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_x(float)", x_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression> rhs_r_expr(
            ef->create_parameter( r_type.get(), 1));
        mi::base::Handle<mi::neuraylib::IExpression> rhs_g_expr(
            ef->create_parameter( g_type.get(), 2));
        mi::base::Handle<mi::neuraylib::IExpression_list> rhs_args(
            ef->create_expression_list());
        rhs_args->add_expression( "r", rhs_r_expr.get());
        rhs_args->add_expression( "g", rhs_g_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> rhs_direct_call(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_rhs(color,color,color)", rhs_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "x", x_direct_call.get());
        body_args->add_expression( "rhs", rhs_direct_call.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call(
                "mdl::" TEST_MDL "::md_wrap(float,color,color)", body_args.get()));

        // create defaults

        mi::base::Handle<mi::neuraylib::IValue> sqrt_x_value(
            vf->create_float( 0.70710678f));
        mi::base::Handle<mi::neuraylib::IExpression> sqrt_x_expr(
           ef->create_constant( sqrt_x_value.get()));

        mi::base::Handle<mi::neuraylib::IValue_color> c_value(
            vf->create_color( 0, 1, 1));
        mi::base::Handle<mi::neuraylib::IExpression> c_expr(
            ef->create_constant( c_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> r_args(
            ef->create_expression_list());
        r_args->add_expression( "c", c_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> r_expr(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_r(color)", r_args.get()));

        mi::base::Handle<mi::neuraylib::IValue_color> g_value(
            vf->create_color( 0, 1, 0));
        mi::base::Handle<mi::neuraylib::IExpression> g_expr(
            ef->create_constant( g_value.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        defaults->add_expression( "sqrt_x", sqrt_x_expr.get());
        defaults->add_expression( "r", r_expr.get());
        defaults->add_expression( "g", g_expr.get());

        // create parameter annotations

        mi::base::Handle<mi::neuraylib::IAnnotation> sqrt_x_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "sqrt(x.x)"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> sqrt_x_anno_block(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, sqrt_x_anno_block->add_annotation( sqrt_x_annotation.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> r_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "-rhs.r"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> r_anno_block(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, r_anno_block->add_annotation( r_annotation.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> g_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "rhs.g"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> g_anno_block(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, g_anno_block->add_annotation( g_annotation.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation_list> parameter_annotations(
            ef->create_annotation_list());
        parameter_annotations->add_annotation_block( "sqrt_x", sqrt_x_anno_block.get());
        parameter_annotations->add_annotation_block( "r", r_anno_block.get());
        parameter_annotations->add_annotation_block( "g", g_anno_block.get());

        // add the function

        result = module_builder->add_function(
            "md_wrap",
            body.get(),
            parameters.get(),
            defaults.get(),
            parameter_annotations.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // check created module

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::new_materials"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 4, c_module->get_import_count());
        MI_CHECK_EQUAL( 1, c_module->get_material_count());
        MI_CHECK_EQUAL( 0, c_module->get_function_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::new_materials::md_wrap(float,color,color)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IType_list> c_types( c_md->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_md->get_defaults());

        // check the default for sqrt_x of "new_materials::md_wrap(float,color,color)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_sqrt_x_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "sqrt_x"));
        MI_CHECK( c_sqrt_x_expr);
        mi::base::Handle<const mi::neuraylib::IValue_float> c_sqrt_x_value(
            c_sqrt_x_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_CLOSE( c_sqrt_x_value->get_value(), 0.7071068f, 1e-6);

        mi::base::Handle<const mi::neuraylib::IType> c_sqrt_x_type(
            c_types->get_type( "sqrt_x"));
        MI_CHECK( (c_sqrt_x_type->get_all_type_modifiers() & mi::neuraylib::IType::MK_UNIFORM) == 0);

        // check the default for r of "new_materials::md_wrap(float,color,color)"
        mi::base::Handle<const mi::neuraylib::IExpression_call> c_r_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_call>( "r"));
        MI_CHECK( c_r_expr);
        mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
            transaction->access<mi::neuraylib::IFunction_call>( c_r_expr->get_call()));
        MI_CHECK_EQUAL_CSTR( c_fc->get_function_definition(), "mdl::" TEST_MDL "::fd_wrap_r(color)");

        mi::base::Handle<const mi::neuraylib::IType> c_r_expr_type(
            c_types->get_type( "r"));
        MI_CHECK( (c_r_expr_type->get_all_type_modifiers() & mi::neuraylib::IType::MK_UNIFORM) == 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> c_fc_args( c_fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_fc_args_c_expr(
            c_fc_args->get_expression<mi::neuraylib::IExpression_constant>( "c"));
        mi::base::Handle<const mi::neuraylib::IValue_color> c_fc_args_c_value(
            c_fc_args_c_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<mi::neuraylib::IValue_color> m_fc_args_c_value(
            vf->create_color( 0, 1, 1));
        MI_CHECK_EQUAL( 0, vf->compare( c_fc_args_c_value.get(), m_fc_args_c_value.get()));

        // check the default for g of "new_materials::md_wrap(float,color,color)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_g_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "g"));
        MI_CHECK( c_g_expr);
        mi::base::Handle<const mi::neuraylib::IValue_color> c_g_value(
            c_g_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<mi::neuraylib::IValue_color> m_g_value(
            vf->create_color( 0, 1, 0));
        MI_CHECK_EQUAL( 0, vf->compare( c_g_value.get(), m_g_value.get()));

        mi::base::Handle<const mi::neuraylib::IType> c_g_type(
            c_types->get_type( "g"));
        MI_CHECK( (c_g_type->get_all_type_modifiers() & mi::neuraylib::IType::MK_UNIFORM) != 0);
    }
    {
        // create module "mdl::new_operators" with function "my_new_float_add", based on
        // "operator+(float,float)", and adds fd_wrap_x() of the new parameter "new_x" and
        // the constant 2.0

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_operators",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> new_x_type( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "new_x", new_x_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> x_x_expr(
            ef->create_parameter( new_x_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> x_args(
            ef->create_expression_list());
        x_args->add_expression( "x", x_x_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> x_direct_call(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_x(float)", x_args.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> y_value(
            vf->create_float( 2.0));
        mi::base::Handle<mi::neuraylib::IExpression> y_expr(
            ef->create_constant( y_value.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> op_plus_args(
            ef->create_expression_list());
        op_plus_args->add_expression( "x", x_direct_call.get());
        op_plus_args->add_expression( "y", y_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( "mdl::operator+(float,float)", op_plus_args.get()));

        // add the function

        result = module_builder->add_function(
            "my_new_float_add",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add function
        //
        // export float fd_pass_parameter(float x) varying { return x; }
        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> x_type( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "x", x_type.get());

        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_parameter( x_type.get(), 0));

        result = module_builder->add_function(
            "fd_pass_parameter",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add material
        //
        // export material md_pass_parameter(material x) = x;
        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> x_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "x", x_type.get());

        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_parameter( x_type.get(), 0));

        result = module_builder->add_function(
            "md_pass_parameter",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add function
        //
        // export float fd_return_constant() uniform { return 42.f; }

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "fd_return_constant",
            body.get(),
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_UNIFORM,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add material
        //
        // export material md_return_constant() uniform { return material(); }

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType_struct> body_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_struct( body_type.get()));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "md_return_constant",
            body.get(),
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // Edit module "mdl::new_materials" and add AGAIN material
        //
        // export material md_return_constant() uniform { return material(); }
        //
        // This will fail in analyze_module(). Test that the module builder instance recovers from
        // this error and add annotation
        //
        // export annotation ad_dummy(int x);

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType_struct> body_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_struct( body_type.get()));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "md_return_constant",
            body.get(),
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_NOT_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IType> x_type( tf->create_string());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "x", x_type.get());

        result = module_builder->add_annotation(
            "ad_dummy",
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add enum
        //
        // export enum Enum [[ description("enum annotation") ]]
        // {
        //     one = 1 [[ description("one annotation") ]],
        //     two = 2 [[ ad_dummy("test_ad_dummy") ]]
        // };

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IValue> one_value( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression> one_expr( ef->create_constant( one_value.get()));
        mi::base::Handle<mi::neuraylib::IValue> two_value( vf->create_int( 2));
        mi::base::Handle<mi::neuraylib::IExpression> two_expr( ef->create_constant( two_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> enumerators( ef->create_expression_list());
        enumerators->add_expression( "one", one_expr.get());
        enumerators->add_expression( "two", two_expr.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> one_annotation( create_string_annotation( vf.get(),
            ef.get(), "::anno::description(string)", "description", "one annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> one_block(
            ef->create_annotation_block());
        one_block->add_annotation( one_annotation.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> two_annotation( create_string_annotation( vf.get(),
            ef.get(), "::new_materials::ad_dummy(string)", "x", "test ad_dummy"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> two_block(
            ef->create_annotation_block());
        two_block->add_annotation( two_annotation.get());

        mi::base::Handle<mi::neuraylib::IAnnotation_list> enumerator_annotations(
            ef->create_annotation_list());
        enumerator_annotations->add_annotation_block( "one", one_block.get());
        enumerator_annotations->add_annotation_block( "two", two_block.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> enum_annotation( create_string_annotation( vf.get(),
            ef.get(), "::anno::description(string)", "description", "enum annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> annotations(
            ef->create_annotation_block());
        annotations->add_annotation( enum_annotation.get());

        result = module_builder->add_enum_type(
            "Enum",
            enumerators.get(),
            enumerator_annotations.get(),
            annotations.get(),
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add struct
        //
        // export struct foo_struct [[ description("struct annotation") ]] {
        //     int param_int [[ description("param_int annotation") ]];
        //     float param_float = 0.0;
        // };

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> param_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType> param_float( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> fields( tf->create_type_list());
        fields->add_type( "param_int", param_int.get());
        fields->add_type( "param_float", param_float.get());

        mi::base::Handle<mi::neuraylib::IValue> param_float_default_value( vf->create_float( 0.0f));
        mi::base::Handle<mi::neuraylib::IExpression> param_float_default_expr( ef->create_constant( param_float_default_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> field_defaults( ef->create_expression_list());
        field_defaults->add_expression( "param_float", param_float_default_expr.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> param_int_annotation( create_string_annotation( vf.get(),
            ef.get(), "::anno::description(string)", "description", "param_int annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> param_int_block(
            ef->create_annotation_block());
        param_int_block->add_annotation( param_int_annotation.get());
        mi::base::Handle<mi::neuraylib::IAnnotation_list> field_annotations(
            ef->create_annotation_list());
        field_annotations->add_annotation_block( "param_int", param_int_block.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> struct_annotation( create_string_annotation( vf.get(),
            ef.get(), "::anno::description(string)", "description", "struct annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> annotations(
            ef->create_annotation_block());
        annotations->add_annotation( struct_annotation.get());

        result = module_builder->add_struct_type(
            "foo_struct",
            fields.get(),
            field_defaults.get(),
            field_annotations.get(),
            annotations.get(),
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add constant (invalid name!)
        //
        // export float roughly two pi = 2 * 3.14;
        //
        // This will fail. Test that the module builder instance recovers from
        // this error and add constant
        //
        // export float roughly_two_pi = 2 * 3.14;

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IValue> two( vf->create_float( 2.0f));
        mi::base::Handle<mi::neuraylib::IExpression> two_expr( ef->create_constant( two.get()));
        mi::base::Handle<mi::neuraylib::IValue> pi( vf->create_float( 3.14f));
        mi::base::Handle<mi::neuraylib::IExpression> pi_expr( ef->create_constant( pi.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "x", two_expr.get());
        args->add_expression( "y", pi_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            ef->create_direct_call( "mdl::operator*(float,float)", args.get()));

        result = module_builder->add_constant(
            "roughly two pi",
            expr.get(),
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_NOT_EQUAL( 0, result);

        result = module_builder->add_constant(
            "roughly_two_pi",
            expr.get(),
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and set module annotations

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> module_annotation( create_string_annotation( vf.get(),
            ef.get(), "::anno::description(string)", "description", "module annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> module_annotations(
            ef->create_annotation_block());
        module_annotations->add_annotation( module_annotation.get());

        result = module_builder->set_module_annotations(
            module_annotations.get(),
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add an overload of "fd_return_constant" with dummy parameter;
        // then remove "Enum", "foo_struct", the 2nd overload of "fd_return_constant", and "roughly_two_pi",
        // clear module annotations

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> parameter0( tf->create_int());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "parameter0", parameter0.get());

        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "fd_return_constant",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_UNIFORM,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "Enum", 0, context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "foo_struct", 0, context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "ad_dummy", 0, context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "fd_return_constant", 1, context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "roughly_two_pi", 0, context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->set_module_annotations(
            /*module_annotations*/ nullptr,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
#if 0
        // Disabled to avoid ending up with an empty module for import/export tests.
        result = module_builder->clear_module( context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
#endif
    }
}

void check_removed_materials_and_functions(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create module "mdl::removed_features" with materials "new_mat_0" and "new_mat_1" with
        // calls to "state::rounded_corner_normal$1.2(float,bool)" and
        // "df::fresnel_layer$1.3(color,float,bsdf,bsdf,float3)" in their body

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // new_mat_0
        {
            // create parameters

            mi::base::Handle<const mi::neuraylib::IType> across_materials_type( tf->create_bool());
            mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
            parameters->add_type( "across_materials", across_materials_type.get());

            // create body

            mi::base::Handle<mi::neuraylib::IExpression> n_across_materials_expr(
                ef->create_parameter( across_materials_type.get(), 0));
            mi::base::Handle<mi::neuraylib::IExpression_list> rcn_args(
                ef->create_expression_list());
            rcn_args->add_expression( "across_materials", n_across_materials_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> rcn_direct_call(
                ef->create_direct_call( "mdl::state::rounded_corner_normal$1.2(float,bool)",
                    rcn_args.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
                ef->create_expression_list());
            body_args->add_expression( "n", rcn_direct_call.get());
            mi::base::Handle<const mi::neuraylib::IExpression> body(
                ef->create_direct_call(
                    "mdl::" TEST_MDL "::md_float3_arg(float3)", body_args.get()));

            // run uniform analysis

            mi::base::Handle<const mi::neuraylib::IType> body_type( body->get_type());
            mi::base::Handle<const mi::IArray> uniform( module_builder->analyze_uniform(
                body.get(), mi::neuraylib::IType::MK_NONE, context.get()));
            MI_CHECK_CTX( context.get());
            MI_CHECK( uniform);

            // adapt parameter types based on uniform analysis

            mi::base::Handle<mi::neuraylib::IType_list> fixed_parameters( tf->create_type_list());
            for( mi::Size i = 0, n = uniform->get_length(); i < n; ++i) {
                mi::base::Handle<const mi::neuraylib::IType> parameter( parameters->get_type( i));
                const char* name = parameters->get_name( i);
                mi::base::Handle<const mi::IBoolean> element( uniform->get_element<mi::IBoolean>( i));
                if( element->get_value<bool>()) {
                    MI_CHECK( i == 0);
                    parameter = tf->create_alias(
                        parameter.get(), mi::neuraylib::IType::MK_UNIFORM, /*symbol*/ nullptr);
                } else {
                    MI_CHECK( i != 0);
                }
                fixed_parameters->add_type( name, parameter.get());
            }
            parameters = fixed_parameters;

            // add the material

            result = module_builder->add_function(
                "new_mat_0",
                body.get(),
                parameters.get(),
                /*defaults*/ nullptr,
                /*parameter_annotations*/ nullptr,
                /*annotations*/ nullptr,
                /*return_annotations*/ nullptr,
                /*is_exported*/ true,
                /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
                context.get());
            MI_CHECK_CTX( context.get());
            MI_CHECK_EQUAL( 0, result);
        }

        // new_mat_1
        {
            // create parameters

            mi::base::Handle<const mi::neuraylib::IType> ior_type( tf->create_color());
            mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
            parameters->add_type( "ior", ior_type.get());

            // create body

            mi::base::Handle<mi::neuraylib::IExpression> surface_scattering_ior_expr(
                ef->create_parameter( ior_type.get(), 0));
            mi::base::Handle<mi::neuraylib::IExpression_list> surface_scattering_args(
                ef->create_expression_list());
            surface_scattering_args->add_expression( "ior", surface_scattering_ior_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> surface_scattering_expr(
                ef->create_direct_call( "mdl::df::fresnel_layer$1.3(color,float,bsdf,bsdf,float3)",
                    surface_scattering_args.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> surface_args(
                ef->create_expression_list());
            surface_args->add_expression( "scattering", surface_scattering_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> surface_expr(
                ef->create_direct_call( "mdl::material_surface(bsdf,material_emission)",
                    surface_args.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
                ef->create_expression_list());
            body_args->add_expression( "surface", surface_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> body(
                ef->create_direct_call( "mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)",
                    body_args.get()));

            // create defaults

            mi::base::Handle<mi::neuraylib::IValue> ior_value(
                vf->create_color( 0, 0, 0));
            mi::base::Handle<mi::neuraylib::IExpression> ior_expr(
               ef->create_constant( ior_value.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
                ef->create_expression_list());
            defaults->add_expression( "ior", ior_expr.get());

            // add the material

            result = module_builder->add_function(
                "new_mat_1",
                body.get(),
                parameters.get(),
                defaults.get(),
                /*parameter_annotations*/ nullptr,
                /*annotations*/ nullptr,
                /*return_annotations*/ nullptr,
                /*is_exported*/ true,
                /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
                context.get());
            MI_CHECK_CTX( context.get());
            MI_CHECK_EQUAL( 0, result);
        }
    }
}

void check_analyze_uniform_open_graphs(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create "uniform_material" whose "ior" slot is connected to state::normal()

    {
        // instantiate ::state::normal()
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::state::normal()"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( nullptr, &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_uniform_normal"));
    }
    {
        // instantiate ::color(float3) with "color" connected to "analyze_uniform_normal"
        mi::base::Handle<mi::neuraylib::IExpression> color(
           ef->create_call( "analyze_uniform_normal"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
           ef->create_expression_list());
        args->add_expression( "rgb", color.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
               "mdl::color(float3)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_uniform_color"));
    }
    {
        // instantiate the material constructor with "ior" connected to "analyze_uniform_color"
        mi::base::Handle<mi::neuraylib::IExpression> ior(
           ef->create_call( "analyze_uniform_color"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
           ef->create_expression_list());
        args->add_expression( "ior", ior.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
               "mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "analyze_uniform_material"));
    }
    {
        bool query_result = false;
        mi::base::Handle<mi::IString> error_string( transaction->create<mi::IString>());

        // The subgraph starting at "analyze_uniform_color" is ok,
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_color", false, nullptr,
            query_result, error_string.get(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( error_string->get_c_str(), "");

        // But attached to the uniform "ior" slot of a material, it is broken.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_material", false, nullptr,
            query_result, error_string.get(), context.get());
        // MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL_CSTR( error_string->get_c_str(), "ior.rgb");

        // Access the "ior.color" node.
        mi::base::Handle<const mi::neuraylib::IFunction_call> query_fc(
            transaction->access<mi::neuraylib::IFunction_call>(
                "analyze_uniform_color"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> query_args(
            query_fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression> query_expr(
            query_args->get_expression( "rgb"));

        // And query explicitly that node.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_material", false,
            query_expr.get(), query_result, nullptr, context.get());
        // MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( query_result, true);

        // Access the "ior" node (arguments of the root expression need a special handling
        // internally).
        query_fc = transaction->access<mi::neuraylib::IFunction_call>(
                "analyze_uniform_material");
        query_args = query_fc->get_arguments();
        query_expr = query_args->get_expression( "ior");

        // And query explicitly that node.
        query_result = false;
        mdl_factory->analyze_uniform(
            transaction, "analyze_uniform_material", false,
            query_expr.get(), query_result, nullptr, context.get());
        // MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( query_result, true);
    }
}

void check_create_module_utf8(
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdle_api* mdle_api)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory(transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory(transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create module "mdl::123_check_unicode_new_materials" with a call in the body to
        // another module with utf8 name

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::123_check_unicode_new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> a_type( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "a", a_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> a_expr(
            ef->create_parameter( a_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "a", a_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( "mdl::" TEST_NO_1_RUS "::1_module::fd_test(float)",
                body_args.get()));

        // add the function

        result = module_builder->add_function(
            "new_fd_test",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // test variants
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::123_check_unicode_new_variants",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));
        result = module_builder->add_variant(
            "md_1_white",
            "mdl::" TEST_NO_1_RUS "::1_module::md_test(float)",
            /*defaults*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // test create mdle
        mi::base::Handle<mi::IStructure> mdle_data(transaction->create<mi::IStructure>("Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name0(
            mdle_data->get_value<mi::IString>("prototype_name"));
        prototype_name0->set_c_str(
            "mdl::" TEST_NO_1_RUS "::1_module::md_test(float)");

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, DIR_PREFIX "/123_check_unicode_create_mdle_with_unicode_€.mdle", mdle_data.get(), context.get());
        MI_CHECK_EQUAL(result, 0);
    }
    {
        // test create mdle from mdle with Unicode
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mdl_configuration->add_mdl_path( DIR_PREFIX);
        result = mdl_impexp_api->load_module(
            transaction, DIR_PREFIX "/123_check_unicode_create_mdle_with_unicode_€.mdle", context.get());
        MI_CHECK_EQUAL( result, 0);

        mi::base::Handle<const mi::IString> db_name( mdl_factory->get_db_module_name(
            DIR_PREFIX "/123_check_unicode_create_mdle_with_unicode_€.mdle"));
        std::string module_db_name = db_name->get_c_str();

        mi::base::Handle<mi::IStructure> mdle_data(transaction->create<mi::IStructure>("Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name0(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name0->set_c_str( (module_db_name + "::main(float)").c_str());

#ifndef MI_PLATFORM_WINDOWS
        // TODO Reported not to work on Windows. Unclear why.
        result = mdle_api->export_mdle(
            transaction, DIR_PREFIX "/123_check_unicode_create_mdle.mdle", mdle_data.get(), context.get());
        MI_CHECK_EQUAL( result, 0);
#endif

        mdl_configuration->remove_mdl_path( DIR_PREFIX);
    }
}

void check_import_elements_from_string(
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::Size always_imported_modules;
    {
        // import an MDL module from string
        const char* data = "mdl 1.0; export material some_material() = material();";
        MI_CHECK_EQUAL( 0,
            mdl_impexp_api->load_module_from_string( transaction, "::from_string", data));

        // check that the module exists now and has exactly one module definition
        mi::base::Handle<const mi::neuraylib::IModule> module;
        module = transaction->access<mi::neuraylib::IModule>( "mdl::from_string");
        MI_CHECK_EQUAL( 1, module->get_material_count());
        always_imported_modules = module->get_import_count();
    }
    {
        // import an MDL module (from strings) that imports an MDL module that was imported from
        // string

        // ::imports_from_string requires the module cache
        uninstall_external_resolver( mdl_configuration);

        const char* data = "mdl 1.0; import from_string::*;";
        MI_CHECK_EQUAL( 0,
            mdl_impexp_api->load_module_from_string( transaction, "::imports_from_string", data));

        // check that the module exists now and has exactly one imported module
        mi::base::Handle<const mi::neuraylib::IModule> module;
        module = transaction->access<mi::neuraylib::IModule>( "mdl::imports_from_string");
        MI_CHECK_EQUAL( always_imported_modules+1, module->get_import_count());

        install_external_resolver( mdl_configuration);
    }
    {
        // check that it is not possible to shadow modules found via the search path

        // check that the "mdl_elements::test_misc" module has not yet been loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::mdl_elements::test_misc"));
        MI_CHECK( !module);

        // check that creating a "mdl_elements::test_misc" module via import_elements_from_string()
        // fails
        const char* data = "mdl 1.0;";
        MI_CHECK_NOT_EQUAL( 0,
            mdl_impexp_api->load_module_from_string( transaction, "::mdl_elements::test_misc", data));

        // check that the "mdl_elements::test_misc" module still has not yet been loaded
        module = transaction->access<mi::neuraylib::IModule>( "mdl::mdl_elements::test_misc");
        MI_CHECK( !module);

        // check that it could be loaded, i.e., it is actually found in the search path
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction, "::mdl_elements::test_misc"));

        // check that the "mdl_elements::test_misc" module has been loaded
        module = transaction->access<mi::neuraylib::IModule>( "mdl::mdl_elements::test_misc");
        MI_CHECK( module);
    }
}


// save the current MDL search paths in the return value
std::vector<std::string> backup_mdl_paths(
    mi::neuraylib::IMdl_configuration* mdl_configuration)
{
    std::vector<std::string> v;
    mi::Size index = 0;

    while( true) {
        mi::base::Handle<const mi::IString> p( mdl_configuration->get_mdl_path( index));
        if( !p)
            break;
        v.push_back( p->get_c_str());
        ++index;
    }

    return v;
}

// set the MDL search paths to the argument
void restore_mdl_paths(
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    const std::vector<std::string>& module_paths)
{
    mdl_configuration->clear_mdl_paths();
    for( mi::Size i = 0; i < module_paths.size(); ++i)
        mdl_configuration->add_mdl_path( module_paths[i].c_str());
}

void check_mdl_export_reimport(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const char* module_name,
    const char* file_name,                   // different from module_name to allow re-import
    const char* mdl_module_name_from_string, // different from module_name to allow re-import
    mi::Sint32 error_number_file,
    mi::Sint32 error_number_string,
    bool modify_mdl_paths,
    bool bundle_resources,
    bool export_resources_with_module_prefix)
{
    error_number_file = -error_number_file;
    error_number_string = -error_number_string;

#ifndef MI_PLATFORM_WINDOWS
    std::string full_file_name = std::string( DIR_PREFIX) + "/exported/" + file_name;
#else // MI_PLATFORM_WINDOWS
    std::string full_file_name = std::string( DIR_PREFIX) + "\\exported\\"  + file_name;
#endif // MI_PLATFORM_WINDOWS
    fs::create_directory( DIR_PREFIX "/exported");

    // exporter options
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    context->set_option( "bundle_resources", bundle_resources);
    context->set_option( "export_resources_with_module_prefix", export_resources_with_module_prefix);

    std::vector<std::string> mdl_paths;

    if( modify_mdl_paths) {
        mdl_paths = backup_mdl_paths(mdl_configuration);
        mdl_configuration->clear_mdl_paths();
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( DIR_PREFIX));
        std::string path = MI::TEST::mi_src_path( "prod/lib/neuray");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
    }

    // file-based export
    result = mdl_impexp_api->export_module(
        transaction, module_name, full_file_name.c_str(), context.get());
    if( error_number_file == 0) {
        MI_CHECK_CTX( context.get());
    }
    MI_CHECK_EQUAL( error_number_file, result);

    // re-import from file
    if( error_number_file == 0) {
        std::string mdl_module_name_from_file = "::exported::";
        mdl_module_name_from_file += encode( file_name);
        mdl_module_name_from_file.resize( mdl_module_name_from_file.size()-4); // remove ".mdl"
        result = mdl_impexp_api->load_module(
            transaction, mdl_module_name_from_file.c_str(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }

#ifndef RESOLVE_RESOURCES_FALSE
    // string-based export
    mi::base::Handle<mi::IString> exported_module(
        transaction->create<mi::IString>( "String"));
    result = mdl_impexp_api->export_module_to_string(
        transaction, module_name, exported_module.get(), context.get());
    if( error_number_string == 0) {
        MI_CHECK_CTX( context.get());
    }
    MI_CHECK_EQUAL( error_number_string, result);

    // re-import from string
    if( error_number_string == 0) {
        result = mdl_impexp_api->load_module_from_string(
            transaction, mdl_module_name_from_string, exported_module->get_c_str(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
#endif // RESOLVE_RESOURCES_FALSE

    if( modify_mdl_paths)
        restore_mdl_paths( mdl_configuration, mdl_paths);
}

// check that MDL-specific type names for structures and enums are known to the API type system
void check_mdl_types(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // structure types in the standard modules
    const char* struct_types[] = {
        "::df::bsdf_component", "::df::edf_component", "::df::vdf_component"
    };

    for( mi::Size i = 0; i < sizeof( struct_types) / sizeof( const char*); ++i) {
        mi::base::Handle<const mi::neuraylib::IType> type( tf->create_struct( struct_types[i]));
        MI_CHECK( type);
    }

    // enum types in the standard modules
    const char* enum_types[] = {
        "::df::scatter_mode", "::state::coordinate_space", "::tex::gamma_mode", "::tex::wrap_mode"
    };

    for( mi::Size i = 0; i < sizeof( enum_types) / sizeof( const char*); ++i) {
        mi::base::Handle<const mi::neuraylib::IType> type( tf->create_enum( enum_types[i]));
        MI_CHECK( type);
    }
}

void check_enums(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_enum(::" TEST_MDL "::Enum)"));
    MI_CHECK( c_fd);

    mi::base::Handle<const mi::neuraylib::IType_list> types( c_fd->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType_enum> type_enum(
        types->get_type<mi::neuraylib::IType_enum>( "param0"));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", type_enum->get_symbol());

    mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_fd->get_defaults());
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
        c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
    mi::base::Handle<const mi::neuraylib::IValue_enum> c_param0_value(
        c_param0_expr->get_value<mi::neuraylib::IValue_enum>());
    MI_CHECK_EQUAL( 1, c_param0_value->get_index());

    mi::base::Handle<const mi::neuraylib::IType_enum> param0_type(
        tf->create_enum( "::" TEST_MDL "::Enum"));
    mi::base::Handle<mi::neuraylib::IValue_enum> m_param0_value(
        vf->create_enum( param0_type.get(), 0));
    mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
        ef->create_constant( m_param0_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, args->add_expression( "param0", m_param0_expr.get()));

    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
    c_fd->create_function_call( args.get(), &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( m_fc);
    MI_CHECK_EQUAL( 0,
        transaction->store( m_fc.get(), "mdl::" TEST_MDL "::fd_enum(" TEST_MDL "::Enum)_param0"));
}

// check constructors and field access operator for structs
void check_structs( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd;
    mi::base::Handle<const mi::neuraylib::IType_struct> type_struct;
    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_list> types;

    // default constructor
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::foo_struct()");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 0, c_fd->get_parameter_count());
    type_struct = c_fd->get_return_type<mi::neuraylib::IType_struct>();
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", type_struct->get_symbol());

    // elemental constructor
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::foo_struct(int,float)");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 2, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    MI_CHECK_EQUAL_CSTR( "param_int", c_fd->get_parameter_name( zero_size));
    type = types->get_type( 1);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
    MI_CHECK_EQUAL_CSTR( "param_float", c_fd->get_parameter_name( 1));
    type_struct = c_fd->get_return_type<mi::neuraylib::IType_struct>();
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", type_struct->get_symbol());

    // field access operator for param_int
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();
    type_struct = types->get_type<mi::neuraylib::IType_struct>( zero_size);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", type_struct->get_symbol());
    MI_CHECK_EQUAL_CSTR( "s", c_fd->get_parameter_name( zero_size));
    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
}

void check_indexing( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd;

    // length function for deferred arrays
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator_len(%3C0%3E[])");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();

    MI_CHECK_EQUAL_CSTR( "a", c_fd->get_parameter_name( zero_size));

    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
}

void check_immediate_arrays(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_immediate(int[42])"));
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();

    MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( zero_size));
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( type_array->is_immediate_sized());
    MI_CHECK_EQUAL( 42, type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    // create array constructor instance
    {
        // prepare argument for even array slots
        mi::base::Handle<mi::neuraylib::IValue> slot0_value( vf->create_int( 1701));
        mi::base::Handle<mi::neuraylib::IExpression> slot0_expr(
            ef->create_constant( slot0_value.get()));

        // prepare argument for odd array slots
        mi::base::Handle<mi::neuraylib::IExpression> slot1_expr(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));

        // prepare arguments for array constructor
        mi::base::Handle<mi::neuraylib::IExpression_list> array_constructor_args(
            ef->create_expression_list());
        for( mi::Size i = 0; i < 42; ++i) {
            std::ostringstream s;
            s << "value" << i;
            array_constructor_args->add_expression(
                s.str().c_str(), (i%2 == 0) ? slot0_expr.get() : slot1_expr.get());
        }

        // instantiate array constructor definition with these arguments
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd_array_constructor(
            transaction->access<mi::neuraylib::IFunction_definition>( "mdl::T[](...)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc_array_constructor(
            c_fd_array_constructor->create_function_call( array_constructor_args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( m_fc_array_constructor);
        MI_CHECK_EQUAL( 0, transaction->store(
            m_fc_array_constructor.get(), "mdl::" TEST_MDL "::array_constructor_int_42"));
    }

    // instantiate fd_immediate with constant expression as argument
    {
        // prepare argument
        mi::base::Handle<mi::neuraylib::IValue> param0_value(
            vf->create_array( type_array.get()));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
            ef->create_expression_list());
        args->add_expression( "param0", param0_expr.get());

        // instantiate the definition with that argument
        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
            c_fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( m_fc);

        {
            // check the argument
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
               m_fc->get_arguments());
            mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
                args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
            mi::base::Handle<const mi::neuraylib::IValue_array> c_param0_value(
                c_param0_expr->get_value<mi::neuraylib::IValue_array>());
            MI_CHECK_EQUAL( 42, c_param0_value->get_size());

            // set new argument (clone of old argument)
            mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
                ef->clone<mi::neuraylib::IExpression_constant>( c_param0_expr.get()));
            mi::base::Handle<mi::neuraylib::IValue_array> m_param0_value(
                m_param0_expr->get_value<mi::neuraylib::IValue_array>());
            MI_CHECK_EQUAL( -1, m_param0_value->set_size( 43));
            MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
        }

        // store the instance
        MI_CHECK_EQUAL( 0, transaction->store(
            m_fc.get(), "mdl::" TEST_MDL "::fc_immediate_constant"));
    }

    // instantiate fd_immediate with call expression as argument (array constructor)
    {
        // prepare argument
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_call( "mdl::" TEST_MDL "::array_constructor_int_42"));
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
            ef->create_expression_list());
        args->add_expression( "param0", param0_expr.get());

        // instantiate the definition with that argument
        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
            c_fd->create_function_call( args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( m_fc);

        {
            // check the argument
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
               m_fc->get_arguments());
            mi::base::Handle<const mi::neuraylib::IExpression_call> c_param0_expr(
                args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
            MI_CHECK_EQUAL_CSTR(
                "mdl::" TEST_MDL "::array_constructor_int_42", c_param0_expr->get_call());

            // set new argument (clone of old argument)
            mi::base::Handle<mi::neuraylib::IExpression_call> m_param0_expr(
                ef->clone<mi::neuraylib::IExpression_call>( c_param0_expr.get()));
            MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
        }

        // store the instance
        MI_CHECK_EQUAL( 0, transaction->store( m_fc.get(), "mdl::" TEST_MDL "::fc_immediate_call"));
    }
}

void check_deferred_arrays(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;

    // check deferred arrays in a function definition (fd_deferred)
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_deferred(int[N])"));
        MI_CHECK( c_fd);
        MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
        types = c_fd->get_parameter_types();

        MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( zero_size));
        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        type_array = type->get_interface<mi::neuraylib::IType_array>();
        MI_CHECK( !type_array->is_immediate_sized());
        MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
        type = type_array->get_element_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        type_array = type->get_interface<mi::neuraylib::IType_array>();
        MI_CHECK( !type_array->is_immediate_sized());
        MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
        type = type_array->get_element_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

        // instantiate fd_deferred with constant expression as argument
        {
            // prepare argument
            mi::base::Handle<mi::neuraylib::IValue_array> param0_value(
                vf->create_array( type_array.get()));
            MI_CHECK_EQUAL( 0, param0_value->set_size( 1));
            mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
                ef->create_constant( param0_value.get()));
            mi::base::Handle<mi::neuraylib::IExpression_list> args(
                ef->create_expression_list());
            args->add_expression( "param0", param0_expr.get());

            // instantiate the definition with that argument
            mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
                c_fd->create_function_call( args.get(), &result));
            MI_CHECK_EQUAL( 0, result);
            MI_CHECK( m_fc);

            {
                // check the argument
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                   m_fc->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
                    args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
                mi::base::Handle<const mi::neuraylib::IValue_array> c_param0_value(
                    c_param0_expr->get_value<mi::neuraylib::IValue_array>());
                MI_CHECK_EQUAL( 1, c_param0_value->get_size());

                // set new argument
                mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
                    ef->clone<mi::neuraylib::IExpression_constant>( c_param0_expr.get()));
                mi::base::Handle<mi::neuraylib::IValue_array> m_param0_value(
                    m_param0_expr->get_value<mi::neuraylib::IValue_array>());
                MI_CHECK_EQUAL( 0, m_param0_value->set_size( 2));
                MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
            }

            // store the instance
            MI_CHECK_EQUAL( 0, transaction->store(
                m_fc.get(), "mdl::" TEST_MDL "::fc_deferred_constant"));
        }

        // instantiate fd_deferred with call expression as argument (array constructor)
        {
            // prepare argument
            mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
                ef->create_call( "mdl::" TEST_MDL "::array_constructor_int_42"));
            mi::base::Handle<mi::neuraylib::IExpression_list> args(
                ef->create_expression_list());
            args->add_expression( "param0", param0_expr.get());

            // instantiate the definition with that argument
            mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
                c_fd->create_function_call( args.get(), &result));
            MI_CHECK_EQUAL( 0, result);
            MI_CHECK( m_fc);

            {
                // check the argument
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                   m_fc->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression_call> c_param0_expr(
                    args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
                MI_CHECK_EQUAL_CSTR(
                   "mdl::" TEST_MDL "::array_constructor_int_42", c_param0_expr->get_call());

                // set new argument (clone of old argument)
                mi::base::Handle<mi::neuraylib::IExpression_call> m_param0_expr(
                    ef->clone<mi::neuraylib::IExpression_call>( c_param0_expr.get()));
                MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_param0_expr.get()));
            }

            // store the instance
            MI_CHECK_EQUAL( 0, transaction->store(
                m_fc.get(), "mdl::" TEST_MDL "::fc_deferred_call"));
        }
    }

    // check deferred arrays in a material definition (md_deferred)
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_deferred(int[N])"));
        MI_CHECK( c_md);
        MI_CHECK_EQUAL( 1, c_md->get_parameter_count());
        types = c_md->get_parameter_types();

        MI_CHECK_EQUAL_CSTR( "param0", c_md->get_parameter_name( zero_size));
        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        type_array = type->get_interface<mi::neuraylib::IType_array>();
        MI_CHECK( !type_array->is_immediate_sized());
        MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
        type = type_array->get_element_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

        // instantiate md_deferred with constant expression as argument
        {
            // prepare argument
            mi::base::Handle<mi::neuraylib::IValue_array> param0_value(
                vf->create_array( type_array.get()));
            MI_CHECK_EQUAL( 0, param0_value->set_size( 1));
            mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
                ef->create_constant( param0_value.get()));
            mi::base::Handle<mi::neuraylib::IExpression_list> args(
                ef->create_expression_list());
            args->add_expression( "param0", param0_expr.get());

            // instantiate the definition with that argument
            mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
                c_md->create_function_call( args.get(), &result));
            MI_CHECK_EQUAL( 0, result);
            MI_CHECK( m_mi);

            {
                // check the argument
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                   m_mi->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
                    args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
                mi::base::Handle<const mi::neuraylib::IValue_array> c_param0_value(
                    c_param0_expr->get_value<mi::neuraylib::IValue_array>());
                MI_CHECK_EQUAL( 1, c_param0_value->get_size());

                // set new argument
                mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
                    ef->clone<mi::neuraylib::IExpression_constant>( c_param0_expr.get()));
                mi::base::Handle<mi::neuraylib::IValue_array> m_param0_value(
                    m_param0_expr->get_value<mi::neuraylib::IValue_array>());
                MI_CHECK_EQUAL( 0, m_param0_value->set_size( 2));
                MI_CHECK_EQUAL( 0, m_mi->set_argument( "param0", m_param0_expr.get()));
            }

            // store the instance
            MI_CHECK_EQUAL( 0, transaction->store(
                m_mi.get(), "mdl::" TEST_MDL "::mi_deferred_constant"));
        }

        // instantiate md_deferred with call expression as argument (array constructor)
        {
            // prepare argument
            mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
                ef->create_call( "mdl::" TEST_MDL "::array_constructor_int_42"));
            mi::base::Handle<mi::neuraylib::IExpression_list> args(
                ef->create_expression_list());
            args->add_expression( "param0", param0_expr.get());

            // instantiate the definition with that argument
            mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
                c_md->create_function_call( args.get(), &result));
            MI_CHECK_EQUAL( 0, result);
            MI_CHECK( m_mi);

            {
                // check the argument
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                   m_mi->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression_call> c_param0_expr(
                    args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
                MI_CHECK_EQUAL_CSTR(
                    "mdl::" TEST_MDL "::array_constructor_int_42", c_param0_expr->get_call());

                // set new argument (clone of old argument)
                mi::base::Handle<mi::neuraylib::IExpression_call> m_param0_expr(
                    ef->clone<mi::neuraylib::IExpression_call>( c_param0_expr.get()));
                MI_CHECK_EQUAL( 0, m_mi->set_argument( "param0", m_param0_expr.get()));
            }

            // store the instance
            MI_CHECK_EQUAL( 0, transaction->store(
                m_mi.get(), "mdl::" TEST_MDL "::mi_deferred_call"));
        }
    }
}

void check_type_binding(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<mi::neuraylib::IExpression_list> args;

    // check deferred arrays in a material definition (md_deferred_2, binding mismatch)

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_deferred_2(int[N],int[N])"));
    MI_CHECK( c_md);
    MI_CHECK_EQUAL( 2, c_md->get_parameter_count());
    types = c_md->get_parameter_types();

    MI_CHECK_EQUAL_CSTR( "param0", c_md->get_parameter_name( zero_size));
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( !type_array->is_immediate_sized());
    MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    MI_CHECK_EQUAL_CSTR( "param1", c_md->get_parameter_name( 1));
    type = types->get_type( 1);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( !type_array->is_immediate_sized());
    MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    // prepare arguments
    args = ef->create_expression_list();
    {
        mi::base::Handle<mi::neuraylib::IValue_array> param0_value(
            vf->create_array( type_array.get()));
        MI_CHECK_EQUAL( 0, param0_value->set_size( 1));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());

        mi::base::Handle<mi::neuraylib::IValue_array> param1_value(
            vf->create_array( type_array.get()));
        MI_CHECK_EQUAL( 0, param1_value->set_size( 2));
        mi::base::Handle<mi::neuraylib::IExpression> param1_expr(
            ef->create_constant( param1_value.get()));
        args->add_expression( "param1", param1_expr.get());
    }

    // instantiate the definition with that argument
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
        c_md->create_function_call( args.get(), &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( m_mi);

    // compile it (fails because the deferred-sized arrays of both arguments have different length)
    result = 0;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> m_mi_mi(
        m_mi->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::base::Handle<mi::neuraylib::ICompiled_material> c_cm(
        m_mi_mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
    MI_CHECK_NOT_EQUAL( 0, context->get_error_messages_count());
    MI_CHECK( !c_cm);

    // check deferred arrays in a function definition (fd_deferred_2, binding mismatch)

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_deferred_2(int[N],int[N])"));
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 2, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();

    MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( zero_size));
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( !type_array->is_immediate_sized());
    MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    MI_CHECK_EQUAL_CSTR( "param1", c_fd->get_parameter_name( 1));
    type = types->get_type( 1);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( !type_array->is_immediate_sized());
    MI_CHECK_EQUAL( static_cast<mi::Size>( -1), type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    // prepare arguments
    args = ef->create_expression_list();
    {
        mi::base::Handle<mi::neuraylib::IValue_array> param0_value(
            vf->create_array( type_array.get()));
        MI_CHECK_EQUAL( 0, param0_value->set_size( 1));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());

        mi::base::Handle<mi::neuraylib::IValue_array> param1_value(
            vf->create_array( type_array.get()));
        MI_CHECK_EQUAL( 0, param1_value->set_size( 2));
        mi::base::Handle<mi::neuraylib::IExpression> param1_expr(
            ef->create_constant( param1_value.get()));
        args->add_expression( "param1", param1_expr.get());
    }

    // instantiate the definition with that argument
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
        c_fd->create_function_call( args.get(), &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( m_fc);

    // store the call
    MI_CHECK_EQUAL( 0, transaction->store( m_fc.get(), "mdl::" TEST_MDL "::fc_deferred_2"));
    m_fc = 0;

    // create a material instance with m_fc as attachment
    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_deferred(int[N])");
    MI_CHECK( c_md);

    args = ef->create_expression_list();
    {
        mi::base::Handle<mi::neuraylib::IExpression_call> arg(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_2"));
        args->add_expression( "param0", arg.get());
    }

    m_mi = c_md->create_function_call( args.get(), &result);
    m_mi_mi =  m_mi->get_interface<mi::neuraylib::IMaterial_instance>();
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( m_mi);

    // compile it (fails because the deferred-sized arrays of both arguments have different length)
    result = 0;
    c_cm = m_mi_mi->create_compiled_material(
        mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get());
    MI_CHECK_NOT_EQUAL( 0, context->get_error_messages_count());
    MI_CHECK( !c_cm);
}

void check_annotations(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
    // check annotation definitions of a module
    {
        mi::base::Handle<const mi::neuraylib::IModule> mod_anno(
            transaction->access<mi::neuraylib::IModule>(
                "mdl::anno"));
        mi::Size n = mod_anno->get_annotation_definition_count();
        MI_CHECK(n > 0);

        mi::base::Handle<const mi::neuraylib::IAnnotation_definition> anno_null(
            mod_anno->get_annotation_definition(nullptr));
        MI_CHECK(!anno_null);

        mi::base::Handle<const mi::neuraylib::IAnnotation_definition> anno_display_name_def(
            mod_anno->get_annotation_definition("::anno::display_name(string)"));
        MI_CHECK( anno_display_name_def);

        MI_CHECK_EQUAL_CSTR( anno_display_name_def->get_module(), "mdl::anno");
        MI_CHECK_EQUAL_CSTR( anno_display_name_def->get_name(), "::anno::display_name(string)");
        MI_CHECK_EQUAL_CSTR( anno_display_name_def->get_mdl_module_name(), "::anno");
        MI_CHECK_EQUAL_CSTR( anno_display_name_def->get_mdl_simple_name(), "display_name");
        MI_CHECK_EQUAL_CSTR( anno_display_name_def->get_mdl_parameter_type_name( 0), "string");
        MI_CHECK_EQUAL( anno_display_name_def->get_parameter_count(), 1);
        MI_CHECK_EQUAL( anno_display_name_def->get_parameter_index("name"), 0);
        MI_CHECK_EQUAL( anno_display_name_def->get_parameter_index("does_not_exist"), mi::Size(-1));
        MI_CHECK_EQUAL_CSTR( anno_display_name_def->get_parameter_name(0), "name");
        MI_CHECK_EQUAL( anno_display_name_def->get_parameter_name(1), nullptr);
        MI_CHECK_EQUAL( anno_display_name_def->get_semantic(),
            mi::neuraylib::IAnnotation_definition::AS_DISPLAY_NAME_ANNOTATION);
        MI_CHECK_EQUAL( anno_display_name_def->is_exported(), true);

        mi::neuraylib::Mdl_version since, removed;
        anno_display_name_def->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

        mi::base::Handle <const mi::neuraylib::IType_list> anno_param_types(
            anno_display_name_def->get_parameter_types());
        MI_CHECK_EQUAL( anno_param_types->get_size(), 1);
        mi::base::Handle <const mi::neuraylib::IType> anno_param_type(
            anno_param_types->get_type("name"));
        MI_CHECK_EQUAL( anno_param_type->get_kind(), mi::neuraylib::IType::TK_STRING);

        mi::base::Handle<const mi::neuraylib::IExpression_list> anno_defaults(
            anno_display_name_def->get_defaults());
        MI_CHECK( anno_defaults);
        MI_CHECK_EQUAL( anno_defaults->get_size(), 0);

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> anno_annos(
            anno_display_name_def->get_annotations());
        MI_CHECK(!anno_annos);

        mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
            mdl_factory->create_expression_factory(transaction));
        MI_CHECK(ef);
        mi::base::Handle<mi::neuraylib::IValue_factory> vf(
            mdl_factory->create_value_factory(transaction));
        MI_CHECK(ef);

        // added in 1.3
        mi::base::Handle<const mi::neuraylib::IAnnotation_definition> ad(
            mod_anno->get_annotation_definition( "::anno::deprecated()"));
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_3);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

        // removed in MDL 1.3
        ad = mod_anno->get_annotation_definition(
            "::anno::version_number$1.2(int,int,int,int)");
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_1_3);

        // custom annotation
        mi::base::Handle<const mi::neuraylib::IModule> mod_test_mdl(
            transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
        ad = mod_test_mdl->get_annotation_definition( "::" TEST_MDL "::anno_2_int(int,int)");
        ad->get_mdl_version( since, removed);
        MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_7);
        MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

        // test create annotation
        {
            mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
                ef->create_expression_list());
            mi::base::Handle<mi::neuraylib::IValue_string> the_name(
                vf->create_string("The Name"));
            MI_CHECK(the_name);

            mi::base::Handle<mi::neuraylib::IExpression_constant> dname(
                ef->create_constant(the_name.get()));
            MI_CHECK(dname);

            arguments->add_expression("name", dname.get());

            // create with correct arguments
            mi::base::Handle<const mi::neuraylib::IAnnotation> anno_display_name(
                anno_display_name_def->create_annotation(arguments.get()));
            MI_CHECK( anno_display_name);

            anno_display_name_def =
                anno_display_name->get_definition();
            MI_CHECK( anno_display_name_def);

            // create with invalid number of arguments
            arguments->add_expression("do_not_exist", dname.get());
            anno_display_name =
                anno_display_name_def->create_annotation(arguments.get());
            MI_CHECK(!anno_display_name);
        }
        {
            // create with wrong type
            mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
                ef->create_expression_list());
            mi::base::Handle<mi::neuraylib::IValue_bool> the_name(
                vf->create_bool(false));
            MI_CHECK(the_name);

            mi::base::Handle<mi::neuraylib::IExpression_constant> dname(
                ef->create_constant(the_name.get()));
            MI_CHECK(dname);

            arguments->add_expression("name", dname.get());
            mi::base::Handle<const mi::neuraylib::IAnnotation> anno_display_name(
                anno_display_name_def->create_annotation(arguments.get()));
            MI_CHECK(!anno_display_name);
        }
        {
            // create without arguments
            mi::base::Handle<const mi::neuraylib::IAnnotation> anno_display_name(
                anno_display_name_def->create_annotation(nullptr));
            MI_CHECK(!anno_display_name);
        }
    }

    mi::base::Handle<const mi::neuraylib::IAnnotation_block> block;
    mi::base::Handle<const mi::neuraylib::IAnnotation> anno;
    mi::base::Handle<const mi::neuraylib::IAnnotation_definition> anno_def;
    mi::base::Handle<const mi::neuraylib::IAnnotation_list> annos;
    mi::base::Handle<const mi::neuraylib::IExpression_list> args;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> arg;
    mi::base::Handle<const mi::neuraylib::IValue_string> value_string;
    mi::base::Handle<const mi::neuraylib::IValue_float> value_float;

    // check annotations of a module

    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>(
            "mdl::" TEST_MDL));
    MI_CHECK( c_module);

    block = c_module->get_annotations();

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "module description annotation", value_string->get_value());
    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::description(string)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "description");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "string");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 1);
    MI_CHECK_EQUAL( anno_def->get_parameter_index("description"), 0);
    MI_CHECK_EQUAL_CSTR( anno_def->get_parameter_name(0), "description");

    // check annotations of a material definition

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_with_annotations(color,float)"));
    MI_CHECK( c_md);

    block = c_md->get_annotations();

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "material description annotation", value_string->get_value());

    anno = block->get_annotation( 1);
    MI_CHECK_EQUAL_CSTR( "::anno::unused()", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 0, args->get_size());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::unused()");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "unused");
    MI_CHECK(!anno_def->get_mdl_parameter_type_name( 0));
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 0);

    annos = c_md->get_parameter_annotations();
    block = annos->get_annotation_block( "param0");

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param0: one description annotation", value_string->get_value());

    block = annos->get_annotation_block( "param1");

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param1: two description annotations (1)", value_string->get_value());

    anno = block->get_annotation( 1);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param1: two description annotations (2)", value_string->get_value());

    anno = block->get_annotation( 2);
    MI_CHECK_EQUAL_CSTR( "::anno::soft_range(float,float)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 2, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "min");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 0.5f, value_float->get_value());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "max");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 1.5f, value_float->get_value());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::soft_range(float,float)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "soft_range");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "float");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 1), "float");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 2);

    anno = block->get_annotation( 3);
    MI_CHECK_EQUAL_CSTR( "::anno::hard_range(float,float)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 2, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "min");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 0.0f, value_float->get_value());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "max");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 3.0f, value_float->get_value());

    anno = block->get_annotation( 4);
    MI_CHECK_EQUAL_CSTR( "::anno::display_name(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "name");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "md_with_annotations_param1_display_name", value_string->get_value());

    block = c_md->get_return_annotations();

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "return type description annotation", value_string->get_value());

    // check block of a function definition

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_with_annotations(color,float)"));
    MI_CHECK( c_fd);

    block = c_fd->get_annotations();

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "function description annotation", value_string->get_value());

    anno = block->get_annotation( 1);
    MI_CHECK_EQUAL_CSTR( "::anno::unused()", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 0, args->get_size());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::unused()");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "unused");
    MI_CHECK(!anno_def->get_mdl_parameter_type_name( 0));
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 0);

    annos = c_fd->get_parameter_annotations();
    block = annos->get_annotation_block( "param0");

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param0: one description annotation", value_string->get_value());

    block = annos->get_annotation_block( "param1");

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param1: two description annotations (1)", value_string->get_value());

    anno = block->get_annotation( 1);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param1: two description annotations (2)", value_string->get_value());

    anno = block->get_annotation( 2);
    MI_CHECK_EQUAL_CSTR( "::anno::soft_range(float,float)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 2, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "min");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 0.5f, value_float->get_value());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "max");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 1.5f, value_float->get_value());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::soft_range(float,float)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "soft_range");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "float");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 1), "float");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 2);


    anno = block->get_annotation( 3);
    MI_CHECK_EQUAL_CSTR( "::anno::hard_range(float,float)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 2, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "min");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 0.0f, value_float->get_value());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "max");
    value_float = arg->get_value<mi::neuraylib::IValue_float>();
    MI_CHECK_EQUAL( 3.0f, value_float->get_value());

    anno = block->get_annotation( 4);
    MI_CHECK_EQUAL_CSTR( "::anno::display_name(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "name");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "fd_with_annotations_param1_display_name", value_string->get_value());

    block = c_fd->get_return_annotations();

    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "return type description annotation", value_string->get_value());

    // check annotation for enum types

    mi::base::Handle<const mi::neuraylib::IType_list> types( c_module->get_types());
    MI_CHECK_EQUAL( 5, types->get_size());

    mi::base::Handle<const mi::neuraylib::IType_enum> type0(
        types->get_type<mi::neuraylib::IType_enum>( zero_size));
    MI_CHECK( type0);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", types->get_name( 0));

    block = type0->get_annotations();
    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "enum annotation", value_string->get_value());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::description(string)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "description");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "string");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 1);

    block = type0->get_value_annotations( zero_size);
    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "one annotation", value_string->get_value());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::description(string)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "description");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "string");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 1);

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::description(string)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "description");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "string");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 1);


    block = type0->get_value_annotations( 1);
    MI_CHECK( !block);

    // check annotation for struct types

    mi::base::Handle<const mi::neuraylib::IType_struct> type1(
        types->get_type<mi::neuraylib::IType_struct>( 1));
    MI_CHECK( type1);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", types->get_name( 1));

    block = type1->get_annotations();
    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "struct annotation", value_string->get_value());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::description(string)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "description");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "string");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 1);

    block = type1->get_field_annotations( zero_size);
    anno = block->get_annotation( zero_size);
    MI_CHECK_EQUAL_CSTR( "::anno::description(string)", anno->get_name());
    args = anno->get_arguments();
    MI_CHECK_EQUAL( 1, args->get_size());
    arg = args->get_expression<mi::neuraylib::IExpression_constant>( "description");
    value_string = arg->get_value<mi::neuraylib::IValue_string>();
    MI_CHECK_EQUAL_CSTR( "param_int annotation", value_string->get_value());

    anno_def = anno->get_definition();
    MI_CHECK( anno_def);
    MI_CHECK_EQUAL_CSTR( anno_def->get_name(), "::anno::description(string)");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_module_name(), "::anno");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_simple_name(), "description");
    MI_CHECK_EQUAL_CSTR( anno_def->get_mdl_parameter_type_name( 0), "string");
    MI_CHECK_EQUAL( anno_def->get_parameter_count(), 1);

    block = type1->get_field_annotations( 1);
    MI_CHECK( !block);
}

// check default initializers that reference an earlier parameter
void check_parameter_indices(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md;
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc;
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc;
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi;
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi;

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<const mi::neuraylib::IValue_int> c_value_int;
    mi::base::Handle<const mi::neuraylib::IValue_float> c_value_float;
    mi::base::Handle<const mi::neuraylib::IValue_color> c_value_color;
    mi::base::Handle<const mi::neuraylib::IExpression> c_expr;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_expr_constant;
    mi::base::Handle<const mi::neuraylib::IExpression_call> c_expr_call;
    mi::base::Handle<const mi::neuraylib::IExpression_parameter> c_expr_parameter;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults, c_args;
    mi::base::Handle<mi::neuraylib::IExpression_list> args;

    // check parameter references of a function definition

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_2_index(int,float)");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_2_index(int,float)", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_2_index", c_fd->get_mdl_simple_name());

    types = c_fd->get_parameter_types();
    type = types->get_type( "param0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    type = types->get_type( "param1");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    defaults = c_fd->get_defaults();
    c_expr = defaults->get_expression( "param0");
    MI_CHECK( !c_expr);

    c_expr_call = defaults->get_expression<mi::neuraylib::IExpression_call>( "param1");
    MI_CHECK( c_expr_call);
    type = c_expr_call->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
    const char* name = c_expr_call->get_call();
    c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
    MI_CHECK( c_fc);
    const char* definition_name = c_fc->get_mdl_function_definition();
    MI_CHECK_EQUAL_CSTR( definition_name, "float(int)");
    c_args = c_fc->get_arguments();
    c_expr_parameter = c_args->get_expression<mi::neuraylib::IExpression_parameter>( zero_size);
    MI_CHECK( c_expr_parameter);
    type = c_expr_parameter->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    MI_CHECK_EQUAL( c_expr_parameter->get_index(), 0);

    // create a corresponding function call, supplying an argument only for "param0"
    {
        args = ef->create_expression_list();
        mi::base::Handle<mi::neuraylib::IValue_int> param0_value( vf->create_int( -42));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());
        m_fc = c_fd->create_function_call( args.get(), &result);
        MI_CHECK_EQUAL( 0, result);

        // check value of "param1"
        c_args = m_fc->get_arguments();
        c_expr_call = c_args->get_expression<mi::neuraylib::IExpression_call>( "param1");
        MI_CHECK( c_expr_call);
        type = c_expr_call->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
        const char* name = c_expr_call->get_call();
        c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
        MI_CHECK( c_fc);
        const char* definition_name = c_fc->get_mdl_function_definition();
        MI_CHECK_EQUAL_CSTR( definition_name, "float(int)");
        c_args = c_fc->get_arguments();
        c_expr_constant = c_args->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
        MI_CHECK( c_expr_constant);
        c_value_int = c_expr_constant->get_value<mi::neuraylib::IValue_int>();
        MI_CHECK( c_value_int);
        MI_CHECK_EQUAL( -42, c_value_int->get_value());
        c_fc = 0;
        m_fc = 0;
    }

    // create a corresponding function call, supplying arguments for both parameters
    {
        args = ef->create_expression_list();
        mi::base::Handle<mi::neuraylib::IValue_int> param0_value( vf->create_int( -42));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());
        mi::base::Handle<mi::neuraylib::IValue_float> param1_value( vf->create_float( -43.0f));
        mi::base::Handle<mi::neuraylib::IExpression> param1_expr(
            ef->create_constant( param1_value.get()));
        args->add_expression( "param1", param1_expr.get());
        m_fc = c_fd->create_function_call( args.get(), &result);
        MI_CHECK_EQUAL( 0, result);

        // check value of "param0"
        c_args = m_fc->get_arguments();
        c_expr_constant = args->get_expression<mi::neuraylib::IExpression_constant>( "param0");
        MI_CHECK( c_expr_constant);
        c_value_int = c_expr_constant->get_value<mi::neuraylib::IValue_int>();
        MI_CHECK( c_value_int);
        MI_CHECK_EQUAL( -42, c_value_int->get_value());

        // check value of "param1"
        c_expr_constant = args->get_expression<mi::neuraylib::IExpression_constant>( "param1");
        MI_CHECK( c_expr_constant);
        c_value_float = c_expr_constant->get_value<mi::neuraylib::IValue_float>();
        MI_CHECK( c_value_float);
        MI_CHECK_EQUAL( -43.0f, c_value_float->get_value());
        m_fc = 0;
    }

    // check parameter reference via array constructor

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_2_index_nested(int,int[N])");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_2_index_nested(int,int[N])", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_2_index_nested", c_fd->get_mdl_simple_name());
    MI_CHECK_EQUAL( 2, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();

    type = types->get_type( "param0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    type = types->get_type( "param1");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    defaults = c_fd->get_defaults();
    c_expr = defaults->get_expression( "param0");
    MI_CHECK( !c_expr);

    c_expr_call = defaults->get_expression<mi::neuraylib::IExpression_call>( "param1");
    MI_CHECK( c_expr_call);
    type = c_expr_call->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
     name = c_expr_call->get_call();
    c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
    MI_CHECK( c_fc);
    definition_name = c_fc->get_mdl_function_definition();
    MI_CHECK_EQUAL_CSTR( definition_name, "T[](...)");
    c_args = c_fc->get_arguments();
    c_expr_parameter = c_args->get_expression<mi::neuraylib::IExpression_parameter>( zero_size);
    MI_CHECK( c_expr_parameter);
    type = c_expr_parameter->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    MI_CHECK_EQUAL( c_expr_parameter->get_index(), 0);

    // check parameter references of a material definition

    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_2_index(color,color)");
    MI_CHECK( c_md);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_md->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_md->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::md_2_index(color,color)", c_md->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "md_2_index", c_md->get_mdl_simple_name());
    types = c_md->get_parameter_types();

    type = types->get_type( "tint0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
    type = types->get_type( "tint1");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());

    defaults = c_md->get_defaults();
    c_expr = defaults->get_expression( "tint0");
    MI_CHECK( !c_expr);

    c_expr_parameter = defaults->get_expression<mi::neuraylib::IExpression_parameter>( "tint1");
    MI_CHECK( c_expr_parameter);
    type = c_expr_parameter->get_type();
    MI_CHECK_EQUAL( c_expr_parameter->get_index(), 0);

    // create a corresponding material instance, supplying an argument only for "tint0"
    {
        args = ef->create_expression_list();
        mi::base::Handle<mi::neuraylib::IValue_color> tint0_value(
            vf->create_color( 0.5f, 0.5f, 0.5f));
        mi::base::Handle<mi::neuraylib::IExpression> tint0_expr(
            ef->create_constant( tint0_value.get()));
        args->add_expression( "tint0", tint0_expr.get());
        m_mi = c_md->create_function_call( args.get(), &result);
        MI_CHECK_EQUAL( 0, result);

        // check value of "tint1"
        c_args = m_mi->get_arguments();
        c_expr_constant = c_args->get_expression<mi::neuraylib::IExpression_constant>( "tint1");
        MI_CHECK( c_expr_constant);
        type = c_expr_constant->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
        c_value_color = c_expr_constant->get_value<mi::neuraylib::IValue_color>();
        MI_CHECK( c_value_color);
        MI_CHECK( vf->compare( tint0_value.get(), c_value_color.get()) == 0);

        // for later tests
        MI_CHECK_EQUAL( 0, transaction->store(
            m_mi.get(), "mdl::" TEST_MDL "::mi_2_index_used_index"));
    }

    // create a corresponding material instance, supplying arguments for both parameters
    {
        args = ef->create_expression_list();
        mi::base::Handle<mi::neuraylib::IValue_color> tint0_value(
            vf->create_color( 0.5f, 0.5f, 0.5f));
        mi::base::Handle<mi::neuraylib::IExpression> tint0_expr(
            ef->create_constant( tint0_value.get()));
        args->add_expression( "tint0", tint0_expr.get());
        mi::base::Handle<mi::neuraylib::IValue_color> tint1_value(
            vf->create_color( 0.6f, 0.6f, 0.6f));
        mi::base::Handle<mi::neuraylib::IExpression> tint1_expr(
            ef->create_constant( tint1_value.get()));
        args->add_expression( "tint1", tint1_expr.get());
        m_mi = c_md->create_function_call( args.get(), &result);
        MI_CHECK_EQUAL( 0, result);

        // check value of "tint0"
        c_args = m_mi->get_arguments();
        c_expr_constant = c_args->get_expression<mi::neuraylib::IExpression_constant>( "tint0");
        MI_CHECK( c_expr_constant);
        type = c_expr_constant->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
        c_value_color = c_expr_constant->get_value<mi::neuraylib::IValue_color>();
        MI_CHECK( c_value_color);
        MI_CHECK( vf->compare( tint0_value.get(), c_value_color.get()) == 0);

        // check value of "tint1"
        c_expr_constant = c_args->get_expression<mi::neuraylib::IExpression_constant>( "tint1");
        MI_CHECK( c_expr_constant);
        type = c_expr_constant->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
        c_value_color = c_expr_constant->get_value<mi::neuraylib::IValue_color>();
        MI_CHECK( c_value_color);
        MI_CHECK( vf->compare( tint1_value.get(), c_value_color.get()) == 0);

        // for later tests
        MI_CHECK_EQUAL( 0, transaction->store(
           m_mi.get(), "mdl::" TEST_MDL "::mi_2_index_unused_index"));
    }

    // check parameter reference via array constructor

    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_2_index_nested(color,color[N])");
    MI_CHECK( c_md);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_md->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_md->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR(
        "::" TEST_MDL "::md_2_index_nested(color,color[N])",
        c_md->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "md_2_index_nested", c_md->get_mdl_simple_name());
    MI_CHECK_EQUAL( 2, c_md->get_parameter_count());
    types = c_md->get_parameter_types();

    type = types->get_type( "tint0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
    type = types->get_type( "tint1");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());

    defaults = c_md->get_defaults();
    c_expr = defaults->get_expression( "tint0");
    MI_CHECK( !c_expr);

    c_expr_call = defaults->get_expression<mi::neuraylib::IExpression_call>( "tint1");
    MI_CHECK( c_expr_call);
    type = c_expr_call->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
     name = c_expr_call->get_call();
    c_fc = transaction->access<mi::neuraylib::IFunction_call>( name);
    MI_CHECK( c_fc);
    definition_name = c_fc->get_mdl_function_definition();
    MI_CHECK_EQUAL_CSTR( definition_name, "T[](...)");
    c_args = c_fc->get_arguments();
    c_expr_parameter = c_args->get_expression<mi::neuraylib::IExpression_parameter>( zero_size);
    MI_CHECK( c_expr_parameter);
    type = c_expr_parameter->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
    MI_CHECK_EQUAL( c_expr_parameter->get_index(), 0);
}

void check_material_parameter(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md;
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi;

    // create a material instance mdl::" TEST_MDL "::mi_arg
    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_0()");
    MI_CHECK( c_md);
    m_mi = c_md->create_function_call( 0, &result);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 0, transaction->store( m_mi.get(), "mdl::" TEST_MDL "::mi_arg"));
    m_mi = 0;

    // prepare argument: a call expression referencing mdl::" TEST_MDL "::mi_arg
    mi::base::Handle<mi::neuraylib::IExpression_call> arg(
        ef->create_call( "mdl::" TEST_MDL "::mi_arg"));
    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());
    args->add_expression( "material0", arg.get());

    // instantiate mdl::" TEST_MDL "::md_thin_walled with mdl::" TEST_MDL "::mi_arg as argument for
    // "material0"
    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_thin_walled(material)");
    MI_CHECK( c_md);
    m_mi = c_md->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 0, transaction->store( m_mi.get(), "mdl::" TEST_MDL "::mi_thin_walled"));
}

void check_reexported_function(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md;
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi;

    // prepare argument: a call expression referencing to mdl::" TEST_MDL "::fc_normal
    mi::base::Handle<mi::neuraylib::IExpression_call> arg(
        ef->create_call( "mdl::" TEST_MDL "::fc_normal"));
    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());
    args->add_expression( "normal", arg.get());

    // instantiate mdl::" TEST_MDL "::md_reexport(float3) with mdl::" TEST_MDL "::fc_normal as argument for "normal"
    c_md = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_reexport(float3)");
    MI_CHECK( c_md);
    m_mi = c_md->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 0, transaction->store( m_mi.get(), "mdl::" TEST_MDL "::mi_reexport"));
}

void dump(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const mi::neuraylib::ICompiled_material* cm,
    std::ostream& s)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    for( mi::Size i = 0; i < cm->get_parameter_count(); ++i) {
        mi::base::Handle<const mi::neuraylib::IValue> argument( cm->get_argument( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            vf->dump( argument.get(), name.str().c_str(), 1));
        const char* pn = cm->get_parameter_name( i);
        if( !pn)
            pn = "n/a";
        s << "    argument (original parameter name: " << pn << ") "
          << result->get_c_str() << std::endl;
    }

    for( mi::Uint32 i = 0; i < cm->get_temporary_count(); ++i) {
        mi::base::Handle<const mi::neuraylib::IExpression> temporary( cm->get_temporary( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            ef->dump( temporary.get(), name.str().c_str(), 1));
        s << "    temporary " << result->get_c_str() << std::endl;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
    mi::base::Handle<const mi::IString> result(
        ef->dump( body.get(), 0, 1));
    s << "    body " << result->get_c_str() << std::endl;
}

void check_icompiled_material(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    bool class_compilation)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    mi::base::Handle<const mi::neuraylib::IFunction_call> mi;
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi_mi;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    context->set_option("meters_per_scene_unit", 42.0f);
    context->set_option("wavelength_min", 44.0f);
    context->set_option("wavelength_max", 46.0f);
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_tmp_ms");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 42.0f, cm->get_mdl_meters_per_scene_unit());
    MI_CHECK_EQUAL( 44.0f, cm->get_mdl_wavelength_min());
    MI_CHECK_EQUAL( 46.0f, cm->get_mdl_wavelength_max());
    MI_CHECK_EQUAL( 1, cm->get_temporary_count());
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 1, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "tint");
    }

    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_tmp_bsdf");

    context->set_option("meters_per_scene_unit", 43.0f);
    context->set_option("wavelength_min", 45.0f);
    context->set_option("wavelength_max", 47.0f);

    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 43.0f, cm->get_mdl_meters_per_scene_unit());
    MI_CHECK_EQUAL( 45.0f, cm->get_mdl_wavelength_min());
    MI_CHECK_EQUAL( 47.0f, cm->get_mdl_wavelength_max());
    MI_CHECK_EQUAL( 1, cm->get_temporary_count());
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 1, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "tint");
    }

    {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( body->get_arguments());

        // check some default values
        mi::base::Handle<const mi::neuraylib::IExpression_constant> ior_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "ior"));
        mi::base::Handle<const mi::neuraylib::IValue_color> ior_value(
            ior_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<const mi::neuraylib::IValue_color> exp_ior_value(
            vf->create_color( 1.0f, 1.0f, 1.0f));
        MI_CHECK_EQUAL( 0, vf->compare( ior_value.get(), exp_ior_value.get()));
        mi::base::Handle<const mi::neuraylib::IExpression_constant> thin_walled_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "thin_walled"));
        mi::base::Handle<const mi::neuraylib::IValue_bool> thin_walled(
            thin_walled_expr->get_value<mi::neuraylib::IValue_bool>());
        MI_CHECK_EQUAL( false, thin_walled->get_value());

        // check temporary index for the common sub-expression
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> surface(
            args->get_expression<mi::neuraylib::IExpression_temporary>( "surface"));
        MI_CHECK_EQUAL( 0, surface->get_index());
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> backface(
            args->get_expression<mi::neuraylib::IExpression_temporary>( "backface"));
        MI_CHECK_EQUAL( 0, backface->get_index());

        // check temporary 0
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> temporary(
            cm->get_temporary<mi::neuraylib::IExpression_direct_call>( 0));
        MI_CHECK_EQUAL_CSTR(
            "mdl::material_surface(bsdf,material_emission)", temporary->get_definition());
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments(
            temporary->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> scattering(
            arguments->get_expression<mi::neuraylib::IExpression_direct_call>( "scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments2(
            scattering->get_arguments());
        MI_CHECK_EQUAL_CSTR(
            "mdl::df::diffuse_reflection_bsdf(color,float,string)", scattering->get_definition());
        if( !class_compilation) {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> tint_expr(
                arguments2->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
            mi::base::Handle<const mi::neuraylib::IValue_color> tint_value(
                tint_expr->get_value<mi::neuraylib::IValue_color>());
            mi::base::Handle<const mi::neuraylib::IValue_color> exp_tint_value(
                vf->create_color( 1.0f, 1.0f, 1.0f));
             MI_CHECK_EQUAL( 0, vf->compare( ior_value.get(), exp_ior_value.get()));
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression_parameter> tint_expr(
                arguments2->get_expression<mi::neuraylib::IExpression_parameter>( "tint"));
            MI_CHECK_EQUAL( 0, tint_expr->get_index());
        }
        mi::base::Handle<const mi::neuraylib::IExpression_constant> roughness_expr(
            arguments2->get_expression<mi::neuraylib::IExpression_constant>( "roughness"));
        mi::base::Handle<const mi::neuraylib::IValue_float> roughness_value(
            roughness_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_EQUAL( 0.0f, roughness_value->get_value());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> handle_expr(
            arguments2->get_expression<mi::neuraylib::IExpression_constant>( "handle"));
        mi::base::Handle<const mi::neuraylib::IValue_string> handle_value(
            handle_expr->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK_EQUAL_CSTR( "", handle_value->get_value());

        // check argument 0
        if( class_compilation) {
            mi::base::Handle<const mi::neuraylib::IValue_color> argument(
                cm->get_argument<mi::neuraylib::IValue_color>( 0));
            mi::base::Handle<const mi::neuraylib::IValue_color> exp_argument(
                vf->create_color( 1.0f, 1.0f, 1.0f));
            MI_CHECK_EQUAL( 0, vf->compare( argument.get(), exp_argument.get()));
        }
    }

    // check that the ternary operator is not inlined in class compilation mode
    mi = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_ternary_operator_argument");

    context->set_option("meters_per_scene_unit", 1.0f);
    context->set_option("wavelength_min", 380.0f);
    context->set_option("wavelength_max", 780.0f);

    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 1, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "cond");
    }

    // check that resource sharing is disabled in class compilation mode
    mi = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_resource_sharing");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 2, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "tex0");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "tex1");
    }

    // check that certain functions are folded in the MDL integration in instance compilation mode
#ifndef RESOLVE_RESOURCES_FALSE
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( body->get_arguments());

        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> surface(
            args->get_expression<mi::neuraylib::IExpression_direct_call>( "surface"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_surface(
            surface->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> scattering(
            args_surface->get_expression<mi::neuraylib::IExpression_direct_call>( "scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_scattering(
            scattering->get_arguments());

        if( !class_compilation) {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
            MI_CHECK( tint);
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_direct_call>( "tint"));
            MI_CHECK( tint);
        }
    }
    {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( body->get_arguments());

        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> backface(
            args->get_expression<mi::neuraylib::IExpression_direct_call>( "backface"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_backface(
            backface->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> scattering(
            args_backface->get_expression<mi::neuraylib::IExpression_direct_call>( "scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_scattering(
            scattering->get_arguments());

        if( !class_compilation) {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
            MI_CHECK( tint);
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_direct_call>( "tint"));
            MI_CHECK( tint);
        }
    }
#else // RESOLVE_RESOURCES_FALSE
    // Skip test if resolve_resources is set to false. The resulting compiled material has a
    // completely different structure.
#endif // RESOLVE_RESOURCES_FALSE

    // check that certain parameters are folded in class compilation mode
    // (1) base case without any context flags
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding2");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 4, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.x");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param1");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 3), "param0");
    }

    // (2) fold all bool parameters
    context->set_option( "fold_all_bool_parameters", true);
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 3, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.x");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param1");
    }
    context->set_option( "fold_all_bool_parameters", false);

    // (3) fold all enum parameters
    context->set_option( "fold_all_enum_parameters", true);
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 3, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.x");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param0");
    }
    context->set_option( "fold_all_enum_parameters", false);

    // (4) fold a parameter by name
    mi::base::Handle<mi::IArray> array( factory->create<mi::IArray>( "String[1]"));
    mi::base::Handle<mi::IString> element( array->get_element<mi::IString>( zero_size));
    element->set_c_str( "param2.x");
    result = context->set_option( "fold_parameters", array.get());
    MI_CHECK_EQUAL( 0, result);
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 3, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param1");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param0");
    }
    context->set_option( "fold_all_enum_parameters", false);

    // (5a) fold geometry.cutout_opacity (with temporaries)
    context->set_option( "fold_trivial_cutout_opacity", true);
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding_cutout_opacity");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_trivial_cutout_opacity", false);

    // (5b) fold geometry.cutout_opacity (with parameter reference)
    context->set_option( "fold_trivial_cutout_opacity", true);
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding_cutout_opacity2");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_trivial_cutout_opacity", false);

    // (5c) fold geometry.cutout_opacity (material wrapped into copy-constructor-like meta-material)
    mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_folding_cutout_opacity2(material)"));
    mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
    mi::base::Handle<mi::neuraylib::IExpression> arg(
        ef->create_call( "mdl::" TEST_MDL "::mi_folding_cutout_opacity"));
    args->add_expression( "m", arg.get());
    mi = md->create_function_call( args.get(), &result);
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    MI_CHECK_EQUAL( 0, result);

    context->set_option( "fold_trivial_cutout_opacity", true);
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        // finding geometry.cutout_opacity fails (a path like m.geometry.cutout_opacity might work)
        MI_CHECK_EQUAL( 2, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "m.o1");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "m.o2");
    }
    context->set_option( "fold_trivial_cutout_opacity", false);

    // (6a) fold transparent layers (a simple case)
    context->set_option( "fold_transparent_layers", true);
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding_transparent_layers");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_transparent_layers", false);

    // (6b) fold transparent layers (more complicated case)
    context->set_option( "fold_transparent_layers", true);
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding_transparent_layers2");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_transparent_layers", false);

    // (6c) fold transparent layers (nested case)
    context->set_option( "fold_transparent_layers", true);
    mi = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding_transparent_layers3");
    mi_mi = mi->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi_mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_transparent_layers", false);

    // check that cycles in the call graph are detected
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_cycle(color)"));

        // Create two function calls
        mi::base::Handle<mi::neuraylib::IFunction_call> fc1( fd->create_function_call( 0));
        check_success( fc1);
        transaction->store( fc1.get(), "mdl::" TEST_MDL "::fc_cycle1");

        mi::base::Handle<mi::neuraylib::IFunction_call> fc2( fd->create_function_call( 0));
        check_success( fc2);
        transaction->store( fc2.get(), "mdl::" TEST_MDL "::fc_cycle2");

        // Create cycle
        mi::neuraylib::Argument_editor ae_fc1( transaction, "mdl::" TEST_MDL "::fc_cycle1", mdl_factory);
        check_success( ae_fc1.set_call( "tint", "mdl::" TEST_MDL "::fc_cycle2") == 0);

        mi::neuraylib::Argument_editor ae_fc2( transaction, "mdl::" TEST_MDL "::fc_cycle2", mdl_factory);
        check_success( ae_fc2.set_call( "tint", "mdl::" TEST_MDL "::fc_cycle1") == 0);

        // Create material instance
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_1(color)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> mi( md->create_function_call( 0));
        check_success( mi);
        transaction->store( mi.get(), "mdl::" TEST_MDL "::mi_cycle");

        // Call cycle
        mi::neuraylib::Argument_editor ae_mi( transaction, "mdl::" TEST_MDL "::mi_cycle", mdl_factory);
        check_success( ae_mi.set_call( "tint", "mdl::" TEST_MDL "::fc_cycle1") == -6);
    }

    // compile all material instances in the DB to test hashes (and to increase testing of the
    // internal code paths)
    const char* material_instances[] = {
        "mdl::" TEST_MDL "::mi_0",
        "mdl::" TEST_MDL "::mi_1",
        "mdl::" TEST_MDL "::mi_2_index_used_index",
        "mdl::" TEST_MDL "::mi_2_index_unused_index",
        "mdl::" TEST_MDL "::mi_tmp_ms",
        "mdl::" TEST_MDL "::mi_tmp_bsdf",
        "mdl::" TEST_MDL "::mi_deferred_constant",
        "mdl::" TEST_MDL "::mi_deferred_call",
        "mdl::" TEST_MDL "::mi_arg",
        "mdl::" TEST_MDL "::mi_thin_walled",
        "mdl::" TEST_MDL "::mi_ternary_operator_argument",
        "mdl::" TEST_MDL "::mi_ternary_operator_default",
        "mdl::" TEST_MDL "::mi_ternary_operator_body",
        "mdl::" TEST_MDL "::mi_resource_sharing",
        "mdl::" TEST_MDL "::mi_folding",
        "mdl::" TEST_MDL "::mi_jit",
        "mdl::" TEST_MDL "::mi_reexport"
    };

    mi::Size count = sizeof( material_instances) / sizeof( const char*);
    std::vector<mi::base::Uuid> hashes;
    for( mi::Size i = 0; i < count; ++i) {
        mi::base::Handle<const mi::neuraylib::IFunction_call> mi(
            transaction->access<mi::neuraylib::IFunction_call>( material_instances[i]));
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi_mi(
            mi->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi_mi->create_compiled_material( flags, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( cm);
        mi::base::Uuid hash = cm->get_hash();
        hashes.push_back( hash);
#if 0
        fprintf( stderr, "%s\n", material_instances[i]);
        fprintf( stderr, "    overall: %08x %08x %08x %08x\n",
            hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
        for( mi::Uint32 i = mi::neuraylib::SLOT_FIRST; i <= mi::neuraylib::SLOT_LAST; ++i) {
            hash = cm->get_slot_hash( mi::neuraylib::Material_slot( i));
            fprintf( stderr, "    slot %u: %08x %08x %08x %08x\n", i,
                hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
        }
        dump( transaction, mdl_factory, cm.get(), std::cerr);
#endif
    }

    // compare hashes
    for( mi::Uint32 i = 0; i < count; ++i)
        for( mi::Uint32 j = i+1; j < count; ++j) {
            std::string name_i = material_instances[i];
            std::string name_j = material_instances[j];
            bool hashes_equal = hashes[i] == hashes[j];
            // parameterless materials (class and instance compilation)
            if(         name_i == "mdl::" TEST_MDL "::mi_0"
                     && name_j == "mdl::" TEST_MDL "::mi_deferred_constant")
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_0"
                     && name_j == "mdl::" TEST_MDL "::mi_deferred_call")
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_0"
                     && name_j == "mdl::" TEST_MDL "::mi_arg")
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_deferred_constant"
                     && name_j == "mdl::" TEST_MDL "::mi_arg")
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_deferred_call"
                     && name_j == "mdl::" TEST_MDL "::mi_arg")
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_tmp_ms"
                     && name_j == "mdl::" TEST_MDL "::mi_tmp_bsdf")
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_deferred_constant"
                     && name_j == "mdl::" TEST_MDL "::mi_deferred_call")
                MI_CHECK( hashes_equal);
            // parameterless materials (instance compilation mode only)
            else if(    name_i == "mdl::" TEST_MDL "::mi_0"
                     && name_j == "mdl::" TEST_MDL "::mi_2_index_used_index" && !class_compilation)
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_2_index_used_index"
                     && name_j == "mdl::" TEST_MDL "::mi_deferred_constant" && !class_compilation)
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_2_index_used_index"
                     && name_j == "mdl::" TEST_MDL "::mi_deferred_call" && !class_compilation)
                MI_CHECK( hashes_equal);
            else if(    name_i == "mdl::" TEST_MDL "::mi_2_index_used_index"
                     && name_j == "mdl::" TEST_MDL "::mi_arg" && !class_compilation)
                MI_CHECK( hashes_equal);
            // equal materials (class compilation mode, actual arguments of simple parameters do
            // not matter here)
            else if(    name_i == "mdl::" TEST_MDL "::mi_2_index_used_index"
                     && name_j == "mdl::" TEST_MDL "::mi_2_index_unused_index" && class_compilation)
                MI_CHECK( hashes_equal);
            else
                MI_CHECK( !hashes_equal);
        }

    // Test the expression path encoded in the compiled material parameter names
    // and also the `get_connected_function_db_name` function to retrieve the db of the function call for a given parameter
    if (class_compilation)
    {
        mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        // more than one path possible if the instance parameter is used multiple times <path, expected_db_name, default_call>
        auto check_path = [](const mi::neuraylib::ICompiled_material* compiled_material,
                             const char* instance_db_name,
                             const char* param_path,
                             const char* expected_connected_db_name,
                             bool default_call = false)
        {
            MI_CHECK(compiled_material);
            mi::Size param_count = compiled_material->get_parameter_count();

            // the order of the parameters in the compiled material is not specified
            // so instead of specifying the index, use search for the expected paths
            mi::Size index = -1;
            for (mi::Size i = 0; i < param_count; ++i) {
                const char* current_param_path = compiled_material->get_parameter_name(i);
                if (std::strcmp(current_param_path, param_path) == 0) {
                    index = i;
                    break;
                }
            }
            MI_CHECK_NOT_EQUAL_MSG(index, -1, "expected parameter paths not found");

            mi::Sint32 err = 0;
            mi::base::Handle<const mi::IString> res(compiled_material->get_connected_function_db_name(instance_db_name, index, &err));
            if (!expected_connected_db_name) { // nullptr expected
                MI_CHECK_EQUAL(err, -3);
                MI_CHECK_EQUAL(res.is_valid_interface(), false);
                return;
            }
            MI_CHECK_ZERO(err);
            MI_CHECK(res);
            std::string connectedName = res->get_c_str();
            if (default_call) {
                // we don't know the name of default constructed calls
                size_t pos = connectedName.rfind('_');
                MI_CHECK_NOT_EQUAL(pos, std::string::npos);
                connectedName = connectedName.substr(0, pos);
                MI_CHECK_EQUAL_CSTR(connectedName.c_str(), expected_connected_db_name);
            }
            else
                MI_CHECK_EQUAL_CSTR(connectedName.c_str(), expected_connected_db_name);
        };

        MI_CHECK_ZERO(mdl_impexp_api->load_module(transaction, "::test_class_param_paths", context.get()));
        MI_CHECK_CTX(context.get());

        // tests for structures
        mi::base::Handle<const mi::neuraylib::IFunction_definition> main_default_node_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::test_class_param_paths::main_defaults(::test_class_param_paths::lookup_value)"));
        MI_CHECK(main_default_node_def);
        mi::base::Handle<const mi::neuraylib::IFunction_definition> struct_constructor_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::test_class_param_paths::lookup_value(bool,color,float)"));
        MI_CHECK(struct_constructor_def);
        mi::base::Handle<const mi::neuraylib::IFunction_definition> main_indirect_node_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::test_class_param_paths::main_indirect(color)"));
        MI_CHECK(main_indirect_node_def);
        mi::base::Handle<const mi::neuraylib::IFunction_definition> extract_node_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::test_class_param_paths::extract_value(::test_class_param_paths::lookup_value)"));
        MI_CHECK(extract_node_def);
        mi::base::Handle<const mi::neuraylib::IFunction_definition> constructor_attachement_node_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::test_class_param_paths::create_value(float)"));
        MI_CHECK(constructor_attachement_node_def);

        // structure value in case the function call for construction is not needed
        std::string a_main_db_name = "mdl_instance::test_class_param_paths::a";
        mi::base::Handle<mi::neuraylib::IFunction_call> a_node(main_default_node_def->create_function_call(nullptr));
        mi::base::Handle<mi::neuraylib::IMaterial_instance> a_node_mat(a_node->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<const mi::neuraylib::ICompiled_material> a_node_compiled(a_node_mat->create_compiled_material(flags));
        MI_CHECK_ZERO(transaction->store(a_node_mat.get(), a_main_db_name.c_str()));
        MI_CHECK_EQUAL(a_node_compiled->get_parameter_count(), 1);
        check_path(a_node_compiled.get(), a_main_db_name.c_str(), "lookup", nullptr); // expected a struct value here, not a call

        // create a function call for the constructor that uses default arguments
        std::string b_main_db_name = "mdl_instance::test_class_param_paths::b";
        std::string b_constructor_db_name = "mdl_instance::test_class_param_paths::b_constructor";
        MI_CHECK_ZERO(transaction->copy(a_main_db_name.c_str(), b_main_db_name.c_str()));
        mi::base::Handle<mi::neuraylib::IFunction_call> b_constructor_node(struct_constructor_def->create_function_call(nullptr));
        MI_CHECK_ZERO(transaction->store(b_constructor_node.get(), b_constructor_db_name.c_str()));
        {
            mi::neuraylib::Argument_editor ae(transaction, b_main_db_name.c_str(), mdl_factory, true);
            MI_CHECK_ZERO(ae.set_call("lookup", b_constructor_db_name.c_str()));
        }
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> b_node_mat(transaction->access<mi::neuraylib::IMaterial_instance>(b_main_db_name.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> b_node_compiled(b_node_mat->create_compiled_material(flags));
        MI_CHECK_EQUAL(b_node_compiled->get_parameter_count(), 3); // constructor with 3 literals
        // get back the constructor call
        check_path(b_node_compiled.get(), b_main_db_name.c_str(), "lookup.valid", b_constructor_db_name.c_str());
        check_path(b_node_compiled.get(), b_main_db_name.c_str(), "lookup.value", b_constructor_db_name.c_str());
        check_path(b_node_compiled.get(), b_main_db_name.c_str(), "lookup.alpha", b_constructor_db_name.c_str());

        // create a function attachment that reads from a structure created by it's constructor
        std::string c_main_db_name = "mdl_instance::test_class_param_paths::c";
        std::string c_extract_db_name = "mdl_instance::test_class_param_paths::c_extract";
        std::string c_constructor_db_name = "mdl_instance::test_class_param_paths::c_constructor";
        MI_CHECK_ZERO(transaction->copy(b_constructor_db_name.c_str(), c_constructor_db_name.c_str()));
        mi::base::Handle<mi::neuraylib::IFunction_call> c_node(main_indirect_node_def->create_function_call(nullptr));
        mi::base::Handle<mi::neuraylib::IMaterial_instance> c_node_mat(c_node->get_interface<mi::neuraylib::IMaterial_instance>());
        MI_CHECK_ZERO(transaction->store(c_node_mat.get(), c_main_db_name.c_str()));
        mi::base::Handle<mi::neuraylib::IFunction_call> c_extract_node(extract_node_def->create_function_call(nullptr));
        MI_CHECK_ZERO(transaction->store(c_extract_node.get(), c_extract_db_name.c_str()));
        {
            mi::neuraylib::Argument_editor ae(transaction, c_extract_db_name.c_str(), mdl_factory, true);
            MI_CHECK_ZERO(ae.set_call("lookup", c_constructor_db_name.c_str()));
        }
        {
            mi::neuraylib::Argument_editor ae(transaction, c_main_db_name.c_str(), mdl_factory, true);
            MI_CHECK_ZERO(ae.set_call("tint", c_extract_db_name.c_str()));
        }
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> c_node_mat2(transaction->access<mi::neuraylib::IMaterial_instance>(c_main_db_name.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> c_node_compiled(c_node_mat2->create_compiled_material(flags));

        // get back the constructor call behind the extract
        MI_CHECK_EQUAL(c_node_compiled->get_parameter_count(), 3);
        check_path(c_node_compiled.get(), c_main_db_name.c_str(), "tint.lookup.valid", c_constructor_db_name.c_str());
        check_path(c_node_compiled.get(), c_main_db_name.c_str(), "tint.lookup.value", c_constructor_db_name.c_str());
        check_path(c_node_compiled.get(), c_main_db_name.c_str(), "tint.lookup.alpha", c_constructor_db_name.c_str());

        // same as before, this time with another function call attached to the constructor
        std::string d_main_db_name = "mdl_instance::test_class_param_paths::d";
        std::string d_extract_db_name = "mdl_instance::test_class_param_paths::d_extract";
        std::string d_constructor_db_name = "mdl_instance::test_class_param_paths::d_constructor";
        std::string d_constructor_attachment_db_name = "mdl_instance::test_class_param_paths::d_constructor_attachment";
        MI_CHECK_ZERO(transaction->copy(c_main_db_name.c_str(), d_main_db_name.c_str()));
        MI_CHECK_ZERO(transaction->copy(c_extract_db_name.c_str(), d_extract_db_name.c_str()));
        MI_CHECK_ZERO(transaction->copy(c_constructor_db_name.c_str(), d_constructor_db_name.c_str()));
        mi::base::Handle<mi::neuraylib::IFunction_call> d_constructor_attachment_node(constructor_attachement_node_def->create_function_call(nullptr));
        MI_CHECK_ZERO(transaction->store(d_constructor_attachment_node.get(), d_constructor_attachment_db_name.c_str()));
        {
            mi::neuraylib::Argument_editor ae(transaction, d_constructor_db_name.c_str(), mdl_factory, true);
            MI_CHECK_ZERO(ae.set_call("value", d_constructor_attachment_db_name.c_str()));
        }
        {
            mi::neuraylib::Argument_editor ae(transaction, d_extract_db_name.c_str(), mdl_factory, true);
            MI_CHECK_ZERO(ae.set_call("lookup", d_constructor_db_name.c_str()));
        }
        {
            mi::neuraylib::Argument_editor ae(transaction, d_main_db_name.c_str(), mdl_factory, true);
            MI_CHECK_ZERO(ae.set_call("tint", d_extract_db_name.c_str()));
        }
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> d_node_mat(transaction->access<mi::neuraylib::IMaterial_instance>(d_main_db_name.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> d_node_compiled(d_node_mat->create_compiled_material(flags));
        MI_CHECK_EQUAL(d_node_compiled->get_parameter_count(), 3);
        check_path(d_node_compiled.get(), d_main_db_name.c_str(), "tint.lookup.valid", d_constructor_db_name.c_str());
        check_path(d_node_compiled.get(), d_main_db_name.c_str(), "tint.lookup.value.scale", d_constructor_attachment_db_name.c_str());
        check_path(d_node_compiled.get(), d_main_db_name.c_str(), "tint.lookup.alpha", d_constructor_db_name.c_str());

        // tests for arrays
        mi::base::Handle<const mi::neuraylib::IFunction_definition> main_array_node_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::test_class_param_paths::main_array(float,float,color[4],int)"));
        MI_CHECK(main_array_node_def);

        // structure value in case the function call for construction is not needed
        std::string x_main_db_name = "mdl_instance::test_class_param_paths::x";
        mi::base::Handle<mi::neuraylib::IFunction_call> x_node(main_array_node_def->create_function_call(nullptr));
        mi::base::Handle<mi::neuraylib::IMaterial_instance> x_node_mat(x_node->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<const mi::neuraylib::ICompiled_material> x_node_compiled(x_node_mat->create_compiled_material(flags));
        MI_CHECK_ZERO(transaction->store(x_node_mat.get(), x_main_db_name.c_str()));
        MI_CHECK_EQUAL(x_node_compiled->get_parameter_count(), 5);
        check_path(x_node_compiled.get(), x_main_db_name.c_str(), "data.value0", "mdl::T[]", true);
        check_path(x_node_compiled.get(), x_main_db_name.c_str(), "data.value1.rgb.y", "mdl::operator*", true);
        check_path(x_node_compiled.get(), x_main_db_name.c_str(), "data.value2.scale", "mdl::test_class_param_paths::create_value", true);
        check_path(x_node_compiled.get(), x_main_db_name.c_str(), "data.value3.scale_2", "mdl::test_class_param_paths::create_value_2", true);
        check_path(x_node_compiled.get(), x_main_db_name.c_str(), "index", nullptr);
    }
}

void check_overloaded_hashing(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::Sint32 result = 0;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    result = mdl_impexp_api->load_module(transaction, "::test_mdl_hashing", context.get());
    MI_CHECK_CTX(context.get());
    MI_CHECK_EQUAL(result, 0);

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>("mdl::test_mdl_hashing"));
    mi::Size material_count = module->get_material_count();
    MI_CHECK_EQUAL(0, result);
    MI_CHECK_EQUAL(119, material_count);

        // Compile the material instance.
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory(transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory(transaction));

    std::vector<mi::base::Uuid> all_hashes;
    std::vector<std::string> material_names;
    std::vector<mi::base::Handle<mi::neuraylib::ICompiled_material const> > all_materials;

    mi::Size material_index = 0;
    for (mi::Size i = 0; i < material_count; i++) {
        const char* material_name = module->get_material(i);
        material_names.push_back(material_name);
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(material_name));
        MI_CHECK(md);

        std::string instance_name("mdl::test_mdl_hashing::mi_" + std::to_string(i));
        {
            mi::base::Handle<mi::neuraylib::IFunction_call> mi(
                md->create_function_call(0, &result));
            MI_CHECK_EQUAL(0, result);
            MI_CHECK_EQUAL(0, transaction->store(mi.get(), instance_name.c_str()));
        }

        mi::base::Handle<const mi::neuraylib::IFunction_call>
            mi(transaction->access<const mi::neuraylib::IFunction_call>(
                instance_name.c_str()));
        MI_CHECK(mi.is_valid_interface());
        mi::base::Handle<const mi::neuraylib::IMaterial_instance>
            mi_mi(mi->get_interface<mi::neuraylib::IMaterial_instance>());
        MI_CHECK(mi_mi.is_valid_interface());

        {
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());
            mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;
            mi::base::Handle<const mi::neuraylib::ICompiled_material>
                cm(mi_mi->create_compiled_material(flags, context.get()));
            MI_CHECK_CTX(context);
            MI_CHECK(cm);

            mi::base::Uuid compiled_hash = cm->get_hash();
            int other_index = 0;
            for (auto& other_hash : all_hashes) {
                // Do not compare test base and variant.
                if (other_hash != compiled_hash || (material_index == 1 && other_index == 0)) {
                    // All good.
                } else {
                    std::string& other_material_name = material_names[other_index];
                    mi::neuraylib::ICompiled_material const* other_cm = all_materials[other_index].get();
                    std::cout << "*** " << other_material_name << "\n";
                    dump(transaction, mdl_factory, other_cm, std::cout);
                    std::cout << "*** " << material_name << ":\n";
                    dump(transaction, mdl_factory, cm.get(), std::cout);
                    MI_CHECK_NOT_EQUAL(other_hash.m_id1, compiled_hash.m_id1);
                }
                other_index++;
            }
            all_materials.push_back(make_handle_dup(cm.get()));
            all_hashes.push_back(compiled_hash);
        }
        material_index++;
    }
}

void check_matrix( mi::neuraylib::ITransaction* transaction)
{
    // see test_mdl.mdl for rationale
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_matrix(float3x2)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IType_list> types( c_fd->get_parameter_types());

    mi::base::Handle<const mi::neuraylib::IType_matrix> matrix_type(
        types->get_type<mi::neuraylib::IType_matrix>( zero_size));
    MI_CHECK_EQUAL( 3, matrix_type->get_size());
    mi::base::Handle<const mi::neuraylib::IType_vector> vector_type(
        matrix_type->get_element_type());
    MI_CHECK_EQUAL( 2, vector_type->get_size());
    mi::base::Handle<const mi::neuraylib::IType_atomic> element_type(
        vector_type->get_element_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, element_type->get_kind());

    mi::base::Handle<const mi::neuraylib::IValue_float> v;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());

    mi::base::Handle<const mi::neuraylib::IExpression_constant> default_expr(
        defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size));
    mi::base::Handle<const mi::neuraylib::IValue_matrix> matrix_value(
        default_expr->get_value<mi::neuraylib::IValue_matrix>());

    mi::base::Handle<const mi::neuraylib::IValue_vector> column0_value(
        matrix_value->get_value( 0));
    v = column0_value->get_value<mi::neuraylib::IValue_float>( 0);
    MI_CHECK_EQUAL( 0.0f, v->get_value());
    v = column0_value->get_value<mi::neuraylib::IValue_float>( 1);
    MI_CHECK_EQUAL( 1.0f, v->get_value());

    mi::base::Handle<const mi::neuraylib::IValue_vector> column1_value(
        matrix_value->get_value( 1));
    v = column1_value->get_value<mi::neuraylib::IValue_float>( 0);
    MI_CHECK_EQUAL( 2.0f, v->get_value());
    v = column1_value->get_value<mi::neuraylib::IValue_float>( 1);
    MI_CHECK_EQUAL( 3.0f, v->get_value());

    mi::base::Handle<const mi::neuraylib::IValue_vector> column2_value(
        matrix_value->get_value( 2));
    v = column2_value->get_value<mi::neuraylib::IValue_float>( 0);
    MI_CHECK_EQUAL( 4.0f, v->get_value());
    v = column2_value->get_value<mi::neuraylib::IValue_float>( 1);
    MI_CHECK_EQUAL( 5.0f, v->get_value());

    mi::math::Matrix<mi::Float32,3,2> m;
    result = get_value( matrix_value.get(), m);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( m[0][0], 0.0f);
    MI_CHECK_EQUAL( m[0][1], 1.0f);
    MI_CHECK_EQUAL( m[1][0], 2.0f);
    MI_CHECK_EQUAL( m[1][1], 3.0f);
    MI_CHECK_EQUAL( m[2][0], 4.0f);
    MI_CHECK_EQUAL( m[2][1], 5.0f);
}

// check that a deep copy is done when defaults are used due to lack of arguments
void check_deep_copy_of_defaults( mi::neuraylib::ITransaction* transaction)
{
    // get DB name of the function call referenced in the default for param0 of
    // fd_default_call()
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_default_call(int)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());
    mi::base::Handle<const mi::neuraylib::IExpression_call> param0_default(
        defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
    const char* param0_default_call = param0_default->get_call();

    // instantiate fd_default_call() using the default,
    // get DB name of the function call referenced in the argument param0
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc( c_fd->create_function_call( 0, &result));
    MI_CHECK_EQUAL( 0, result);
    mi::base::Handle<const mi::neuraylib::IExpression_list> args( m_fc->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression_call> param0_argument(
        args->get_expression<mi::neuraylib::IExpression_call>( "param0"));
    const char* param0_argument_call = param0_argument->get_call();

    // check that both are different (but have the same prefix)
    MI_CHECK_NOT_EQUAL_CSTR( param0_argument_call, param0_default_call);
    MI_CHECK( strncmp( param0_argument_call, "mdl::" TEST_MDL "::fd_1_", 20) == 0);
    MI_CHECK( strncmp( param0_default_call, "mdl::" TEST_MDL "::fd_1(int)_", 25) == 0);
}

// check that attachments used in defaults are immutable
void check_immutable_defaults(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    {
        // get DB name of the function call referenced in the default for param0 of
        // fd_default_call()
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_default_call(int)"));
        MI_CHECK( c_fd);
        mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression_call> param0_default(
            defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
        const char* param0_default_call = param0_default->get_call();

        // setting the param0 argument of the referenced function call should fail
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            transaction->edit<mi::neuraylib::IFunction_call>( param0_default_call));
        MI_CHECK( fc);
        mi::base::Handle<mi::neuraylib::IValue> value( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression> expression(
            ef->create_constant( value.get()));
        MI_CHECK_EQUAL( -4, fc->set_argument( "param0", expression.get()));
    }
    {
        // get DB name of the material instance referenced in the default for param0 of
        // md_default_call
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_default_call(material)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_md->get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression_call> param0_default(
            defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
        const char* param0_default_call = param0_default->get_call();

        // setting the tint argument of the referenced material instance should fail
        mi::base::Handle<mi::neuraylib::IFunction_call> mi(
            transaction->edit<mi::neuraylib::IFunction_call>( param0_default_call));
        MI_CHECK( mi);
        mi::base::Handle<mi::neuraylib::IValue> value( vf->create_color( 1.0, 1.0, 1.0));
        mi::base::Handle<mi::neuraylib::IExpression> expression(
            ef->create_constant( value.get()));
        MI_CHECK_EQUAL( -4, mi->set_argument( "tint", expression.get()));
    }
}

void check_set_get_value(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    {
        // bool / IValue_bool
        mi::base::Handle<mi::neuraylib::IValue_bool> data( vf->create_bool( false));
        bool value = true;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, false);
        result = set_value( data.get(), true);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, true);
    }
    {
        // mi::Sint32 / IValue_int
        mi::base::Handle<mi::neuraylib::IValue_int> data( vf->create_int( 1));
        mi::Sint32 value = 0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1);
        result = set_value( data.get(), 2);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2);
    }
    {
        // mi::Sint32 / IValue_enum
        mi::base::Handle<const mi::neuraylib::IType_enum> type(
            tf->create_enum( "::" TEST_MDL "::Enum"));
        mi::base::Handle<mi::neuraylib::IValue_enum> data( vf->create_enum( type.get(), 0));
        mi::Sint32 value = 0;
        const char* name = 0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1); // index 0 => value 1
        result = get_value( data.get(), name);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( name, "one");

        result = set_value( data.get(), 2);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2);
        result = get_value( data.get(), name);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( name, "two");

        result = set_value( data.get(), "one");
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1);
        result = get_value( data.get(), name);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( name, "one");
    }
    {
        // mi::Float32 / IValue_float
        mi::base::Handle<mi::neuraylib::IValue_float> data( vf->create_float( 1.0f));
        mi::Float32 value = 0.0f;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0f);
        result = set_value( data.get(), 2.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0f);
    }
    {
        // mi::Float64 / IValue_float
        mi::base::Handle<mi::neuraylib::IValue_float> data( vf->create_float( 1.0f));
        mi::Float64 value = 0.0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0);
        result = set_value( data.get(), 2.0);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0);
    }
    {
        // mi::Float32 / IValue_double
        mi::base::Handle<mi::neuraylib::IValue_double> data( vf->create_double( 1.0));
        mi::Float32 value = 0.0f;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0f);
        result = set_value( data.get(), 2.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0f);
    }
    {
        // mi::Float64 / IValue_double
        mi::base::Handle<mi::neuraylib::IValue_double> data( vf->create_double( 1.0));
        mi::Float64 value = 0.0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 1.0);
        result = set_value( data.get(), 2.0);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 2.0);
    }
    {
        // const char* / IValue_string
        mi::base::Handle<mi::neuraylib::IValue_string> data( vf->create_string( "foo"));
        const char* value = 0;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( value, "foo");
        result = set_value( data.get(), "bar");
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL_CSTR( value, "bar");
    }
    {
        // mi::math::Vector<mi::Sint32,3> / IValue_vector
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_vector(
            tf->create_vector( type_int.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue_vector> data( vf->create_vector( type_vector.get()));
        mi::math::Vector<mi::Sint32,3> value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == (mi::math::Vector<mi::Sint32,3>( 0)));
        result = set_value( data.get(), mi::math::Vector<mi::Sint32,3>( 4, 5, 6));
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == (mi::math::Vector<mi::Sint32,3>( 4, 5, 6)));
    }
    {
        // mi::math::Matrix<mi::Float32,3,2> / IValue_matrix
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_float( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_vector(
            tf->create_vector( type_float.get(), 2));
        mi::base::Handle<const mi::neuraylib::IType_matrix> type_matrix(
            tf->create_matrix( type_vector.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue_matrix> data( vf->create_matrix( type_matrix.get()));
        mi::math::Matrix<mi::Float32,3,2> value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == (mi::math::Matrix<mi::Float32,3,2>( 0)));
        mi::math::Matrix<mi::Float32,3,2> m( 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
        result = set_value( data.get(), m);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == m);
    }
    {
        // mi::math::Color / IValue_color
        mi::base::Handle<mi::neuraylib::IValue_color> data( vf->create_color( 1.0f, 2.0f, 3.0f));
        mi::math::Color value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Color( 1.0f, 2.0f, 3.0f));
        result = set_value( data.get(), mi::math::Color( 4.0f, 5.0f, 6.0f));
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Color( 4.0f, 5.0f, 6.0f));
    }
    {
        // mi::math::Spectrum / IValue_color
        mi::base::Handle<mi::neuraylib::IValue_color> data( vf->create_color( 1.0f, 2.0f, 3.0f));
        mi::math::Spectrum value;
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Spectrum( 1.0f, 2.0f, 3.0f));
        result = set_value( data.get(), mi::math::Spectrum( 4.0f, 5.0f, 6.0f));
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( value == mi::math::Spectrum( 4.0f, 5.0f, 6.0f));
    }
    {
        // mi::Sint32[3] / IValue_array
        mi::base::Handle<const mi::neuraylib::IType> type_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType_array> type_array(
            tf->create_immediate_sized_array( type_int.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue_array> data( vf->create_array( type_array.get()));
        mi::Sint32 value = -1;
        result = get_value( data.get(), zero_size, value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 0);
        result = set_value( data.get(), zero_size, 42);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), zero_size, value);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value, 42);
        result = get_value( data.get(), 3, value);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), 3, 42);
        MI_CHECK_EQUAL( result, -3);

        int value2[3] = { 43, 44, 45 };
        result = set_value( data.get(), value2, 3);
        MI_CHECK_EQUAL( result, 0);
        int value3[3];
        result = get_value( data.get(), value3, 3);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value2[0], value3[0]);
        MI_CHECK_EQUAL( value2[1], value3[1]);
        MI_CHECK_EQUAL( value2[2], value3[2]);

    }
    {
        // (mi::Sint32,mi::Float32) / IValue_struct
        mi::base::Handle<const mi::neuraylib::IType_struct> type_struct(
            tf->create_struct( "::" TEST_MDL "::foo_struct"));
        mi::base::Handle<mi::neuraylib::IValue_struct> data( vf->create_struct( type_struct.get()));
        mi::Sint32 value_int = -1;
        mi::Float32 value_float = -1.0f;
        result = get_value( data.get(), zero_size, value_int);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_int, 0);
        result = get_value( data.get(), 1, value_float);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_float, 0.0f);
        result = set_value( data.get(), "param_int", 42);
        MI_CHECK_EQUAL( result, 0);
        result = set_value( data.get(), "param_float", -42.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), "param_int", value_int);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_int, 42);
        result = get_value( data.get(), "param_float", value_float);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_float, -42.0f);
        result = set_value( data.get(), zero_size, 43);
        MI_CHECK_EQUAL( result, 0);
        result = set_value( data.get(), 1, -43.0f);
        MI_CHECK_EQUAL( result, 0);
        result = get_value( data.get(), zero_size, value_int);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_int, 43);
        result = get_value( data.get(), 1, value_float);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( value_float, -43.0f);
        result = get_value( data.get(), 2, value_int);
        MI_CHECK_EQUAL( result, -3);
        result = get_value( data.get(), zero_string, value_int);
        MI_CHECK_EQUAL( result, -3);
        result = get_value( data.get(), "invalid", value_int);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), 2, 42);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), zero_string, 42);
        MI_CHECK_EQUAL( result, -3);
        result = set_value( data.get(), "invalid", 42);
        MI_CHECK_EQUAL( result, -3);
    }
}

void check_wrappers(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // check a material definition (non-array parameter/argument)
    {
        mi::neuraylib::Definition_wrapper dw_fail( transaction, "invalid", 0);
        MI_CHECK( !dw_fail.is_valid());

        mi::neuraylib::Definition_wrapper dw(
            transaction, "mdl::" TEST_MDL "::md_1(color)", mdl_factory);
        MI_CHECK( dw.is_valid());

        MI_CHECK_EQUAL( dw.get_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);
        MI_CHECK( dw.is_material());
        MI_CHECK_EQUAL_CSTR( dw.get_mdl_definition(),
            "::" TEST_MDL "::md_1(color)");
        MI_CHECK_EQUAL_CSTR( dw.get_module(), "mdl::" TEST_MDL);

        MI_CHECK_EQUAL( 1, dw.get_parameter_count());
        MI_CHECK_EQUAL_CSTR( dw.get_parameter_name( 0), "tint");
        MI_CHECK( !dw.get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            dw.get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> type( parameter_types->get_type( "tint"));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_COLOR, type->get_kind());
        MI_CHECK( !parameter_types->get_type( "invalid"));
        MI_CHECK( !parameter_types->get_type( zero_string));
        type = dw.get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_STRUCT, type->get_kind());

        mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( dw.get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression> default_(
            defaults->get_expression( "tint"));
        MI_CHECK( default_);
        mi::math::Color c;
        MI_CHECK_EQUAL( 0, dw.get_default( "tint", c));
        MI_CHECK( c == mi::math::Color( 1.0f, 1.0f, 1.0f));

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> av( dw.get_annotations());
        MI_CHECK( !av);
        mi::base::Handle<const mi::neuraylib::IAnnotation_list> al( dw.get_parameter_annotations());
        MI_CHECK_EQUAL( 0, al->get_size());
        MI_CHECK( !al->get_annotation_block( zero_size));
        MI_CHECK( !al->get_annotation_block( "tint"));
        MI_CHECK( !al->get_annotation_block( "invalid"));
        MI_CHECK( !al->get_annotation_block( zero_string));
        av = dw.get_return_annotations();
        MI_CHECK( !av);

        mi::base::Handle<mi::neuraylib::IFunction_call> mi(
            dw.create_instance<mi::neuraylib::IFunction_call>());
        MI_CHECK_EQUAL( 0, transaction->store( mi.get(), "mdl::" TEST_MDL "::mi_1_wrapper"));
    }

    // check a material instance (non-array parameter/argument)
    {
        mi::neuraylib::Argument_editor ae_fail( transaction, "invalid", 0);
        MI_CHECK( !ae_fail.is_valid());

        mi::neuraylib::Argument_editor ae( transaction, "mdl::" TEST_MDL "::mi_1_wrapper", mdl_factory);
        MI_CHECK( ae.is_valid());

        MI_CHECK_EQUAL( ae.get_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);
        MI_CHECK( ae.is_material());
        MI_CHECK_EQUAL_CSTR( ae.get_definition(),
            "mdl::" TEST_MDL "::md_1(color)");
        MI_CHECK_EQUAL_CSTR( ae.get_mdl_definition(),
            "::" TEST_MDL "::md_1(color)");

        MI_CHECK_EQUAL( 1, ae.get_parameter_count());
        MI_CHECK_EQUAL_CSTR( ae.get_parameter_name( 0), "tint");
        MI_CHECK( !ae.get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType> type( ae.get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_STRUCT, type->get_kind());

        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( "tint"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( zero_size));

        mi::math::Color c;
        result = ae.get_value( "tint", c);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( c == mi::math::Color( 1.0f));
        result = ae.set_value( "tint", mi::math::Color( 2.0f));
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( "tint", c);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( c == mi::math::Color( 2.0f, 2.0f, 2.0f));

        result = ae.reset_argument( "tint");
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( "tint", c);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( c == mi::math::Color( 1.0f, 1.0f, 1.0f));

        MI_CHECK_EQUAL( 0, ae.set_call( "tint", "mdl::" TEST_MDL "::fc_uniform"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CALL, ae.get_argument_kind( "tint"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CALL, ae.get_argument_kind( zero_size));
        MI_CHECK_EQUAL_CSTR( ae.get_call( "tint"), "mdl::" TEST_MDL "::fc_uniform");

        result = ae.set_value( "tint", mi::math::Color( 3.0f));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( "tint"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( zero_size));
        result = ae.get_value( "tint", c);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK( c == mi::math::Color( 3.0f, 3.0f, 3.0f));
    }

    // check a function definition (array parameter/argument)
    {
        mi::neuraylib::Definition_wrapper dw_fail( transaction, "invalid", 0);
        MI_CHECK( !dw_fail.is_valid());

        mi::neuraylib::Definition_wrapper dw(
            transaction, "mdl::" TEST_MDL "::fd_deferred(int[N])", mdl_factory);
        MI_CHECK( dw.is_valid());

        MI_CHECK_EQUAL( dw.get_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);
        MI_CHECK( !dw.is_material());
        MI_CHECK_EQUAL_CSTR( dw.get_mdl_definition(), "::" TEST_MDL "::fd_deferred(int[N])");
        MI_CHECK_EQUAL_CSTR( dw.get_module(), "mdl::" TEST_MDL);

        MI_CHECK_EQUAL( 1, dw.get_parameter_count());
        MI_CHECK_EQUAL_CSTR( dw.get_parameter_name( 0), "param0");
        MI_CHECK( !dw.get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            dw.get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> type( parameter_types->get_type( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
        MI_CHECK( !parameter_types->get_type( "invalid"));
        MI_CHECK( !parameter_types->get_type( zero_string));
        type = dw.get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());

        mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( dw.get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression> default_(
            defaults->get_expression( "param0"));
        MI_CHECK( default_);

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> av( dw.get_annotations());
        MI_CHECK( !av);
        mi::base::Handle<const mi::neuraylib::IAnnotation_list> al( dw.get_parameter_annotations());
        MI_CHECK_EQUAL( 0, al->get_size());
        MI_CHECK( !al->get_annotation_block( zero_size));
        MI_CHECK( !al->get_annotation_block( "param0"));
        MI_CHECK( !al->get_annotation_block( "invalid"));
        MI_CHECK( !al->get_annotation_block( zero_string));
        av = dw.get_return_annotations();
        MI_CHECK( !av);

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            dw.create_instance<mi::neuraylib::IFunction_call>());
        MI_CHECK_EQUAL( 0, transaction->store(
            fc.get(), "mdl::" TEST_MDL "::fd_deferred(int[N])_wrapper"));
    }

    // check a function call (array parameter/argument)
    {
        mi::neuraylib::Argument_editor ae_fail( transaction, "invalid", 0);
        MI_CHECK( !ae_fail.is_valid());

        mi::neuraylib::Argument_editor ae(
            transaction, "mdl::" TEST_MDL "::fd_deferred(int[N])_wrapper", mdl_factory);
        MI_CHECK( ae.is_valid());

        MI_CHECK_EQUAL( ae.get_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);
        MI_CHECK( !ae.is_material());
        MI_CHECK_EQUAL_CSTR( ae.get_definition(), "mdl::" TEST_MDL "::fd_deferred(int[N])");
        MI_CHECK_EQUAL_CSTR( ae.get_mdl_definition(), "::" TEST_MDL "::fd_deferred(int[N])");

        MI_CHECK_EQUAL( 1, ae.get_parameter_count());
        MI_CHECK_EQUAL_CSTR( ae.get_parameter_name( 0), "param0");
        MI_CHECK( !ae.get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType> type( ae.get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());

        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( zero_size));

        mi::Size length = 0;
        MI_CHECK_EQUAL( 0, ae.get_array_length( "param0", length));
        MI_CHECK_EQUAL( 2, length);
        MI_CHECK_EQUAL( 0, ae.set_array_size( "param0", 3));
        MI_CHECK_EQUAL( 0, ae.get_array_length( "param0", length));
        MI_CHECK_EQUAL( 3, length);

        result = ae.reset_argument( "param0");
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, ae.get_array_length( "param0", length));
        MI_CHECK_EQUAL( 2, length);

        // set/get individual array elements
        mi::Sint32 s = 1;
        result = ae.get_value( "param0", zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, s);
        result = ae.set_value( "param0", zero_size, 42);
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( "param0", zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 42, s);
        s = 1;
        result = ae.get_value( zero_size, zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 42, s);
        result = ae.set_value( zero_size, zero_size, 43);
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( zero_size, zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 43, s);

        result = ae.get_value( "param0", 3, s);
        MI_CHECK_EQUAL( -3, result);
        result = ae.set_value( "param0", 3, 42);
        MI_CHECK_EQUAL( -3, result);

        // set/get entire array in one call
        mi::Sint32 a[2];
        result = ae.get_value( "param0", a, 2);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( a[0], 43);
        MI_CHECK_EQUAL( a[1], 0);
        mi::Sint32 b[3] {44, 45, 46};
        result = ae.set_value( "param0", b, 3);
        MI_CHECK_EQUAL( 0, result);
        mi::Sint32 c[3];
        result = ae.get_value( "param0", c, 3);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( c[0], 44);
        MI_CHECK_EQUAL( c[1], 45);
        MI_CHECK_EQUAL( c[2], 46);
        result = ae.get_value( "param0", c, 2);
        MI_CHECK_EQUAL( -5, result);

        mi::Sint32 e[3];
        result = ae.get_value( zero_size, e, 3);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( e[0], 44);
        MI_CHECK_EQUAL( e[1], 45);
        MI_CHECK_EQUAL( e[2], 46);
        mi::Sint32 f[2] {47, 48};
        result = ae.set_value( zero_size, f, 2);
        MI_CHECK_EQUAL( 0, result);
        mi::Sint32 g[2];
        result = ae.get_value( "param0", g, 2);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( g[0], 47);
        MI_CHECK_EQUAL( g[1], 48);
        result = ae.get_value( "param0", g, 3);
        MI_CHECK_EQUAL( -5, result);

        MI_CHECK_EQUAL( 0, ae.set_call( "param0", "mdl::" TEST_MDL "::fc_deferred_constant"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CALL, ae.get_argument_kind( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CALL, ae.get_argument_kind( zero_size));
        MI_CHECK_EQUAL_CSTR( ae.get_call( "param0"), "mdl::" TEST_MDL "::fc_deferred_constant");

        result = ae.set_value( "param0", zero_size, 44);
        MI_CHECK_EQUAL( -3, result);
        result = ae.set_array_size( "param0", 1);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( "param0"));
        MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, ae.get_argument_kind( zero_size));
        result = ae.set_value( "param0", zero_size, 45);
        MI_CHECK_EQUAL( 0, result);
        result = ae.get_value( "param0", zero_size, s);
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 45, s);
    }

    // check function definitions with generic types
    {
        const char* names[] = {
            "mdl::T[](...)",
            "mdl::operator_len(%3C0%3E[])",
            "mdl::operator[](%3C0%3E[],int)",
            "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)",
            "mdl::operator_cast(%3C0%3E)",
        };

        for( const auto& name: names) {
            mi::neuraylib::Definition_wrapper dw(
                transaction,  name, mdl_factory);
            MI_CHECK( dw.is_valid());
            mi::base::Handle<mi::neuraylib::IFunction_call> fc(
                dw.create_instance<mi::neuraylib::IFunction_call>());
            MI_CHECK( !fc);
        }
    }

    // check instantiation/argument reset with argument taken from range annotation
    {
        mi::neuraylib::Definition_wrapper dw(
            transaction, "mdl::" TEST_MDL "::md_with_annotations(color,float)", mdl_factory);
        MI_CHECK( dw.is_valid());

        mi::base::Handle<mi::neuraylib::IFunction_call> mi(
            dw.create_instance<mi::neuraylib::IFunction_call>());
        MI_CHECK( mi);
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( mi->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> param1_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "param1"));
        mi::base::Handle<const mi::neuraylib::IValue_float> param1_value(
            param1_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_EQUAL( param1_value->get_value(), 1.0f);

        // change argument
        mi::base::Handle<mi::neuraylib::IValue_float> value( vf->create_float( 2.0f));
        mi::base::Handle<mi::neuraylib::IExpression> expr( ef->create_constant( value.get()));
        MI_CHECK_EQUAL( 0, mi->set_argument( "param1", expr.get()));

        MI_CHECK_EQUAL( 0, transaction->store( mi.get(), "mi_with_annotations"));

        mi::neuraylib::Argument_editor ae(
            transaction, "mi_with_annotations", mdl_factory);
        MI_CHECK( ae.is_valid());

        mi::Float32 param1 = 0.0f;
        MI_CHECK_EQUAL( 0, ae.get_value( "param1", param1));
        MI_CHECK_EQUAL( param1, 2.0f);

        // reset argument
        MI_CHECK_EQUAL( 0, ae.reset_argument( "param1"));
        MI_CHECK_EQUAL( 0, ae.get_value( "param1", param1));
        MI_CHECK_EQUAL( param1, 1.0f);
    }

    // check instantiation/argument reset with argument taken from range annotation
    {
        mi::neuraylib::Definition_wrapper dw(
            transaction, "mdl::" TEST_MDL "::fd_with_annotations(color,float)", mdl_factory);
        MI_CHECK( dw.is_valid());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            dw.create_instance<mi::neuraylib::IFunction_call>());
        MI_CHECK( fc);
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> param1_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "param1"));
        mi::base::Handle<const mi::neuraylib::IValue_float> param1_value(
            param1_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_EQUAL( param1_value->get_value(), 1.0f);

        // change argument
        mi::base::Handle<mi::neuraylib::IValue_float> value( vf->create_float( 2.0f));
        mi::base::Handle<mi::neuraylib::IExpression> expr( ef->create_constant( value.get()));
        MI_CHECK_EQUAL( 0, fc->set_argument( "param1", expr.get()));

        MI_CHECK_EQUAL( 0, transaction->store( fc.get(), "fc_with_annotations"));

        mi::neuraylib::Argument_editor ae(
            transaction, "fc_with_annotations", mdl_factory);
        MI_CHECK( ae.is_valid());

        mi::Float32 param1 = 0.0f;
        MI_CHECK_EQUAL( 0, ae.get_value( "param1", param1));
        MI_CHECK_EQUAL( param1, 2.0f);

        // reset argument
        MI_CHECK_EQUAL( 0, ae.reset_argument( "param1"));
        MI_CHECK_EQUAL( 0, ae.get_value( "param1", param1));
        MI_CHECK_EQUAL( param1, 1.0f);
    }
}

void check_uniform_auto_varying(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    {
        mi::base::Handle<const mi::neuraylib::IType> type;
        mi::base::Handle<const mi::neuraylib::IType_list> types;
        mi::base::Handle<const mi::neuraylib::IExpression> arg;
        mi::base::Handle<const mi::neuraylib::IExpression_list> args;

        // check properties
        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_parameters_uniform_auto_varying(int,int,int)"));
        types = c_md->get_parameter_types();

        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( "param0_uniform");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( "param1_auto");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( 2);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());
        type = types->get_type( "param2_varying");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());

        mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi(
            transaction->access<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::mi_parameters_uniform_auto_varying"));
        args = c_mi->get_arguments();

        arg = args->get_expression( zero_size);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param0_uniform");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 1);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param1_auto");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 2);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param2_varying");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying(int,int,int)"));
        types = c_fd->get_parameter_types();

        type = types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( "param0_uniform");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());
        type = types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( "param1_auto");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        type = types->get_type( 2);
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());
        type = types->get_type( "param2_varying");
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());

        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_return_uniform()");
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());

        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_return_auto()");
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_return_varying()");
        type = c_fd->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());

        mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
            transaction->access<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::fc_parameters_uniform_auto_varying"));
        args = c_mi->get_arguments();

        arg = args->get_expression( zero_size);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param0_uniform");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 1);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param1_auto");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( 2);
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());
        arg = args->get_expression( "param2_varying");
        type = arg->get_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        c_fc = transaction->access<mi::neuraylib::IFunction_call>(
            "mdl::" TEST_MDL "::fc_return_uniform");
        type = c_fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, type->get_all_type_modifiers());

        c_fc = transaction->access<mi::neuraylib::IFunction_call>(
            "mdl::" TEST_MDL "::fc_return_auto");
        type = c_fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_NONE, type->get_all_type_modifiers());

        c_fc = transaction->access<mi::neuraylib::IFunction_call>(
            "mdl::" TEST_MDL "::fc_return_varying");
        type = c_fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::MK_VARYING, type->get_all_type_modifiers());
    }

    {
        // check set_argument() -- reject varying arguments for uniform parameters
        mi::base::Handle<mi::neuraylib::IExpression_call> expr_uniform(
            ef->create_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        mi::base::Handle<mi::neuraylib::IExpression_call> expr_auto(
            ef->create_call( "mdl::" TEST_MDL "::fc_return_auto"));
        mi::base::Handle<mi::neuraylib::IExpression_call> expr_varying(
            ef->create_call( "mdl::" TEST_MDL "::fc_return_varying"));

        mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
            transaction->edit<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::mi_parameters_uniform_auto_varying"));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( zero_size, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param0_uniform", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( zero_size, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param0_uniform", expr_auto.get()));
        MI_CHECK_EQUAL( -5, m_mi->set_argument( zero_size, expr_varying.get()));
        MI_CHECK_EQUAL( -5, m_mi->set_argument( "param0_uniform", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_mi->set_argument( 1, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param1_auto", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 1, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param1_auto", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 1, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param1_auto", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_mi->set_argument( 2, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param2_varying", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 2, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param2_varying", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( 2, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_mi->set_argument( "param2_varying", expr_varying.get()));

        mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
            transaction->edit<mi::neuraylib::IFunction_call>(
                "mdl::" TEST_MDL "::fc_parameters_uniform_auto_varying"));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( zero_size, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0_uniform", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( zero_size, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0_uniform", expr_auto.get()));
        MI_CHECK_EQUAL( -5, m_fc->set_argument( zero_size, expr_varying.get()));
        MI_CHECK_EQUAL( -5, m_fc->set_argument( "param0_uniform", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_fc->set_argument( 1, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param1_auto", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 1, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param1_auto", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 1, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param1_auto", expr_varying.get()));

        MI_CHECK_EQUAL( 0, m_fc->set_argument( 2, expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param2_varying", expr_uniform.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 2, expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param2_varying", expr_auto.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( 2, expr_varying.get()));
        MI_CHECK_EQUAL( 0, m_fc->set_argument( "param2_varying", expr_varying.get()));

        // check set_call() -- reject all modifier variations
        MI_CHECK_EQUAL(  0, expr_uniform->set_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        MI_CHECK_EQUAL( -4, expr_uniform->set_call( "mdl::" TEST_MDL "::fc_return_auto"));
        MI_CHECK_EQUAL( -4, expr_uniform->set_call( "mdl::" TEST_MDL "::fc_return_varying"));
        MI_CHECK_EQUAL( -4, expr_auto->set_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        MI_CHECK_EQUAL(  0, expr_auto->set_call( "mdl::" TEST_MDL "::fc_return_auto"));
        MI_CHECK_EQUAL( -4, expr_auto->set_call( "mdl::" TEST_MDL "::fc_return_varying"));
        MI_CHECK_EQUAL( -4, expr_varying->set_call( "mdl::" TEST_MDL "::fc_return_uniform"));
        MI_CHECK_EQUAL( -4, expr_varying->set_call( "mdl::" TEST_MDL "::fc_return_auto"));
        MI_CHECK_EQUAL(  0, expr_varying->set_call( "mdl::" TEST_MDL "::fc_return_varying"));
    }

#if 0 // this can only be triggered by overwriting DB elements
    {
        // check compilation of material where a uniform parameter has a varying function with auto
        // return type (=> treated as varying return type according to spec) attached
        result = 0;
        mi::base::Handle<const mi::neuraylib::IFunction_call> mi(
            transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_parameter_uniform"));
        mi::base::Handle<mi::neuraylib::ICompiled_material> cm(
            mi_mi->create_compiled_material(
                mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, 1.0f, 380.0f, 780.0f, &result));
        MI_CHECK_EQUAL( -3, result);
        MI_CHECK( !cm);
    }
#endif
}

void check_export_flag( mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
    {
        // non-exported function definition
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_non_exported()"));
        MI_CHECK( fd);

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( 0, &result));
        MI_CHECK( !fc);
        MI_CHECK_EQUAL( -4, result);
    }
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    {
        // material that uses a non-exported function definition
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_uses_non_exported_function()"));
        MI_CHECK( md);

        mi::base::Handle<mi::neuraylib::IFunction_call> mi(
            md->create_function_call( 0, &result));
        MI_CHECK( mi);
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<mi::neuraylib::IMaterial_instance> mi_mi(
            mi->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<mi::neuraylib::ICompiled_material> cm(
            mi_mi->create_compiled_material(
                mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( cm);
    }
    {
        // material that uses a non-exported material definition
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_uses_non_exported_material()"));
        MI_CHECK( md);

        mi::base::Handle<mi::neuraylib::IFunction_call> mi(
            md->create_function_call( 0, &result));
        MI_CHECK( mi);
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<mi::neuraylib::IMaterial_instance> mi_mi(
            mi->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<mi::neuraylib::ICompiled_material> cm(
            mi_mi->create_compiled_material(
                mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( cm);
    }
}

// check that references held by the MDL module are still valid
// (not contained in get_scene_element_references() to avoid cyclic links)
void check_module_references( mi::neuraylib::IDatabase* database)
{
    mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());

    // check that "mdl::" TEST_MDL "::fd_remove(int)" exists and schedule it for removal
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_remove(int)"));
    MI_CHECK( fd);
    fd = 0;
    MI_CHECK_EQUAL( 0, transaction->remove( "mdl::" TEST_MDL "::fd_remove(int)"));
    MI_CHECK_EQUAL( 0, transaction->commit());

    // the GC should not remove "mdl::" TEST_MDL "::fd_remove(int)" (it is supposed to be referenced by its module)
    database->garbage_collection();

    // check that "mdl::" TEST_MDL "::fd_remove(int)" is not gone
    transaction = scope->create_transaction();
    fd = transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_remove(int)");
    MI_CHECK( fd);

    // iterate over all functions and materials of "mdl::" TEST_MDL and check that
    // the DB elements still exists
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
    MI_CHECK( module);
    mi::Size function_count = module->get_function_count();
    for( mi::Size i = 0; i < function_count; ++i) {
        const char* name = module->get_function( i);
        fd = transaction->access<mi::neuraylib::IFunction_definition>( name);
        MI_CHECK( fd);
    }
    fd = 0;
    mi::Size material_count = module->get_material_count();
    for( mi::Size i = 0; i < material_count; ++i) {
        const char* name = module->get_material( i);
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>( name));
        MI_CHECK( md);
    }
    module = 0;

    MI_CHECK_EQUAL( 0, transaction->commit());
}

void check_backends(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::Float32_4_4 m(
        0.0f, 0.1f, 0.2f, 0.3f,
        1.0f, 1.1f, 1.2f, 1.3f,
        2.0f, 2.1f, 2.2f, 2.3f,
        3.0f, 3.1f, 3.2f, 3.3f
    );

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_llvm(
        mdl_backend_api->get_backend( mi::neuraylib::IMdl_backend_api::MB_LLVM_IR));
    MI_CHECK( be_llvm);

    MI_CHECK_EQUAL( 0, be_llvm->set_option( "num_texture_spaces", "16"));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_ptx(
        mdl_backend_api->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
    MI_CHECK( be_ptx);

    MI_CHECK_EQUAL( 0, be_ptx->set_option( "num_texture_spaces", "16"));
    MI_CHECK_EQUAL( 0, be_ptx->set_option( "sm_version", "50"));
    MI_CHECK_EQUAL( 0, be_ptx->set_option( "output_format", "PTX"));

    mi::Size size = 0;
    MI_CHECK( be_ptx->get_device_library( size));
    MI_CHECK( size);

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_glsl(
        mdl_backend_api->get_backend( mi::neuraylib::IMdl_backend_api::MB_GLSL));
    MI_CHECK( be_glsl);

    MI_CHECK_EQUAL( 0, be_glsl->set_option( "glsl_version", "450"));

    // place constants into the SSBO segment
    MI_CHECK_EQUAL( 0, be_glsl->set_option( "glsl_max_const_data", "0"));
    MI_CHECK_EQUAL( 0, be_glsl->set_option( "glsl_place_uniforms_into_ssbo", "on"));

    MI_CHECK_EQUAL( 0, be_glsl->set_option( "glsl_remap_functions",
        "_ZN4base12file_textureEu10texture_2du5coloru5colorN4base9mono_modeEN4base23"
        "texture_coordinate_infoEu6float2u6float2N3tex9wrap_modeEN3tex9wrap_modeEb"
        " = my_file_texture"));

    mi::Sint32 errors = -1;

    mi::Uint32 flags_instance_compilation = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::Uint32 flags_class_compilation    = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // compile material displacement to LLVM, PTX, GLSL
    {
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
            mi->create_compiled_material( flags_instance_compilation, context.get()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_cc(
            mi->create_compiled_material( flags_class_compilation, context.get()));

        // LLVM, WITH SIMD, no RO segment
        errors = -1;

        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_simd", "on"));
        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_ro_segment", "off"));

        // LLVM, WITH SIMD, WITH RO segment
        errors = -1;

        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_simd", "on"));
        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_ro_segment", "on"));

        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_llvm_simd(
                be_llvm->translate_material_expression(
                transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
            MI_CHECK( code_llvm_simd);
            MI_CHECK_CTX( context.get());
            MI_CHECK( code_llvm_simd->get_code());
            MI_CHECK( code_llvm_simd->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_llvm_simd->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "displacement", code_llvm_simd->get_callable_function( 0));

            MI_CHECK_EQUAL( 0, code_llvm_simd->get_texture_count());

            MI_CHECK_EQUAL( 1, code_llvm_simd->get_ro_data_segment_count());
            MI_CHECK_EQUAL_CSTR( "RO", code_llvm_simd->get_ro_data_segment_name( 0));
            MI_CHECK_EQUAL( 1152, code_llvm_simd->get_ro_data_segment_size( 0));
            MI_CHECK_EQUAL( 0.0f, ((float *)code_llvm_simd->get_ro_data_segment_data(0))[0]);
            MI_CHECK_EQUAL( 1.0f, ((float *)code_llvm_simd->get_ro_data_segment_data(0))[1]);
            MI_CHECK_EQUAL( 2.0f, ((float *)code_llvm_simd->get_ro_data_segment_data(0))[2]);

            MI_CHECK_EQUAL( 0, code_llvm_simd->get_code_segment_count());

            MI_CHECK_EQUAL( 3, code_llvm_simd->get_string_constant_count());
            {
                int i = 1;
                MI_CHECK_EQUAL_CSTR("something", code_llvm_simd->get_string_constant(i)); ++i;
                MI_CHECK_EQUAL_CSTR("abc",       code_llvm_simd->get_string_constant(i)); ++i;
            }
        }

        // PTX, WITH RO segment
        errors = -1;

        MI_CHECK_EQUAL( 0, be_ptx->set_option( "enable_ro_segment", "off"));

        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx(
                be_ptx->translate_material_expression(
                transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
            MI_CHECK( code_ptx);
            MI_CHECK_CTX( context.get());
            MI_CHECK( code_ptx->get_code());
            MI_CHECK( code_ptx->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_ptx->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "displacement", code_ptx->get_callable_function( 0));

            MI_CHECK_EQUAL( 0, code_ptx->get_texture_count());

            MI_CHECK_EQUAL( 0, code_ptx->get_ro_data_segment_count());

            MI_CHECK_EQUAL( 0, code_ptx->get_code_segment_count());

            MI_CHECK_EQUAL( 3, code_ptx->get_string_constant_count());
            {
                int i = 1;
                MI_CHECK_EQUAL_CSTR("something", code_ptx->get_string_constant(i)); ++i;
                MI_CHECK_EQUAL_CSTR("abc",       code_ptx->get_string_constant(i)); ++i;
            }
        }

        // PTX, code segment on
        errors = -1;

        MI_CHECK_EQUAL( 0, be_ptx->set_option( "enable_ro_segment", "on"));

        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx(
                be_ptx->translate_material_expression(
                transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
            MI_CHECK( code_ptx);
            MI_CHECK_CTX( context.get());
            MI_CHECK( code_ptx->get_code());
            MI_CHECK( code_ptx->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_ptx->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "displacement", code_ptx->get_callable_function( 0));

            MI_CHECK_EQUAL( 0, code_ptx->get_texture_count());

            MI_CHECK_EQUAL( 1, code_ptx->get_ro_data_segment_count());
            MI_CHECK_EQUAL_CSTR( "RO", code_ptx->get_ro_data_segment_name( 0));
            MI_CHECK_EQUAL( 1152, code_ptx->get_ro_data_segment_size( 0));
            MI_CHECK_EQUAL( 0.0f, ((float *)code_ptx->get_ro_data_segment_data(0))[0]);
            MI_CHECK_EQUAL( 1.0f, ((float *)code_ptx->get_ro_data_segment_data(0))[1]);
            MI_CHECK_EQUAL( 2.0f, ((float *)code_ptx->get_ro_data_segment_data(0))[2]);

            MI_CHECK_EQUAL( 0, code_ptx->get_code_segment_count());

            MI_CHECK_EQUAL( 3, code_ptx->get_string_constant_count());
            {
                int i = 1;
                MI_CHECK_EQUAL_CSTR("something", code_ptx->get_string_constant(i)); ++i;
                MI_CHECK_EQUAL_CSTR("abc",       code_ptx->get_string_constant(i)); ++i;
            }
        }

        // GLSL, instance compilation
        errors = -1;
        mi::base::Handle<const mi::neuraylib::ITarget_code> code_glsl(
            be_glsl->translate_material_expression(
                transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
        MI_CHECK( code_glsl);
        MI_CHECK_CTX( context.get());
        MI_CHECK( code_glsl->get_code());
        MI_CHECK( code_glsl->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code_glsl->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code_glsl->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code_glsl->get_texture_count());

        MI_CHECK_EQUAL( 0, code_glsl->get_code_segment_count());
#if 0 // not yet supported
        MI_CHECK_EQUAL( 7, code_glsl->get_code_segment_count());
        {
            size_t i = 0;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "version", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "extensions", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) == 0);
            MI_CHECK_EQUAL_CSTR( "defines", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment(i));
            MI_CHECK( code_glsl->get_code_segment_size(i) > 0);
            MI_CHECK_EQUAL_CSTR( "state", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment(i));
            // MI_CHECK( code_glsl->get_code_segment_size(i) >= 0);
            MI_CHECK_EQUAL_CSTR( "structs", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "prototypes", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "functions", code_glsl->get_code_segment_description( i));
            ++i;
        }
#endif

        MI_CHECK_EQUAL( 3, code_glsl->get_string_constant_count());
        {
            int i = 1;
            MI_CHECK_EQUAL_CSTR("something", code_glsl->get_string_constant(i)); ++i;
            MI_CHECK_EQUAL_CSTR("abc",       code_glsl->get_string_constant(i)); ++i;
        }

        // GLSL, class compilation
        errors = -1;
        code_glsl = be_glsl->translate_material_expression(
            transaction, cm_cc.get(), "geometry.displacement", "displacement", context.get());
        MI_CHECK( code_glsl);
        MI_CHECK_CTX( context.get());
        MI_CHECK( code_glsl->get_code());
        MI_CHECK( code_glsl->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code_glsl->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code_glsl->get_callable_function( 0));

        // MI_CHECK_EQUAL( 0, code_glsl->get_texture_count());

        MI_CHECK_EQUAL( 1, code_glsl->get_ro_data_segment_count());
        MI_CHECK_EQUAL_CSTR( "mdl_buffer", code_glsl->get_ro_data_segment_name( 0));
        MI_CHECK_EQUAL( 1152, code_glsl->get_ro_data_segment_size( 0));
        MI_CHECK_EQUAL( 0.0f, ((float *)code_glsl->get_ro_data_segment_data(0))[0]);
        MI_CHECK_EQUAL( 1.0f, ((float *)code_glsl->get_ro_data_segment_data(0))[1]);
        MI_CHECK_EQUAL( 2.0f, ((float *)code_glsl->get_ro_data_segment_data(0))[2]);

        MI_CHECK_EQUAL( 0, code_glsl->get_code_segment_count());
#if 0 // not yet supported
        MI_CHECK_EQUAL( 7, code_glsl->get_code_segment_count());
        {
            size_t i = 0;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "version", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "extensions", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) == 0);
            MI_CHECK_EQUAL_CSTR( "defines", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "state", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "structs", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "prototypes", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "functions", code_glsl->get_code_segment_description( i));
            ++i;
        }
#endif

        MI_CHECK_EQUAL( 2, code_glsl->get_string_constant_count());
        {
            int i = 0;
            MI_CHECK_EQUAL_CSTR("abc",       code_glsl->get_string_constant(++i));
        }
    }

    // compile material part to LLVM, PTX, GLSL
    {
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
            mi->create_compiled_material( flags_instance_compilation));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_cc(
            mi->create_compiled_material( flags_class_compilation));

        // LLVM, no SIMD, no RO segment
        errors = -1;

        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_simd", "off"));
        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_ro_segment", "off"));

        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_llvm_scalar(
                be_llvm->translate_material_expression(
                    transaction,
                    cm_ic.get(),
                    "surface.scattering.components.value0.component.tint",
                    "tint",
                    context.get()));
            MI_CHECK( code_llvm_scalar);
            MI_CHECK_CTX( context.get());
            MI_CHECK( code_llvm_scalar->get_code());
            MI_CHECK( code_llvm_scalar->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_llvm_scalar->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "tint", code_llvm_scalar->get_callable_function( 0));

            MI_CHECK_EQUAL( 2, code_llvm_scalar->get_texture_count());
            MI_CHECK_EQUAL_CSTR( "", code_llvm_scalar->get_texture(0));
            MI_CHECK_NOT_EQUAL( 0, code_llvm_scalar->get_texture( 1));

            MI_CHECK_EQUAL( 0, code_llvm_scalar->get_ro_data_segment_count());

            MI_CHECK_EQUAL( 0, code_llvm_scalar->get_code_segment_count());

            MI_CHECK_EQUAL( 0, code_llvm_scalar->get_string_constant_count());
        }

        // PTX

        MI_CHECK_EQUAL( 0, be_ptx->set_option( "enable_ro_segment", "off"));

        // switch to LLVM-IR output
        MI_CHECK_EQUAL( 0, be_ptx->set_option( "output_format", "LLVM-IR"));

        errors = -1;
        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx(
                be_ptx->translate_material_expression(
                    transaction,
                    cm_ic.get(),
                    "surface.scattering.components.value0.component.tint",
                    "tint",
                    context.get()));
            MI_CHECK( code_ptx);
            MI_CHECK_CTX( context.get());
            MI_CHECK( code_ptx->get_code());
            MI_CHECK( code_ptx->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_ptx->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "tint", code_ptx->get_callable_function( 0));

            MI_CHECK_EQUAL( 2, code_ptx->get_texture_count());
            MI_CHECK_EQUAL_CSTR( "", code_ptx->get_texture(0));
            MI_CHECK_NOT_EQUAL( 0, code_ptx->get_texture( 1));

            MI_CHECK_EQUAL( 0, code_ptx->get_ro_data_segment_count());

            MI_CHECK_EQUAL( 0, code_ptx->get_code_segment_count());

            MI_CHECK_EQUAL( 0, code_ptx->get_string_constant_count());
        }

        // GLSL, instance compilation
        errors = -1;
        mi::base::Handle<const mi::neuraylib::ITarget_code> code_glsl(
            be_glsl->translate_material_expression(
                transaction,
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint",
                context.get()));
        MI_CHECK( code_glsl);
        MI_CHECK_CTX( context.get());
        MI_CHECK( code_glsl->get_code());
        MI_CHECK( code_glsl->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code_glsl->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "tint", code_glsl->get_callable_function( 0));

        MI_CHECK_EQUAL( 2, code_glsl->get_texture_count());
        MI_CHECK_EQUAL_CSTR( "", code_glsl->get_texture( 0));
        MI_CHECK_NOT_EQUAL( 0, code_glsl->get_texture( 1));

        MI_CHECK_EQUAL( 0, code_glsl->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code_glsl->get_code_segment_count());
#if 0 // not yet supported
        MI_CHECK_EQUAL( 7, code_glsl->get_code_segment_count());
        {
            size_t i = 0;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "version", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "extensions", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "defines", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "state", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK(code_glsl->get_code_segment(i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR("structs", code_glsl->get_code_segment_description(i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "prototypes", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "functions", code_glsl->get_code_segment_description( i));
            ++i;
        }
#endif

        MI_CHECK_EQUAL( 0, code_glsl->get_string_constant_count());

        // GLSL, class compilation
        errors = -1;
        code_glsl = be_glsl->translate_material_expression(
            transaction,
            cm_cc.get(),
            "surface.scattering.components.value0.component.tint",
            "tint",
            context.get());
        MI_CHECK( code_glsl);
        MI_CHECK_CTX( context.get());
        MI_CHECK( code_glsl->get_code());
        MI_CHECK( code_glsl->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code_glsl->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "tint", code_glsl->get_callable_function( 0));

        MI_CHECK_EQUAL( 2, code_glsl->get_texture_count());
        MI_CHECK_EQUAL_CSTR( "", code_glsl->get_texture( 0));
        MI_CHECK_NOT_EQUAL( 0, code_glsl->get_texture( 1));

        MI_CHECK_EQUAL( 0, code_glsl->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code_glsl->get_code_segment_count());
#if 0 // not yet supported
        MI_CHECK_EQUAL( 7, code_glsl->get_code_segment_count());
        {
            size_t i = 0;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "version", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "extensions", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "defines", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "state", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "structs", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            // MI_CHECK( code_glsl->get_code_segment_size( i) >= 0);
            MI_CHECK_EQUAL_CSTR( "prototypes", code_glsl->get_code_segment_description( i));
            ++i;
            MI_CHECK( code_glsl->get_code_segment( i));
            MI_CHECK( code_glsl->get_code_segment_size( i) > 0);
            MI_CHECK_EQUAL_CSTR( "functions", code_glsl->get_code_segment_description( i));
            ++i;
        }
#endif
        MI_CHECK_EQUAL( 0, code_glsl->get_string_constant_count());
    }

    // compile environment to LLVM, PTX, GLSL
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::" TEST_MDL "::fd_0()"));
        MI_CHECK(fd_def);

        mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
            transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_jit"));
        MI_CHECK(fc);

        // LLVM, no SIMD, no RO segment
        errors = -1;

        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_simd", "off"));
        MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_ro_segment", "off"));

        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_llvm_scalar(
                be_llvm->translate_environment(
                transaction, fc.get(), "env", context.get()));
            MI_CHECK( code_llvm_scalar);
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);
            MI_CHECK( code_llvm_scalar->get_code());
            MI_CHECK( code_llvm_scalar->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_llvm_scalar->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "env", code_llvm_scalar->get_callable_function( 0));

            MI_CHECK_EQUAL( 0, code_llvm_scalar->get_texture_count());

            MI_CHECK_EQUAL( 0, code_llvm_scalar->get_string_constant_count());
        }

        // PTX

        MI_CHECK_EQUAL( 0, be_ptx->set_option( "enable_ro_segment", "off"));

        // switch to PTX output
        MI_CHECK_EQUAL( 0, be_ptx->set_option( "output_format", "PTX"));

        errors = -1;
        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx(
                be_ptx->translate_environment(
                transaction, fc.get(), "env", context.get()));
            MI_CHECK( code_ptx);
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);
            MI_CHECK( code_ptx->get_code());
            MI_CHECK( code_ptx->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_ptx->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "env", code_ptx->get_callable_function( 0));

            MI_CHECK_EQUAL( 0, code_ptx->get_texture_count());
        }

        // GLSL
        errors = -1;
        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> code_glsl(
                be_glsl->translate_environment(
                transaction, fc.get(), "env", context.get()));
            MI_CHECK( code_glsl);
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);
            MI_CHECK( code_glsl->get_code());
            MI_CHECK( code_glsl->get_code_size() > 0);

            MI_CHECK_EQUAL( 1, code_glsl->get_callable_function_count());
            MI_CHECK_EQUAL_CSTR( "env", code_glsl->get_callable_function( 0));

            MI_CHECK_EQUAL( 0, code_glsl->get_texture_count());

            MI_CHECK_EQUAL( 1, code_glsl->get_ro_data_segment_count());
            MI_CHECK_EQUAL_CSTR( "mdl_buffer", code_glsl->get_ro_data_segment_name( 0));
            MI_CHECK_EQUAL( 1152, code_glsl->get_ro_data_segment_size( 0));
        }

        // test link units: LLVM
        errors = -1;
        {
            mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
                transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
            mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
                mi->create_compiled_material( flags_instance_compilation));

            MI_CHECK_EQUAL( 0, be_llvm->set_option( "num_texture_spaces", "16"));
            MI_CHECK_EQUAL( 0, be_llvm->set_option( "enable_simd", "on"));

            mi::base::Handle<mi::neuraylib::ILink_unit> unit(
                be_llvm->create_link_unit( transaction, context.get()));
            MI_CHECK( unit.is_valid_interface());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

            errors = unit->add_material_expression(
                cm_ic.get(), "geometry.displacement", "displacement", context.get());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint_0", context.get());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value1.component.tint",
                "tint_1", context.get());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value2.component.tint",
                "tint_2", context.get());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

            errors = unit->add_function(
                fc.get(), mi::neuraylib::ILink_unit::FEC_ENVIRONMENT, "env", context.get());
            MI_CHECK_EQUAL( 0, errors);

            errors = unit->add_function(
                fd_def.get(), mi::neuraylib::ILink_unit::FEC_CORE, NULL, context.get());
            MI_CHECK_EQUAL(0, errors);

            mi::base::Handle<const mi::neuraylib::ITarget_code> code_llvm(
                be_llvm->translate_link_unit( unit.get(), context.get()));
            MI_CHECK( code_llvm.is_valid_interface());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);
            MI_CHECK( code_llvm->get_code());
            MI_CHECK( code_llvm->get_code_size() > 0);

            MI_CHECK_EQUAL(6, code_llvm->get_callable_function_count());
            MI_CHECK_EQUAL(4, code_llvm->get_texture_count());
            MI_CHECK_EQUAL( code_llvm->get_texture_gamma( 1), mi::neuraylib::ITarget_code::GM_GAMMA_DEFAULT);
            MI_CHECK_EQUAL( code_llvm->get_texture_gamma( 2), mi::neuraylib::ITarget_code::GM_GAMMA_LINEAR);
            MI_CHECK_EQUAL( code_llvm->get_texture_gamma( 3), mi::neuraylib::ITarget_code::GM_GAMMA_SRGB);
            MI_CHECK_EQUAL( code_llvm->get_texture_selector( 1), nullptr);
            MI_CHECK_EQUAL_CSTR( code_llvm->get_texture_selector( 2), "G");
            MI_CHECK_EQUAL_CSTR( code_llvm->get_texture_selector( 3), "R");
            MI_CHECK_EQUAL(0, code_llvm->get_bsdf_measurement_count());
            MI_CHECK_EQUAL(2, code_llvm->get_light_profile_count());
            MI_CHECK_EQUAL(3, code_llvm->get_string_constant_count());
            MI_CHECK_EQUAL_CSTR("something", code_llvm->get_string_constant(1));
            MI_CHECK_EQUAL_CSTR("abc",       code_llvm->get_string_constant(2));
        }

        // test link units: PTX
        {
            mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
                transaction->access<mi::neuraylib::IMaterial_instance>("mdl::" TEST_MDL "::mi_jit"));
            mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
                mi->create_compiled_material(flags_instance_compilation, context.get()));

            MI_CHECK_EQUAL(0, be_ptx->set_option("num_texture_spaces", "16"));

            mi::base::Handle<mi::neuraylib::ILink_unit> unit(
                be_ptx->create_link_unit(transaction, context.get()));
            MI_CHECK(unit.is_valid_interface());
            MI_CHECK_EQUAL(context->get_error_messages_count(), 0);

            errors = unit->add_material_expression(
                cm_ic.get(), "geometry.displacement", "displacement", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint_0", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value1.component.tint",
                "tint_1", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value2.component.tint",
                "tint_2", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_function(
                fc.get(), mi::neuraylib::ILink_unit::FEC_ENVIRONMENT, "env", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_function(
                fd_def.get(), mi::neuraylib::ILink_unit::FEC_CORE, NULL, context.get());
            MI_CHECK_EQUAL(0, errors);

            mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx(
                be_ptx->translate_link_unit(unit.get(), context.get()));
            MI_CHECK(code_ptx.is_valid_interface());
            MI_CHECK_EQUAL(context->get_error_messages_count(), 0);
            MI_CHECK(code_ptx->get_code());
            MI_CHECK(code_ptx->get_code_size() > 0);

            MI_CHECK_EQUAL(6, code_ptx->get_callable_function_count());
            MI_CHECK_EQUAL(4, code_ptx->get_texture_count());
            MI_CHECK_EQUAL(0, code_ptx->get_bsdf_measurement_count());
            MI_CHECK_EQUAL(2, code_ptx->get_light_profile_count());
            MI_CHECK_EQUAL(3, code_ptx->get_string_constant_count());
            MI_CHECK_EQUAL_CSTR("something", code_ptx->get_string_constant(1));
            MI_CHECK_EQUAL_CSTR("abc",       code_ptx->get_string_constant(2));
        }

        // test link units: GLSL
        errors = -1;
        {
            mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
                transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
            mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
                mi->create_compiled_material( flags_instance_compilation, context.get()));

            MI_CHECK_EQUAL( 0, be_glsl->set_option( "num_texture_spaces", "16"));

            mi::base::Handle<mi::neuraylib::ILink_unit> unit(
                be_glsl->create_link_unit( transaction, context.get()));
            MI_CHECK( unit.is_valid_interface());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

            errors = unit->add_material_expression(
                cm_ic.get(), "geometry.displacement", "displacement", context.get());
            MI_CHECK_EQUAL( 0, errors);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint_0", context.get());
            MI_CHECK_EQUAL( 0, errors);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value1.component.tint",
                "tint_1", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_material_expression(
                cm_ic.get(),
                "surface.scattering.components.value2.component.tint",
                "tint_2", context.get());
            MI_CHECK_EQUAL(0, errors);

            errors = unit->add_function(
                fc.get(), mi::neuraylib::ILink_unit::FEC_ENVIRONMENT, "env", context.get());
            MI_CHECK_EQUAL( 0, errors);

            errors = unit->add_function(
                fd_def.get(), mi::neuraylib::ILink_unit::FEC_CORE, NULL, context.get());
            MI_CHECK_EQUAL(0, errors);

            mi::base::Handle<const mi::neuraylib::ITarget_code> code_glsl(
                be_glsl->translate_link_unit( unit.get(), context.get()));
            MI_CHECK( code_glsl.is_valid_interface());
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0);
            MI_CHECK( code_glsl->get_code());
            MI_CHECK( code_glsl->get_code_size() > 0);

            MI_CHECK_EQUAL(6, code_glsl->get_callable_function_count());
            MI_CHECK_EQUAL(4, code_glsl->get_texture_count());
            MI_CHECK_EQUAL(0, code_glsl->get_bsdf_measurement_count());
            MI_CHECK_EQUAL(2, code_glsl->get_light_profile_count());
            MI_CHECK_EQUAL(3, code_glsl->get_string_constant_count());
            MI_CHECK_EQUAL_CSTR("something", code_glsl->get_string_constant(1));
            MI_CHECK_EQUAL_CSTR("abc",       code_glsl->get_string_constant(2));
        }
    }
}

void check_create_archive(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::IMdl_archive_api* mdl_archive_api)
{
    std::string directory = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/create");

    MI_CHECK_EQUAL_CSTR( ".ies,.mbsdf,.txt,.html", mdl_archive_api->get_extensions_for_compression());

    result =  mdl_archive_api->set_extensions_for_compression( 0);
    MI_CHECK_EQUAL( -1, result);

    result =  mdl_archive_api->set_extensions_for_compression( "");
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL_CSTR( "", mdl_archive_api->get_extensions_for_compression());

    result =  mdl_archive_api->set_extensions_for_compression( ".doc");
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL_CSTR( ".doc", mdl_archive_api->get_extensions_for_compression());

    uninstall_external_resolver( mdl_configuration);

    result = mdl_archive_api->create_archive( 0, DIR_PREFIX "/test_create_archives.mdr", 0);
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->create_archive( directory.c_str(), 0, 0);
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives_wrong_extension", 0);
    MI_CHECK_EQUAL( -2, result);

    mi::base::Handle<mi::IArray> wrong_array_type( transaction->create<mi::IArray>( "Sint32[42]"));
    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives.mdr", wrong_array_type.get());
    MI_CHECK_EQUAL( -3, result);

    wrong_array_type = transaction->create<mi::IArray>( "Interface[1]");
    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives.mdr", wrong_array_type.get());
    MI_CHECK_EQUAL( -3, result);

    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives_wrong_package.mdr", 0);
    MI_CHECK_EQUAL( -4, result);

    mi::base::Handle<mi::IDynamic_array> fields(
        transaction->create<mi::IDynamic_array>( "Manifest_field[]"));
    fields->set_length( 2);

    mi::base::Handle<mi::IStructure> field0( fields->get_value<mi::IStructure>( zero_size));
    mi::base::Handle<mi::IString> key0( field0->get_value<mi::IString>( "key"));
    key0->set_c_str( "foo");
    mi::base::Handle<mi::IString> value0( field0->get_value<mi::IString>( "value"));
    value0->set_c_str( "bar");

    mi::base::Handle<mi::IStructure> field1( fields->get_value<mi::IStructure>( 1));
    mi::base::Handle<mi::IString> key1( field1->get_value<mi::IString>( "key"));
    key1->set_c_str( "foo");
    mi::base::Handle<mi::IString> value1( field1->get_value<mi::IString>( "value"));
    value1->set_c_str( "baz");

#ifndef MI_PLATFORM_WINDOWS
    std::string archive = DIR_PREFIX "/test_create_archives.mdr";
#else
    std::string archive = DIR_PREFIX "\\test_create_archives.mdr";
#endif

    result = mdl_archive_api->create_archive(
        directory.c_str(), archive.c_str(), fields.get());
    MI_CHECK_EQUAL( 0, result);

    fs::remove( archive);

    install_external_resolver( mdl_configuration);

#ifdef EXTERNAL_ENTITY_RESOLVER
    result = mdl_archive_api->create_archive( directory.c_str(), archive.c_str(), fields.get());
    MI_CHECK_EQUAL( -5, result);
#endif
}

void check_extract_archive( mi::neuraylib::IMdl_archive_api* mdl_archive_api)
{
    std::string archive       = MI::TEST::mi_src_path( "prod/lib/neuray/test_archives.mdr");
    std::string wrong_archive = MI::TEST::mi_src_path( "prod/lib/neuray/" TEST_MDL_FILE ".mdl");

    result = mdl_archive_api->extract_archive( 0, ".");
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->extract_archive( archive.c_str(), 0);
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->extract_archive( wrong_archive.c_str(), ".");
    MI_CHECK_EQUAL( -2, result);

    result = mdl_archive_api->extract_archive( "non-existing.mdr", ".");
    MI_CHECK_EQUAL( -2, result);

    result = mdl_archive_api->extract_archive( archive.c_str(), DIR_PREFIX "/extracted");
    MI_CHECK_EQUAL( 0, result);

    result = mdl_archive_api->extract_archive( archive.c_str(), DIR_PREFIX "/extracted-non-existing");
    MI_CHECK_EQUAL( 0, result);
}

void check_get_manifest( mi::neuraylib::IMdl_archive_api* mdl_archive_api)
{
    std::string archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/query/test_archives.mdr");
    std::string wrong_archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/" TEST_MDL_FILE ".mdl");

    mi::base::Handle<const mi::neuraylib::IManifest> manifest;

    manifest = mdl_archive_api->get_manifest( 0);
    MI_CHECK( !manifest);

    manifest = mdl_archive_api->get_manifest( "non-existing.mdr");
    MI_CHECK( !manifest);

    manifest = mdl_archive_api->get_manifest( wrong_archive.c_str());
    MI_CHECK( !manifest);

    manifest = mdl_archive_api->get_manifest( archive.c_str());
    MI_CHECK( manifest);

    mi::Size count = manifest->get_number_of_fields();
    MI_CHECK_EQUAL( count, 13);

    const char* key = manifest->get_key( 0);
    MI_CHECK_EQUAL_CSTR( key, "mdl");

    key = manifest->get_key( 13);
    MI_CHECK( !key);

    const char* value = manifest->get_value( 0);
    MI_CHECK_EQUAL_CSTR( value, "1.3");

    value = manifest->get_value( 22);
    MI_CHECK( !value);

    count = manifest->get_number_of_fields( 0);
    MI_CHECK_EQUAL( count, 0);

    count = manifest->get_number_of_fields( "non-existing");
    MI_CHECK_EQUAL( count, 0);

    count = manifest->get_number_of_fields( "mdl");
    MI_CHECK_EQUAL( count, 1);

    count = manifest->get_number_of_fields( "exports.function");
    MI_CHECK_EQUAL( count, 10);

    value = manifest->get_value( 0, 0);
    MI_CHECK( !value);

    value = manifest->get_value( "non-existing", 0);
    MI_CHECK( !value);

    value = manifest->get_value( "mdl", 0);
    MI_CHECK_EQUAL_CSTR( value, "1.3");

    value = manifest->get_value( "mdl", 1);
    MI_CHECK( !value);

    value = manifest->get_value( "exports.function", 0);
    MI_CHECK_EQUAL_CSTR( value, "::test_archives::fd_in_archive");

    value = manifest->get_value( "exports.function", 9);
    MI_CHECK_EQUAL_CSTR( value, "::test_archives::fd_in_archive_bsdf_measurement_absolute");

    value = manifest->get_value( "exports.function", 10);
    MI_CHECK( !value);

    value = manifest->get_value( "userdefined", 0);
    MI_CHECK( !value);
}

void check_get_file( mi::neuraylib::IMdl_archive_api* mdl_archive_api)
{
    std::string archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/query/test_archives.mdr");
    std::string wrong_archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/" TEST_MDL_FILE ".mdl");

    mi::base::Handle<mi::neuraylib::IReader> reader;

    reader = mdl_archive_api->get_file( 0, "foo");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( "foo", 0);
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( "non-existing.mdr", "foo");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( wrong_archive.c_str(), "foo");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( archive.c_str(), "test_archives.mdl");
    MI_CHECK( reader);

#ifndef MI_PLATFORM_WINDOWS
    reader = mdl_archive_api->get_file( archive.c_str(), "test_archives/test_in_archive.png");
#else
    reader = mdl_archive_api->get_file( archive.c_str(), "test_archives\\test_in_archive.png");
#endif
    MI_CHECK( reader);

    reader = mdl_archive_api->get_file( archive.c_str(), "non-existing.mdl");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( archive.c_str(), "test_archives");
    MI_CHECK( !reader);
}

void check_resources(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IValue_texture> texture;
    mi::base::Handle<mi::neuraylib::IValue_light_profile> light_profile;
    mi::base::Handle<mi::neuraylib::IValue_bsdf_measurement> bsdf_measurement;
    mi::neuraylib::IType_texture::Shape shape = mi::neuraylib::IType_texture::TS_2D;

    // textures

    texture = mdl_factory->create_texture(
        transaction,
        "/mdl_elements/resources/test.png",
        shape,
        1.0f,
        "R",
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( texture);
    MI_CHECK_EQUAL_CSTR( texture->get_file_path(), "/mdl_elements/resources/test.png");
    MI_CHECK_EQUAL( texture->get_gamma(), 1.0f);
    MI_CHECK_EQUAL_CSTR( texture->get_selector(), "R");

    texture = mdl_factory->create_texture(
        transaction,
        "/test_archives/test_in_archive.png",
        shape,
        2.2f,
        "G",
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( texture);
    MI_CHECK_EQUAL_CSTR( texture->get_file_path(), "/test_archives/test_in_archive.png");
    MI_CHECK_EQUAL( texture->get_gamma(), 2.2f);
    MI_CHECK_EQUAL_CSTR( texture->get_selector(), "G");

    texture = mdl_factory->create_texture(
        /*transaction*/ nullptr,
        "/mdl_elements/resources/test.png",
        shape,
        0.0f,
        /*selector*/ nullptr,
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -1);
    MI_CHECK( !texture);
    context->clear_messages();

    texture = mdl_factory->create_texture(
        transaction,
        /*file_path*/ nullptr,
        shape,
        0.0f,
        /*selector*/ nullptr,
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -1);
    MI_CHECK( !texture);
    context->clear_messages();

    texture = mdl_factory->create_texture(
        transaction,
        "mdl_elements/resources/test.png",
        shape,
        0.0f,
        /*selector*/ nullptr,
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -2);
    MI_CHECK( !texture);
    context->clear_messages();

    texture = mdl_factory->create_texture(
        transaction,
        "/mdl_elements/resources/test.ies",
        shape,
        0.0f,
        /*selector*/ nullptr,
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -3);
    MI_CHECK( !texture);
    context->clear_messages();

    texture = mdl_factory->create_texture(
        transaction,
        "/mdl_elements/resources/non-existing.png",
        shape,
        0.0f,
        /*selector*/ nullptr,
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -4);
    MI_CHECK( !texture);
    context->clear_messages();

    // -5 difficult to test

    texture = mdl_factory->create_texture(
        transaction,
        "/mdl_elements/resources/test.png",
        shape,
        0.0f,
        "X",
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -7);
    MI_CHECK( !texture);
    context->clear_messages();

    texture = mdl_factory->create_texture(
        transaction,
        "/mdl_elements/resources/test.dds",
        shape,
        0.0f,
        "X",
        /*shared*/ false,
        context.get());
    MI_CHECK_CTX_CODE( context.get(), -10);
    MI_CHECK( !texture);
    context->clear_messages();

    // light profiles

    light_profile = mdl_factory->create_light_profile(
        transaction, "/mdl_elements/resources/test.ies", /*shared*/ false, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( light_profile);
    MI_CHECK_EQUAL_CSTR( light_profile->get_file_path(), "/mdl_elements/resources/test.ies");
    context->clear_messages();

    light_profile = mdl_factory->create_light_profile(
        transaction, "/test_archives/test_in_archive.ies", /*shared*/ false, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( light_profile);
    MI_CHECK_EQUAL_CSTR( light_profile->get_file_path(), "/test_archives/test_in_archive.ies");
    context->clear_messages();

    light_profile = mdl_factory->create_light_profile(
        /*transaction*/ nullptr, "/mdl_elements/resources/test.ies", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -1);
    MI_CHECK( !light_profile);
    context->clear_messages();

    light_profile = mdl_factory->create_light_profile(
        transaction, /*file_path*/ nullptr, /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -1);
    MI_CHECK( !light_profile);
    context->clear_messages();

    light_profile = mdl_factory->create_light_profile(
        transaction, "mdl_elements/resources/test.ies", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -2);
    MI_CHECK( !light_profile);
    context->clear_messages();

    light_profile = mdl_factory->create_light_profile(
        transaction, "/mdl_elements/resources/test.png", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -3);
    MI_CHECK( !light_profile);
    context->clear_messages();

    light_profile = mdl_factory->create_light_profile(
        transaction, "/mdl_elements/resources/non-existing.ies", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -4);
    MI_CHECK( !light_profile);
    context->clear_messages();

    // -5 difficult to test

    light_profile = mdl_factory->create_light_profile(
        transaction, "/mdl_elements/resources/test_file_format_error.ies", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -7);
    MI_CHECK( !light_profile);
    context->clear_messages();

    // BSDF measurements

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, "/mdl_elements/resources/test.mbsdf", /*shared*/ false, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( bsdf_measurement);
    MI_CHECK_EQUAL_CSTR( bsdf_measurement->get_file_path(), "/mdl_elements/resources/test.mbsdf");
    context->clear_messages();

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, "/test_archives/test_in_archive.mbsdf", /*shared*/ false, context.get());
    MI_CHECK_CTX( context.get());
    MI_CHECK( bsdf_measurement);
    MI_CHECK_EQUAL_CSTR( bsdf_measurement->get_file_path(), "/test_archives/test_in_archive.mbsdf");
    context->clear_messages();

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        /*transaction*/ nullptr, "/mdl_elements/resources/test.mbsdf", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -1);
    MI_CHECK( !bsdf_measurement);
    context->clear_messages();

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, /*file_path*/ nullptr, /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -1);
    MI_CHECK( !bsdf_measurement);
    context->clear_messages();

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, "mdl_elements/resources/test.mbsdf", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -2);
    MI_CHECK( !bsdf_measurement);
    context->clear_messages();

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, "/mdl_elements/resources/non-existing.png", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -3);
    MI_CHECK( !bsdf_measurement);
    context->clear_messages();

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, "/mdl_elements/resources/non-existing.mbsdf", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -4);
    MI_CHECK( !bsdf_measurement);
    context->clear_messages();

    // -5 difficult to test

    bsdf_measurement = mdl_factory->create_bsdf_measurement(
        transaction, "/bsdf_measurement/test_file_format_error.mbsdf", /*shared*/ false, context.get());
    MI_CHECK_CTX_CODE( context.get(), -7);
    MI_CHECK( !bsdf_measurement);
    context->clear_messages();
}

// Test IImage::reset_reader(), ILightprofile::reset_reader(), IBsdf_measurement::reset_reader().
// This functionality is not really MDL-related, we just misuse IMdl_archive::get_file() as a simple
// way to obtain a reader to resources.
void check_resources2(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_archive_api* mdl_archive_api,
    mi::neuraylib::IImage_api* image_api)
{
    std::string archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/query/test_archives.mdr");

    mi::base::Handle<mi::neuraylib::IReader> reader;

    {
        std::string member = std::string( "test_archives") + dir_sep + "test_in_archive.png";
        reader = mdl_archive_api->get_file( archive.c_str(), member.c_str());
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::IImage> img(
           transaction->create<mi::neuraylib::IImage>( "Image"));
        result  = img->reset_reader( reader.get(), "png");
        MI_CHECK_EQUAL( result, 0);
    }

    {
        std::string member = std::string( "test_archives") + dir_sep + "test_in_archive.ies";
        reader = mdl_archive_api->get_file( archive.c_str(), member.c_str());
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::ILightprofile> lp(
           transaction->create<mi::neuraylib::ILightprofile>( "Lightprofile"));
        result  = lp->reset_reader( reader.get());
        MI_CHECK_EQUAL( result, 0);
    }

    {
        std::string member = std::string( "test_archives") + dir_sep + "test_in_archive.mbsdf";
        reader = mdl_archive_api->get_file( archive.c_str(), member.c_str());
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::IBsdf_measurement> bsdfm(
            transaction->create<mi::neuraylib::IBsdf_measurement>( "Bsdf_measurement"));
        result = bsdfm->reset_reader( reader.get());
        MI_CHECK_EQUAL( result, 0);
    }

    // selectors via intermediate canvas
    {
        std::string member = std::string( "test_archives") + dir_sep + "test_in_archive.png";
        reader = mdl_archive_api->get_file( archive.c_str(), member.c_str());
        MI_CHECK( reader);

        mi::base::Handle<mi::neuraylib::ICanvas> canvas;
        canvas = image_api->create_canvas_from_reader( reader.get(), "png");
        MI_CHECK( canvas);
        MI_CHECK_EQUAL_CSTR( canvas->get_type(), "Rgb");
        canvas = image_api->create_canvas_from_reader( reader.get(), "png", "R");
        MI_CHECK( canvas);
        MI_CHECK_EQUAL_CSTR( canvas->get_type(), "Sint8");

        mi::base::Handle<mi::neuraylib::IImage> img(
           transaction->create<mi::neuraylib::IImage>( "Image"));
        bool ok = img->set_from_canvas( canvas.get(), /*selector*/ nullptr);
        MI_CHECK( ok);
        MI_CHECK( !img->get_selector());
        ok = img->set_from_canvas( canvas.get(), "whatever");
        MI_CHECK( ok);
        MI_CHECK_EQUAL_CSTR( img->get_selector(), "whatever");
    }
}

void export_canvas(
    const char* filename, const mi::neuraylib::ICanvas* canvas, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    result = mdl_impexp_api->export_canvas( filename, canvas);
    MI_CHECK_EQUAL( 0, result);
}

// Check whether the canvases are equal except for small rounding errors.
bool check_canvas_nearly_equal(
    const mi::neuraylib::ICanvas* canvas_a,
    const mi::neuraylib::ICanvas* canvas_b,
    size_t num_image_floats)
{
    mi::base::Handle<const mi::neuraylib::ITile> tile_a( canvas_a->get_tile());
    mi::base::Handle<const mi::neuraylib::ITile> tile_b( canvas_b->get_tile());
    const float* data_a = reinterpret_cast<const float*>( tile_a->get_data());
    const float* data_b = reinterpret_cast<const float*>( tile_b->get_data());
    for( size_t i = 0; i < num_image_floats; ++i)
        if( abs(data_a[i] - data_b[i]) > 0.00001f)
            return false;
    return true;
}

mi::neuraylib::ICanvas* bake_to_canvas(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_distiller_api* mdl_distiller_api,
    mi::neuraylib::INeuray* neuray,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::neuraylib::IMaterial_instance::Compilation_options compile_flags,
    mi::neuraylib::Baker_resource baker_flags)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
        transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_class_baking"));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
        compile_flags, context.get()));
    MI_CHECK_CTX( context.get());
    MI_CHECK( cm);

    mi::base::Handle<const mi::neuraylib::IBaker> baker( mdl_distiller_api->create_baker(
        cm.get(), "surface.scattering.tint", baker_flags, 0));
    MI_CHECK( baker);
    MI_CHECK_EQUAL( baker->is_uniform(), false);
    const char* pixel_type = baker->get_pixel_type();
    MI_CHECK_EQUAL_CSTR( pixel_type, "Rgb_fp");

    mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        neuray->get_api_component<mi::neuraylib::IImage_api>());
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas( pixel_type, width, height));
    result = baker->bake_texture( canvas.get());
    MI_CHECK_EQUAL( 0, result);
    // static int pic_idx = 0;
    // export_canvas( (std::string("test_imdl_module_baked_test_cube_rgb_fp-") +
    //     std::to_string(pic_idx++) + ".png").c_str(), canvas.get(), neuray);

    // additional test for baker
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    mi::base::Handle<mi::IColor> constant( factory->create<mi::IColor>());
    result = baker->bake_constant( constant.get());
    MI_CHECK_EQUAL( 0, result);

    canvas->retain();
    return canvas.get();
}

void check_distiller(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_distiller_api* mdl_distiller_api,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // Distill a compiled material from instance compilation
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_resource_sharing"));


        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( cm);

        mi::base::Handle<mi::IBoolean> poc( transaction->create<mi::IBoolean>( "Boolean"));
        poc->set_value( false);
        mi::base::Handle<mi::IMap> options( transaction->create<mi::IMap>( "Map<Interface>"));
        options->insert( "_poc", poc.get());

        mi::Sint32 errors;
        mi::base::Handle<const mi::neuraylib::ICompiled_material> new_cm(
            mdl_distiller_api->distill_material( cm.get(), "diffuse", options.get(), &errors));
        MI_CHECK_EQUAL( errors, 0);
        MI_CHECK( new_cm);
    }

    {
        // Distill a compiled material from class compilation
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_2_index_used_index"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( cm);
        MI_CHECK( cm->get_parameter_count() > 0);

        mi::base::Handle<mi::IBoolean> poc( transaction->create<mi::IBoolean>( "Boolean"));
        poc->set_value( false);
        mi::base::Handle<mi::IMap> options( transaction->create<mi::IMap>( "Map<Interface>"));
        options->insert( "_poc", poc.get());

        mi::Sint32 errors;
        mi::base::Handle<const mi::neuraylib::ICompiled_material> new_cm(
            mdl_distiller_api->distill_material( cm.get(), "diffuse", options.get(), &errors));
        MI_CHECK_EQUAL( errors, 0);
        MI_CHECK( new_cm);
    }

#ifndef RESOLVE_RESOURCES_FALSE
    // The baked images will be 100 * 100 Rgb_fp values
    const mi::Uint32 width = 100, height = 100;
    const size_t num_image_floats = width * height * 3;

    // Bake an expression of "Rgb_fp" type on GPU (if available)
    mi::base::Handle<mi::neuraylib::ICanvas> reference_canvas(
        bake_to_canvas( transaction, mdl_distiller_api, neuray, width, height,
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS,
            mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK));

    mi::base::Handle<mi::neuraylib::ICanvas> canvas;

    // Bake same expression of "Rgb_fp" type on CPU
    canvas = bake_to_canvas( transaction, mdl_distiller_api, neuray, width, height,
        mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS,
        mi::neuraylib::BAKE_ON_CPU);
    MI_CHECK( check_canvas_nearly_equal(
        reference_canvas.get(), canvas.get(), num_image_floats));

    // Bake the same texture twice on GPU with class compilation, second run uses PTX code cache
    canvas = bake_to_canvas( transaction, mdl_distiller_api, neuray, width, height,
        mi::neuraylib::IMaterial_instance::CLASS_COMPILATION,
        mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK);
    MI_CHECK( check_canvas_nearly_equal(
        reference_canvas.get(), canvas.get(), num_image_floats));

    canvas = bake_to_canvas( transaction, mdl_distiller_api, neuray, width, height,
        mi::neuraylib::IMaterial_instance::CLASS_COMPILATION,
        mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK);
    MI_CHECK( check_canvas_nearly_equal(
        reference_canvas.get(), canvas.get(), num_image_floats));

    // Bake same expression of "Rgb_fp" type on CPU with class compilation
    canvas = bake_to_canvas( transaction, mdl_distiller_api, neuray, width, height,
        mi::neuraylib::IMaterial_instance::CLASS_COMPILATION,
        mi::neuraylib::BAKE_ON_CPU);
    MI_CHECK( check_canvas_nearly_equal(
        reference_canvas.get(), canvas.get(), num_image_floats));

    {
        // Bake an expression of "Float32" type
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_baking"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
        MI_CHECK_CTX( context.get());
        MI_CHECK( cm);

        mi::base::Handle<const mi::neuraylib::IBaker> baker( mdl_distiller_api->create_baker(
            cm.get(), "backface.scattering.tint.r", mi::neuraylib::BAKE_ON_CPU));
        MI_CHECK( baker);
        MI_CHECK_EQUAL( baker->is_uniform(), false);
        const char* pixel_type = baker->get_pixel_type();
        MI_CHECK_EQUAL_CSTR( pixel_type, "Float32");

        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            neuray->get_api_component<mi::neuraylib::IImage_api>());
        mi::base::Handle<mi::neuraylib::ICanvas> canvas( image_api->create_canvas( pixel_type, 100, 100));
        result = baker->bake_texture( canvas.get());
        MI_CHECK_EQUAL( 0, result);
        // export_canvas( "test_imdl_module_baked_test_cube_float.png", canvas.get(), neuray);

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        mi::base::Handle<mi::IFloat32> constant( factory->create<mi::IFloat32>());
        result = baker->bake_constant( constant.get());
        MI_CHECK_EQUAL( 0, result);
    }
#endif // RESOLVE_RESOURCES_FALSE
}

// This test invokes the distiller on several materials, distilling each
// to all available targets in the mdl_distiller plugin. Then, the hashes
// of all distilled materials are compared to make sure that they differ
// when expected.
void check_distiller_for_multiple_targets(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_distiller_api* mdl_distiller_api,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Tested targets. We only test agains mdl_distiller plugin for now.
    const char* targets[] = {
        "diffuse",
        "specular_glossy",
        "ue4",
        "transmissive_pbr"
    };

    const size_t TARGET_COUNT = sizeof(targets) / sizeof(targets[0]);
    mi::base::Uuid hashes[TARGET_COUNT];

    // First, check with material mi_0.
    {
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>("mdl::" TEST_MDL "::mi_0"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, context.get()));
        MI_CHECK_CTX(context.get());
        MI_CHECK(cm);
        mi::base::Uuid compiled_hash = cm->get_hash();

        //mi::base::Handle<mi::IMap> options(transaction->create<mi::IMap>("Map<Interface>"));

        for (size_t i = 0; i < TARGET_COUNT; i++)
        {
            mi::Sint32 errors;
            mi::base::Handle<const mi::neuraylib::ICompiled_material> new_cm(
                mdl_distiller_api->distill_material(cm.get(), targets[i], nullptr /* options */, &errors));
            MI_CHECK_EQUAL(errors, 0);
            MI_CHECK(new_cm);
            hashes[i] = new_cm->get_hash();

            // Make sure that the original material's hash differs from the compiled material's.
            MI_CHECK(hashes[i] != compiled_hash);

            // Make sure that the compiled hash differs from other target hashes.
            for (size_t j = 0; j < i; j++)
            {
                MI_CHECK(hashes[j] != hashes[i]);
            }
        }
    }

    // Now check with a second material mi_1.
    {
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>("mdl::" TEST_MDL "::mi_1"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, context.get()));
        MI_CHECK_CTX(context.get());
        MI_CHECK(cm);
        mi::base::Uuid compiled_hash = cm->get_hash();

        mi::base::Handle<mi::IMap> options(transaction->create<mi::IMap>("Map<Interface>"));

        for (size_t i = 0; i < TARGET_COUNT; i++)
        {
            mi::Sint32 errors;
            mi::base::Handle<const mi::neuraylib::ICompiled_material> new_cm(
                mdl_distiller_api->distill_material(cm.get(), targets[i], options.get(), &errors));
            MI_CHECK_EQUAL(errors, 0);
            MI_CHECK(new_cm);
            hashes[i] = new_cm->get_hash();
            MI_CHECK(hashes[i] != compiled_hash);
            for (size_t j = 0; j < i; j++)
            {
                MI_CHECK(hashes[j] != hashes[i]);
            }
        }
    }
}

void check_type_compatibility(
    mi::neuraylib::INeuray* neuray,
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory(transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory(transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create function call
    mi::base::Handle<const mi::neuraylib::IFunction_definition> def(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::test_compatibility::structure_test(::test_compatibility::struct_one)"));
    MI_CHECK(def);
    mi::base::Handle<mi::neuraylib::IFunction_call> inst(def->create_function_call(nullptr));
    MI_CHECK(inst);

    // create struct constant c
    mi::base::Handle<const mi::neuraylib::IType_struct> struct_type(
        tf->create_struct("::test_compatibility::struct_two"));
    mi::base::Handle<mi::neuraylib::IValue_struct> struct_value(
        vf->create_struct(struct_type.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> c(
        ef->create_constant(struct_value.get()));

    // attempt to assign it to compatible struct
    MI_CHECK_EQUAL(inst->set_argument(zero_size, c.get()), 0);

    mi::base::Handle<const mi::neuraylib::IExpression> cast_call;

    {
        // instance cast operator, pass c as its argument and store it in database
        mi::base::Handle<const mi::neuraylib::IFunction_definition> cast_def(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::operator_cast(%3C0%3E)"));
        MI_CHECK(cast_def);
        mi::base::Handle<mi::neuraylib::IExpression_list> args(
            ef->create_expression_list());
        MI_CHECK_EQUAL(args->add_expression("cast", c.get()), 0);
        // create dummy expression for return type
        mi::base::Handle<const mi::neuraylib::IType_struct> cast_return_type(
            tf->create_struct("::test_compatibility::struct_one"));
        mi::base::Handle<mi::neuraylib::IValue_struct> cast_return_value(
            vf->create_struct(cast_return_type.get()));
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_ret(
            ef->create_constant(cast_return_value.get()));
        // and add it
        MI_CHECK_EQUAL(args->add_expression("cast_return", c_ret.get()), 0);

        mi::base::Handle<mi::neuraylib::IFunction_call> cast_inst(
            cast_def->create_function_call(args.get(), 0));
        MI_CHECK(cast_inst);
        MI_CHECK_EQUAL(transaction->store(cast_inst.get(), "my_cast"), 0);
        mi::base::Handle<const mi::neuraylib::IExpression> cast_call(ef->create_call("my_cast"));
        MI_CHECK(cast_call);

        MI_CHECK_EQUAL(inst->set_argument(zero_size, cast_call.get()), 0);
    }
    {
        mi::Sint32 errors = 0;
        mi::base::Handle<const mi::neuraylib::IType_struct> cast_result_type(
            tf->create_struct("::test_compatibility::struct_one"));
        mi::base::Handle<mi::neuraylib::IExpression> cast_call(
            ef->create_cast(
                c.get(),
                cast_result_type.get(),
                /*cast_db_name=*/nullptr,
                /*force_cast=*/false,
                &errors));
        MI_CHECK(cast_call);
        MI_CHECK_EQUAL(errors, 0);
        MI_CHECK_EQUAL(inst->set_argument(zero_size, cast_call.get()), 0);

        // create an expression of the target type
        mi::base::Handle<const mi::neuraylib::IType_struct> struct_type1(
            tf->create_struct("::test_compatibility::struct_one"));
        mi::base::Handle<mi::neuraylib::IValue_struct> struct_value1(
            vf->create_struct(struct_type1.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c1(
            ef->create_constant(struct_value1.get()));

        mi::base::Handle<mi::neuraylib::IExpression> cast_call1(
            ef->create_cast(
                c1.get(),
                cast_result_type.get(),
                /*cast_db_name=*/nullptr,
                /*force_cast=*/false,
                &errors));
        MI_CHECK_EQUAL(errors, 0);
        MI_CHECK(cast_call1);
        // this should return c1
        MI_CHECK_EQUAL(c1, cast_call1);

        // test force option
        cast_call1 = ef->create_cast(
            c1.get(),
            cast_result_type.get(),
            /*cast_db_name=*/nullptr,
            /*force_cast=*/true,
            &errors);
        MI_CHECK(cast_call1);
        MI_CHECK_EQUAL(errors, 0);
        MI_CHECK_NOT_EQUAL(c1, cast_call1);

        // test error cases

        // invalid input
        cast_call1 = ef->create_cast(
            nullptr,
            cast_result_type.get(),
            /*cast_db_name=*/nullptr,
            /*force_cast=*/true,
            &errors);
        MI_CHECK(!cast_call1);
        MI_CHECK_EQUAL(errors, -1);

        cast_call1 = ef->create_cast(
            c1.get(),
            nullptr,
            /*cast_db_name=*/nullptr,
            /*force_cast=*/true,
            &errors);
        MI_CHECK(!cast_call1);
        MI_CHECK_EQUAL(errors, -1);

        // incompatible type
        mi::base::Handle<const mi::neuraylib::IType_int> int_type(
            tf->create_int());
        cast_call1 = ef->create_cast(
            c1.get(),
            int_type.get(),
            /*cast_db_name=*/nullptr,
            /*force_cast=*/true,
            &errors);
        MI_CHECK(!cast_call1);
        MI_CHECK_EQUAL(errors, -2);

        // test db name that already exists
        mi::base::Handle<mi::neuraylib::IExpression_call> c0(
            cast_call->get_interface<mi::neuraylib::IExpression_call>());
        cast_call1 = ef->create_cast(
            c1.get(),
            cast_result_type.get(),
            c0->get_call(),
            /*force_cast=*/true,
            &errors);
        MI_CHECK(cast_call1);
        mi::base::Handle<mi::neuraylib::IExpression_call> c_new(
            cast_call1->get_interface<mi::neuraylib::IExpression_call>());
        MI_CHECK_EQUAL(errors, 0);
        MI_CHECK_NOT_EQUAL(c_new->get_call(), c0->get_call());
    }

    // test automaic cast insertion

    // function call creation
    {
        // ... with argument that requires cast
        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(def->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_call> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_call>(zero_size)); // this should be a call
        MI_CHECK(arg0);
    }
    // function call creation, parameter type with modifier
    {
        // ... with argument that requires cast
        mi::base::Handle<const mi::neuraylib::IFunction_definition> def_modifier(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_compatibility::structure_test_modifier(::test_compatibility::struct_one)"));
        MI_CHECK(def_modifier);

        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            def_modifier->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_call> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_call>(zero_size)); // this should be a call
        MI_CHECK(arg0);
    }
    // function call creation, argument type with modifier
    {
        // ... with argument that requires cast
        mi::base::Handle<const mi::neuraylib::IFunction_definition> def_no_modifier(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_compatibility::structure_test_func(::test_compatibility::struct_one)"));
        MI_CHECK(def_no_modifier);

        do_create_function_call(
            transaction, "mdl::test_compatibility::struct_return_modifier(bool)", "fc_struct_return_modifier");
        mi::base::Handle<mi::neuraylib::IExpression_call> call(
            ef->create_call( "fc_struct_return_modifier"));

        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", call.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            def_no_modifier->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);
    }
    {
        // ... with argument that does not require cast

        // create constant of correct type
        mi::base::Handle<const mi::neuraylib::IType_struct> struct_type1(
            tf->create_struct("::test_compatibility::struct_one"));
        mi::base::Handle<mi::neuraylib::IValue_struct> struct_value1(
            vf->create_struct(struct_type1.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c1(
            ef->create_constant(struct_value1.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c1.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(def->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_constant>(zero_size)); // this should be a constant
        MI_CHECK(arg0);
    }
    {
        // ... with incompatible argument

        // create constant of correct type
        mi::base::Handle<mi::neuraylib::IValue_int> int_value1(
            vf->create_int(3));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c1(
            ef->create_constant(int_value1.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c1.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(def->create_function_call(args.get(), &errors));
        MI_CHECK(!call_inst);
        MI_CHECK_EQUAL(errors, -2);
    }

    // test material instance creation
    mi::base::Handle<const mi::neuraylib::IFunction_definition> mat_def(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::test_compatibility::structure_test_mat(::test_compatibility::struct_one)"));
    MI_CHECK(mat_def);

    {
        // ... with argument that requires cast
        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            mat_def->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_call> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_call>(zero_size)); // this should be a call
        MI_CHECK(arg0);
    }
    {
        // ... with argument that does not require cast
        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());

        // create constant of correct type
        mi::base::Handle<const mi::neuraylib::IType_struct> struct_type1(
            tf->create_struct("::test_compatibility::struct_one"));
        mi::base::Handle<mi::neuraylib::IValue_struct> struct_value1(
            vf->create_struct(struct_type1.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c1(
            ef->create_constant(struct_value1.get()));

        args->add_expression("v", c1.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            mat_def->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> arg0(
            call_args->get_expression<mi::neuraylib::IExpression_constant>(zero_size)); // this should be a constant
        MI_CHECK(arg0);
    }
    {
        // ... with incompatible argument
        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());

        // create constant of correct type
        mi::base::Handle<mi::neuraylib::IValue_int> int_value1(
            vf->create_int(3));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c1(
            ef->create_constant(int_value1.get()));

        args->add_expression("v", c1.get());
        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(
            mat_def->create_function_call(args.get(), &errors));
        MI_CHECK(!call_inst);
        MI_CHECK_EQUAL(errors, -2);
    }
    // check array constructor creation. not too sure use case this should be valid after all ;)
    {
        // create constant of a type
        mi::base::Handle<const mi::neuraylib::IType_struct> struct_type1(
            tf->create_struct("::test_compatibility::struct_one"));
        mi::base::Handle<mi::neuraylib::IValue_struct> struct_value1(
            vf->create_struct(struct_type1.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c1(
            ef->create_constant(struct_value1.get()));

        // create constant of a compatible type
        mi::base::Handle<const mi::neuraylib::IType_struct> struct_type2(
            tf->create_struct("::test_compatibility::struct_two"));
        mi::base::Handle<mi::neuraylib::IValue_struct> struct_value2(
            vf->create_struct(struct_type2.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> c2(
            ef->create_constant(struct_value2.get()));

        mi::base::Handle<const mi::neuraylib::IFunction_definition> const_def(
            transaction->access<mi::neuraylib::IFunction_definition>("mdl::T[](...)"));
        MI_CHECK(const_def);

        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("value0", c1.get());
        args->add_expression("value1", c2.get());
        args->add_expression("value2", c1.get());
        args->add_expression("value3", c2.get());

        mi::Sint32 errors = 0;
        mi::base::Handle<mi::neuraylib::IFunction_call> call_inst(const_def->create_function_call(args.get(), &errors));
        MI_CHECK(call_inst);
        MI_CHECK_EQUAL(errors, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> call_args(call_inst->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_call> arg1(
            call_args->get_expression<mi::neuraylib::IExpression_call>(1)); // this should be a call
        MI_CHECK(arg1);
        mi::base::Handle<const mi::neuraylib::IExpression_constant> arg2(
            call_args->get_expression<mi::neuraylib::IExpression_constant>(2)); // this should be a constant
        MI_CHECK(arg2);
    }
    {
        // test create_mdle
        mi::base::Handle<mi::IStructure> mdle_data(transaction->create<mi::IStructure>("Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name0(
            mdle_data->get_value<mi::IString>("prototype_name"));
        prototype_name0->set_c_str("mdl::test_compatibility::structure_test_mat(::test_compatibility::struct_one)");
        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c.get());
        MI_CHECK_EQUAL(0, mdle_data->set_value("defaults", args.get()));

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(neuray->get_api_component<mi::neuraylib::IMdle_api>());
        result = mdle_api->export_mdle(
            transaction, DIR_PREFIX "/compatibility_test.mdle", mdle_data.get(), context.get());
        MI_CHECK_EQUAL(result, 0);
    }
    {
        // test create_mdle
        mi::base::Handle<mi::IStructure> mdle_data(transaction->create<mi::IStructure>("Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name0(
            mdle_data->get_value<mi::IString>("prototype_name"));
        prototype_name0->set_c_str("mdl::test_compatibility::structure_test(::test_compatibility::struct_one)");
        mi::base::Handle<mi::neuraylib::IExpression_list> args(ef->create_expression_list());
        args->add_expression("v", c.get());
        MI_CHECK_EQUAL(0, mdle_data->set_value("defaults", args.get()));

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(neuray->get_api_component<mi::neuraylib::IMdle_api>());
        result = mdle_api->export_mdle(
            transaction, DIR_PREFIX "/compatibility_func_test.mdle", mdle_data.get(), context.get());
        MI_CHECK_EQUAL(result, 0);
    }
    {
        // test module builder (new material)

        // MDL 1.5 due to the cast operator

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::compatibility_new_material",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> param0_type( tf->create_bool());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "param0", param0_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> v_v_expr(
            ef->create_parameter( param0_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> v_args(
            ef->create_expression_list());
        v_args->add_expression( "v", v_v_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> v_expr(
            ef->create_direct_call( "mdl::test_compatibility::struct_return(bool)", v_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "v", v_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( "mdl::test_compatibility::structure_test_mat(::test_compatibility::struct_one)",
                body_args.get()));

        // create defaults

        mi::base::Handle<mi::neuraylib::IValue> param0_value(
            vf->create_bool( false));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        defaults->add_expression( "param0", param0_expr.get());

        // add the material

        result = module_builder->add_function(
            "my_new_material",
            body.get(),
            parameters.get(),
            defaults.get(),
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // test create_mdle

        // ::my_new_materials requires the module cache
        uninstall_external_resolver( mdl_configuration );

        mi::base::Handle<mi::IStructure> mdle_data(transaction->create<mi::IStructure>("Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name0(
            mdle_data->get_value<mi::IString>("prototype_name"));
        prototype_name0->set_c_str("mdl::compatibility_new_material::my_new_material(bool)");

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(neuray->get_api_component<mi::neuraylib::IMdle_api>());
        result = mdle_api->export_mdle(
            transaction, DIR_PREFIX "/compatibility_new_material.mdle", mdle_data.get(), context.get());
        MI_CHECK_CTX(context.get());
        MI_CHECK_EQUAL(result, 0);

        install_external_resolver( mdl_configuration );
    }
}

void check_named_temporaries( mi::neuraylib::ITransaction* transaction)
{
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::fd_named_temporaries(int)"));

        std::set<std::string> names;
        for( mi::Size i = 0, n = fd->get_temporary_count(); i < n; ++i) {
            const char* tn = fd->get_temporary_name( i);
            if( tn)
                names.insert( tn);
        }

        MI_CHECK( names.size() == 1);
        MI_CHECK( names.count( "a") == 1);
    }

    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_named_temporaries(float)"));

        std::set<std::string> names;
        for( mi::Size i = 0, n = md->get_temporary_count(); i < n; ++i) {
            const char* tn = md->get_temporary_name( i);
            if( tn)
                names.insert( tn);
        }

        MI_CHECK( names.size() == 11);
        MI_CHECK( names.count( "f0" ) == 1);
        MI_CHECK( names.count( "f1") == 1);
        MI_CHECK( names.count( "f2") == 1);
        MI_CHECK( names.count( "b") == 1);
        MI_CHECK( names.count( "c0") == 1);
        MI_CHECK( names.count( "c1") == 1);
        MI_CHECK( names.count( "s" ) == 1);
        MI_CHECK( names.count( "v0") == 1);
        MI_CHECK( names.count( "f3") == 1);
        MI_CHECK( names.count( "v1") == 1);
        MI_CHECK( names.count( "v2") == 1);

        // From test_mdl.mdl:
        // Keep the order of v0 and f3 to test that CSE does not identify the named expression f3 for
        // param0 with a previous unnamed expression (as part of v0) for param0.
        // float3 v0 = float3(param0, 0.f, 0.f);
        // float f3 = param0;

        bool found_v0 = false;

        for( mi::Size i = 0, n = md->get_temporary_count(); i < n; ++i) {

            const char* tn = md->get_temporary_name( i);
            if( tn && strcmp( tn, "v0") == 0) {

                found_v0 = true;
                mi::base::Handle<const mi::neuraylib::IExpression_direct_call> v0(
                    md->get_temporary<mi::neuraylib::IExpression_direct_call>( i));
                mi::base::Handle<const mi::neuraylib::IExpression_list> args( v0->get_arguments());
                mi::base::Handle<const mi::neuraylib::IExpression> arg0( args->get_expression( zero_size));
                // Not being a temporary implies not having a name. If this starts to fail, change
                // the test to retrieve the index, and check the temporary of that index does not
                // have a name. But temporaries for parameter references are unlikely.
                MI_CHECK_NOT_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_TEMPORARY);
                MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_PARAMETER);
            }
        }

        MI_CHECK( found_v0);

    }
}

void check_db_names( mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<const mi::IString> db_name;

    db_name = mdl_factory->get_db_module_name( nullptr);
    MI_CHECK( !db_name);
    db_name = mdl_factory->get_db_module_name( "state");
    MI_CHECK( !db_name);
    db_name = mdl_factory->get_db_module_name( "::state");
    MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdl::state");

    db_name = mdl_factory->get_db_definition_name( nullptr);
    MI_CHECK( !db_name);
    db_name = mdl_factory->get_db_definition_name( "state::normal()");
    MI_CHECK( !db_name);
    db_name = mdl_factory->get_db_definition_name( "::state::normal()");
    MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdl::state::normal()");
    db_name = mdl_factory->get_db_definition_name( "::/path/to/mod.mdle::fd(int,int)");
    MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdle::/path/to/mod.mdle::fd(int,int)");
}

void check_module_transformer(
    mi::neuraylib::IMdl_configuration* mdl_configuration,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
// The copy command below fails on Windows if source and target are on a network drive and when
// executed from the command-line via CTest (not from Visual Studio).
#ifndef MI_PLATFORM_WINDOWS
    // For strict relative imports and resource paths, all module and resources must be in the
    // same search paths. Copy the inputs such that the output can be exported into the same
    // directory.
    std::string path = MI::TEST::mi_src_path("prod/lib/neuray/mdl_module_transformer");
    fs::copy( path, DIR_PREFIX "/mdl_module_transformer", fs::copy_options::recursive);

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // We need to load the modules to be transformed without any optimization and with the original
    // resource paths. Therefore, we use a separate context for such calls.
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> load_context(
        mdl_factory->create_execution_context());
    load_context->set_option( "optimization_level", static_cast<mi::Sint32>( 0));
    MI_CHECK_CTX( load_context.get());
    load_context->set_option( "keep_original_resource_file_paths", true);
    MI_CHECK_CTX( load_context.get());

    uninstall_external_resolver( mdl_configuration);

    // check that builtin modules cannot be transformed
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::%3Cbuiltins%3E", context.get()));
        MI_CHECK( !module_transformer);
    }
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer( transaction, "mdl::state", context.get()));
        MI_CHECK( !module_transformer);
    }
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer( transaction, "mdl::stdlib", context.get()));
        MI_CHECK( !module_transformer);
    }
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer( transaction, "mdl::base", context.get()));
        MI_CHECK( !module_transformer);
    }

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer/import_declarations";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    mdl_impexp_api->load_module( transaction, "::p1::main_import_declarations", load_context.get());
    MI_CHECK_CTX( load_context.get());

    // check use_absolute_import_declarations() (without alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->use_absolute_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_absolute.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_absolute", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_absolute.orig.mdl",
            path + "/p1/exported_main_import_declarations_absolute.mdl"));
    }

    // check use_relative_import_declarations() (without alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->use_relative_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_relative.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_relative", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_relative.orig.mdl",
            path + "/p1/exported_main_import_declarations_relative.mdl"));
    }

    mdl_impexp_api->load_module(
        transaction, "::p1::main_import_declarations_alias", load_context.get());
    MI_CHECK_CTX( load_context.get());

    // check use_absolute_import_declarations() (with alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations_alias", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->use_absolute_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_alias_absolute.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_alias_absolute", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_alias_absolute.orig.mdl",
            path + "/p1/exported_main_import_declarations_alias_absolute.mdl"));
    }

    // check use_relative_import_declarations() (with alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations_alias", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->use_relative_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_alias_relative.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_alias_relative", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_alias_relative.orig.mdl",
            path + "/p1/exported_main_import_declarations_alias_relative.mdl"));
    }

    // check upgrade_mdl_version() w.r.t. alias removal
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations_alias", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->upgrade_mdl_version(
            mi::neuraylib::MDL_VERSION_1_8, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_alias_alias_removal.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_alias_alias_removal",
            context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_alias_alias_removal.orig.mdl",
            path + "/p1/exported_main_import_declarations_alias_alias_removal.mdl"));
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer/resource_file_paths";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    mdl_impexp_api->load_module( transaction, "::p1::main_resource_file_paths", load_context.get());
    MI_CHECK_CTX( load_context.get());

    // check use_absolute_resource_file_paths()
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_resource_file_paths", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->use_absolute_resource_file_paths(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_resource_file_paths_absolute.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_resource_file_paths_absolute", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_resource_file_paths_absolute.orig.mdl",
            path + "/p1/exported_main_resource_file_paths_absolute.mdl"));
    }

    // check use_relative_resource_file_paths()
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_resource_file_paths", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->use_relative_resource_file_paths(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_resource_file_paths_relative.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_resource_file_paths_relative", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_resource_file_paths_relative.orig.mdl",
            path + "/p1/exported_main_resource_file_paths_relative.mdl"));
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer/upgrade_mdl_version";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    // check upgrade_mdl_version() w.r.t. weak relative import declarations and resource file paths
    {
        mdl_impexp_api->load_module(
            transaction, "::p1::main_upgrade_mdl_version", load_context.get());
        MI_CHECK_CTX( load_context.get());
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_upgrade_mdl_version", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->upgrade_mdl_version(
            mi::neuraylib::MDL_VERSION_1_6, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/p1/exported_main_upgrade_mdl_version.mdl").c_str(), context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_upgrade_mdl_version", context.get());
        MI_CHECK_CTX( context.get());
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_upgrade_mdl_version.orig.mdl",
            path + "/p1/exported_main_upgrade_mdl_version.mdl"));
    }

    // check upgrade_mdl_version() w.r.t. updated function signatures

    struct Data { const char* from; const char* to; mi::neuraylib::Mdl_version to_enum; };
    Data data[] = { { "1_0", "1_3", mi::neuraylib::MDL_VERSION_1_3 },
                    { "1_0", "1_4", mi::neuraylib::MDL_VERSION_1_4 },
                    { "1_0", "1_5", mi::neuraylib::MDL_VERSION_1_5 },
                    { "1_0", "1_6", mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_3", "1_4", mi::neuraylib::MDL_VERSION_1_4 },
                    { "1_3", "1_5", mi::neuraylib::MDL_VERSION_1_5 },
                    { "1_3", "1_6", mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_4", "1_5", mi::neuraylib::MDL_VERSION_1_5 },
                    { "1_4", "1_6", mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_5", "1_6", mi::neuraylib::MDL_VERSION_1_6 }
                };

    for( const auto& d: data) {

        std::string input_mdl_name   = std::string( "::test_mdl_version_") + d.from;
        std::string input_db_name    = std::string( "mdl::test_mdl_version_") + d.from;
        std::string output_file_name = std::string( DIR_PREFIX)
                                           + "/mdl_module_transformer/upgrade_mdl_version"
                                           + "/exported_test_mdl_version_" + d.from
                                           + "_as_mdl_version_" + d.to + ".mdl";
        std::string output_mdl_name  = std::string( "::exported_test_mdl_version_") + d.from
                                           + "_as_mdl_version_" + d.to;

        mdl_impexp_api->load_module( transaction, input_mdl_name.c_str(), load_context.get());
        MI_CHECK_CTX( load_context.get());
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, input_db_name.c_str(), context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->upgrade_mdl_version( d.to_enum, context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module( output_file_name.c_str(), context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        MI_CHECK_CTX( context.get());
        mdl_impexp_api->load_module( transaction, output_mdl_name.c_str(), context.get());
        MI_CHECK_CTX( context.get());
        MI_CHECK_CTX( context.get());
        // No file comparison (too much code whose serialization in text form changes too often in
        // tiny details)
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));

    std::string core1_path = MI::TEST::mi_src_path( "shaders/mdl");
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( core1_path.c_str()));
    std::string core2_path = MI::TEST::mi_src_path( "../examples/mdl");
    mdl_configuration->add_mdl_path( core2_path.c_str());

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    mdl_impexp_api->load_module( transaction, "::nvidia::core_definitions", load_context.get());
    MI_CHECK_CTX( load_context.get());

    // check inlined_imported_modules()
    //
    // Run on ::nvidia::core_definitions (nothing really to inline, since it uses only stdlib and
    // base).
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::nvidia::core_definitions", context.get()));
        MI_CHECK_CTX( context.get());
        module_transformer->inline_imported_modules(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, /*omit_anno_origin*/ false,
            context.get());
        MI_CHECK_CTX( context.get());
        module_transformer->export_module(
            (path + "/exported_inlined_core_definitions.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context.get());
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::exported_inlined_core_definitions", context.get());
        MI_CHECK_CTX( context.get());
        // No file comparison (too much code, formatting changes cause frequent test failures).
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));
    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( core1_path.c_str()));
    mdl_configuration->remove_mdl_path( core2_path.c_str());

    install_external_resolver( mdl_configuration);
#endif // MI_PLATFORM_WINDOWS
}

struct Df_data
{
    mi::neuraylib::Df_data_kind kind;
    mi::Size rx;
    mi::Size ry;
    mi::Size rz;
};

void test_multiscatter_textures(mi::neuraylib::IMdl_backend_api* mdl_backend_api)
{
    std::vector<Df_data> df_data = {
        { mi::neuraylib::DFK_SIMPLE_GLOSSY_MULTISCATTER, 65, 64, 33 },
        { mi::neuraylib::DFK_BACKSCATTERING_GLOSSY_MULTISCATTER, 65, 64, 1 },
        { mi::neuraylib::DFK_BECKMANN_SMITH_MULTISCATTER, 65, 64, 33 },
        { mi::neuraylib::DFK_BECKMANN_VC_MULTISCATTER, 65, 64, 33 },
        { mi::neuraylib::DFK_GGX_SMITH_MULTISCATTER, 65, 64, 33 },
        { mi::neuraylib::DFK_GGX_VC_MULTISCATTER, 65, 64, 33 },
        { mi::neuraylib::DFK_WARD_GEISLER_MORODER_MULTISCATTER, 65, 64, 1 },
        { mi::neuraylib::DFK_SHEEN_MULTISCATTER, 65, 64, 1 }
    };
    const float* data = nullptr;
    mi::Size rx, ry, rz;
    for (const auto& entry : df_data) {
        data = mdl_backend_api->get_df_data_texture(
            entry.kind, rx, ry, rz);
        MI_CHECK(data);
        MI_CHECK_EQUAL(entry.rx, rx);
        MI_CHECK_EQUAL(entry.ry, ry);
        MI_CHECK_EQUAL(entry.rz, rz);
    }

    data = mdl_backend_api->get_df_data_texture(
        mi::neuraylib::DFK_NONE, rx, ry, rz);
    MI_CHECK(!data);

    data = mdl_backend_api->get_df_data_texture(
        mi::neuraylib::DFK_INVALID, rx, ry, rz);
    MI_CHECK(!data);
}

void check_mdl_mangle( mi::neuraylib::ITransaction *transaction)
{
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> def(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::test_compatibility::structure_test_func(::test_compatibility::struct_one)"));
        MI_CHECK(def);

        MI_CHECK_EQUAL_CSTR(
            def->get_mdl_mangled_name(),
            "_ZN18test_compatibility19structure_test_funcEN18test_compatibility10struct_oneE");
    }
    {
        mi::base::Handle<const mi::neuraylib::IFunction_definition> mat_def(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_0()"));
        MI_CHECK(mat_def);

        MI_CHECK_EQUAL_CSTR(
            mat_def->get_mdl_mangled_name(),
            "md_0");
    }
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    fs::remove_all( DIR_PREFIX);
    fs::create_directory( DIR_PREFIX);

    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK( factory);

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());
        MI_CHECK( mdl_factory);

        mi::base::Handle<mi::neuraylib::IMdl_archive_api> mdl_archive_api(
            neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());
        MI_CHECK( mdl_archive_api);

        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK( mdl_configuration);
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( DIR_PREFIX));
        MI_CHECK_EQUAL( 0, mdl_configuration->add_resource_path( DIR_PREFIX));

        mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
            neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
        MI_CHECK( mdl_backend_api);

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        MI_CHECK( mdl_impexp_api);

        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            neuray->get_api_component<mi::neuraylib::IImage_api>());
        MI_CHECK( image_api);

        mi::base::Handle<mi::neuraylib::IMdl_distiller_api> mdl_distiller_api(
            neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
        MI_CHECK( mdl_distiller_api);

        mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
            neuray->get_api_component<mi::neuraylib::IMdl_compiler>());
        MI_CHECK( mdl_compiler);

        result = mdl_compiler->add_builtin_module( "::mybuiltins", native_module);
        MI_CHECK_EQUAL( 0, result);
        result = mdl_impexp_api->load_module( transaction.get(), "::mybuiltins");
        MI_CHECK_EQUAL( 0, result);

        test_multiscatter_textures( mdl_backend_api.get());

        install_external_resolver( mdl_configuration.get() );

        // run the actual tests
        check_mdl_factory( mdl_factory.get());
        check_encoding_decoding( factory.get(), mdl_factory.get());
        check_mdl_import( mdl_configuration.get(), transaction.get(), mdl_impexp_api.get(), mdl_factory.get());
        check_itransaction_methods( transaction.get());
        create_function_calls_and_material_instances( transaction.get(), mdl_factory.get());
        check_imodule( transaction.get(), factory.get(), mdl_factory.get(), mdl_impexp_api.get());
        check_ifunction_definition( transaction.get(), mdl_factory.get());
        check_imaterial_definition( transaction.get(), mdl_factory.get());
        check_ifunction_call( transaction.get(), mdl_factory.get(), mdl_impexp_api.get());
        check_imaterial_instance( transaction.get(), mdl_factory.get(), mdl_impexp_api.get());
        check_variants( transaction.get(), mdl_factory.get());
        check_variants2( transaction.get(), mdl_factory.get());
        check_materials( transaction.get(), mdl_factory.get());
        check_removed_materials_and_functions( transaction.get(), mdl_factory.get());
        check_analyze_uniform_open_graphs( transaction.get(), mdl_factory.get());

        {
            mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(
                neuray->get_api_component<mi::neuraylib::IMdle_api>());
            check_create_module_utf8(
                mdl_configuration.get(), transaction.get(), mdl_factory.get(), mdl_impexp_api.get(), mdle_api.get());
        }

        check_mdl_reload( transaction.get(), mdl_factory.get(), mdl_impexp_api.get());
        check_mdl_reload_compiled_materials( transaction.get(), mdl_factory.get(), mdl_backend_api.get());

        check_import_elements_from_string( mdl_configuration.get(), transaction.get(), mdl_impexp_api.get()); // loads ::mdl_elements::test_misc
        #define ARGS transaction.get(), mdl_configuration.get(), mdl_impexp_api.get(), mdl_factory.get()
        check_mdl_export_reimport( ARGS, "mdl::non_existing", "non_existing_export.mdl", 0, 6002, 6002, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::123_check_unicode_new_materials", "123_check_unicode_new_materials_export.mdl", "::123_check_unicode_new_materials_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::123_check_unicode_new_variants", "123_check_unicode_new_variants_export.mdl", "::123_check_unicode_new_variants_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::base", "test_base_export.mdl", 0, 6004, 6004, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::" TEST_MDL, TEST_MDL_FILE "_export.mdl", "::" TEST_MDL "_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::mdl_elements::test_misc", "test_misc_export.mdl", "::test_misc_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::test_archives", "test_archives_export.mdl", "::test_archives_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::variants", "variants_export.mdl", "::variants_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::resources", "resources_export.mdl", "::resources_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::resources", "resources2_export.mdl", "::resources2_export_string", 0, 6011, true, false, false);
        check_mdl_export_reimport( ARGS, "mdl::resources", "resources3_export.mdl", "::resources3_export_string", 0, 6006, false, true, false);
        check_mdl_export_reimport( ARGS, "mdl::resources", "resources4_export.mdl", "::resources4_export_string", 0, 0, false, false, true);
        check_mdl_export_reimport( ARGS, "mdl::resources", "resources5_export.mdl", "::resources5_export_string", 0, 6006, false, true, true);
        check_mdl_export_reimport( ARGS, "mdl::new_materials", "new_materials_export.mdl", "::new_materials_export_string", 0, 0, false, false, false);
        check_mdl_export_reimport( ARGS, "mdl::from_string", "from_string_export.mdl", "::from_string_export_string", 0, 0, false, false, false);
        // ::imports_from_string requires the module cache
        uninstall_external_resolver( mdl_configuration.get());
        check_mdl_export_reimport( ARGS, "mdl::imports_from_string", "imports_from_string_export.mdl", "::imports_from_string_export_string", 0, 0, false, false, false);
        install_external_resolver( mdl_configuration.get());
        check_mdl_export_reimport( ARGS, "mdl::mybuiltins", "mybuiltins_export.mdl", 0, 6004, 6004, false, false, false);
        #undef ARGS

        check_mdl_types( transaction.get(), mdl_factory.get());
        check_enums( transaction.get(), mdl_factory.get());
        check_structs( transaction.get());
        check_indexing( transaction.get());
        check_immediate_arrays( transaction.get(), mdl_factory.get());
        check_deferred_arrays( transaction.get(), mdl_factory.get());
        check_type_binding( transaction.get(), mdl_factory.get());
        check_annotations( transaction.get(), mdl_factory.get());
        check_parameter_indices( transaction.get(), mdl_factory.get());
        check_material_parameter( transaction.get(), mdl_factory.get());
        check_reexported_function( transaction.get(), mdl_factory.get());
        check_icompiled_material( transaction.get(), factory.get(), mdl_factory.get(), mdl_impexp_api.get(), false);
        check_icompiled_material( transaction.get(), factory.get(), mdl_factory.get(), mdl_impexp_api.get(), true);
        check_overloaded_hashing(transaction.get(), factory.get(), mdl_factory.get(),
            mdl_impexp_api.get());
        check_matrix( transaction.get());
        check_deep_copy_of_defaults( transaction.get());
        check_immutable_defaults( transaction.get(), mdl_factory.get());
        check_set_get_value( transaction.get(), mdl_factory.get());
        check_wrappers( transaction.get(), mdl_factory.get());
        check_uniform_auto_varying( transaction.get(), mdl_factory.get());
        check_export_flag( transaction.get(), mdl_factory.get());
        check_backends( transaction.get(), mdl_backend_api.get(), mdl_factory.get());
        check_create_archive( transaction.get(), mdl_configuration.get(), mdl_archive_api.get());
        check_extract_archive( mdl_archive_api.get());
        check_get_manifest( mdl_archive_api.get());
        check_get_file( mdl_archive_api.get());
        check_resources( transaction.get(), mdl_factory.get());
        check_resources2( transaction.get(), mdl_archive_api.get(), image_api.get());
        check_type_compatibility( neuray, mdl_configuration.get(), transaction.get(), mdl_factory.get());
        check_named_temporaries( transaction.get());
        check_db_names( mdl_factory.get());
        check_module_transformer( mdl_configuration.get(), mdl_impexp_api.get(), transaction.get(), mdl_factory.get());
        check_mdl_mangle( transaction.get());
        MI_CHECK_EQUAL( 0, transaction->commit());

        check_module_references( database.get());

        uninstall_external_resolver( mdl_configuration.get() );
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_imdl_module )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK_EQUAL( 0, mdl_configuration->set_expose_names_of_let_expressions( true));

        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));


        // set module paths
        std::string path = MI::TEST::mi_src_path( "prod/lib");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        path = MI::TEST::mi_src_path( "prod/lib/neuray");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        path = MI::TEST::mi_src_path( "prod");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        path = MI::TEST::mi_src_path( "io/scene");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));


        // load plugins
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_dds));
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_mdl_distiller));
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        run_tests( neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get());
    }

    neuray = 0;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

#endif // __GNUC__

