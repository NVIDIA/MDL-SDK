/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// Code shared by MDL-related unit tests

#ifndef PROD_LIB_NEURAY_TEST_SHARED_MDL_H
#define PROD_LIB_NEURAY_TEST_SHARED_MDL_H

#include <filesystem>
#include <string>
#include <vector>

#include <boost/algorithm/string/replace.hpp>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_entity_resolver.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>


#include <base/system/test/i_test_auto_case.h>

#include "test_shared.h"

namespace fs = std::filesystem;

// === Macros ======================================================================================

// Enable this to install an external entity resolver for most tests.
//
#define EXTERNAL_ENTITY_RESOLVER

// Enable this define to change the external entity resolver such that it returns resources without
// filenames.
//
// #define RESOLVE_RESOURCES_WITHOUT_FILENAMES

// Enable this define to skip/adapt tests if the context option "resolve_resources" defaults to
// false. Note: this define does \em not change the value of the context option itself.
//
// #define RESOLVE_RESOURCES_FALSE

// Just to make the self-contained header check pass.
#ifndef DIR_PREFIX
#define DIR_PREFIX "DIR_PREFIX_not_set"
#endif

// True name of the ::test_mdl module. This allows to test various problematic names without
// changing every other line of the entire test.
#define TEST_MDL      "test_mdl%24"
#define TEST_MDL_FILE "test_mdl$"

// === Constants ===================================================================================

std::string mdle_path = MI::TEST::mi_src_path( "prod/lib/neuray") + "/test.mdle";

const std::string name_hoelzer = u8"H\u00f6lzer";
const std::string name_eiche   = u8"Eiche";
const std::string name_foehre  = u8"F\u00f6hre";
const std::string name_buche   = u8"Buche";
const std::string name_mdl     = u8"mdl";
const std::string name_keyword = u8"keyword";

#define TEST_NO_1_RUS \
    u8"\u0422\u0435\u0441\u0442\u041d\u043e\u043c\u0435\u0440\u041e\u0434\u0438\u043d\u0451"

const char* src_keyword = u8"mdl 1.8;\nexport material mat() = material();\n\n";
const char* src_eiche   = u8"mdl 1.8;\nexport material eichen_material() = material();\n";
const char* src_foehre  = u8"mdl 1.8;\nexport material kiefern_material() = material();\n";
const char* src_buche   = u8"mdl 1.8;\n"
    "import ::'H\u00f6lzer'::Eiche::eichen_material;\n"
    "import 'F\u00f6hre'::kiefern_material;\n"
    "import 'mdl'::keyword::mat;\n"
    "export material eiche() = ::'H\\U000000f6lzer'::'\\x45iche'::eichen_material();\n"
    "export material kiefer() = 'F\\u00f6hre'::kiefern_material();\n"
    "export material buche() = material();\n"
    "export material mat() = 'mdl'::keyword::mat();\n";
const char* src_rus = u8"mdl 1.6;\n"
    "import::state::*;\n"
    "import::df::*;\n"
    "export color fd_test(float a = 1.0) {\n"
    "    return color(state::texture_coordinate(0)) * color(a);\n"
    "}\n"
    "export material md_test(float a = 1.0) = material(\n"
    "    surface: material_surface(\n"
    "        scattering : df::diffuse_reflection_bsdf(tint : color(a)))\n"
    ");\n";

// === General utilities ===========================================================================

// Converts the initializer list of string literals into a dynamic array of IString elements.
mi::IArray* create_istring_array(
    mi::neuraylib::IFactory* factory, std::initializer_list<const char*> l)
{
    auto* result = factory->create<mi::IDynamic_array>( "String[]");
    mi::Size n = 0;

    for( auto element : l) {
        mi::base::Handle<mi::IString> s( factory->create<mi::IString>());
        s->set_c_str( element);
        result->set_length( ++n);
        result->set_element( n-1, s.get());
    }

    return result;
}

// Creates a function call of a definition and stores it in the database (assumming all parameter
// have defaults).
void do_create_function_call(
    mi::neuraylib::ITransaction* transaction,
    const char* definition_name,
    const char* call_name)
{
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>( definition_name));
    MI_CHECK( fd);
    mi::Sint32 result = -1;
    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( nullptr, &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 0, transaction->store( fc.get(), call_name));
}

// Table for encode() and decode() below.
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

// === Search paths ================================================================================


// Saves the current MDL search paths in the return value.
std::vector<std::string> get_mdl_paths( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    std::vector<std::string> v;
    mi::Size n = mdl_configuration->get_mdl_paths_length();
    for ( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const mi::IString> p( mdl_configuration->get_mdl_path( i));
        v.emplace_back( p->get_c_str());
    }
    return v;
}

// Sets the MDL search paths to the argument.
void set_mdl_paths( mi::neuraylib::INeuray* neuray, const std::vector<std::string>& mdl_paths)
{
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    mdl_configuration->clear_mdl_paths();
    for( const auto& mdl_path : mdl_paths)
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( mdl_path.c_str()));
}


// === Entity resolver =============================================================================

// Wrapper for mi::neuraylib::IMdl_resolved_module.
//
// Forwards all calls. Used by My_resolver.
class My_module : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_module>
{
public:
    My_module( mi::neuraylib::IMdl_resolved_module* impl)
      : m_impl( impl, mi::base::DUP_INTERFACE) { }
    const char* get_module_name() const final { return m_impl->get_module_name(); }
    const char* get_filename() const final { return m_impl->get_filename(); }
    mi::neuraylib::IReader* create_reader() const final { return m_impl->create_reader(); }

private:
    mi::base::Handle<IMdl_resolved_module> m_impl;
};

// Wrapper for mi::neuraylib::IMdl_resolved_resource.
//
// Forwards all calls unless RESOLVE_RESOURCES_WITHOUT_FILENAMES is set. Used by My_resolver.
class My_resource_element
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource_element>
{
public:
    My_resource_element(const mi::neuraylib::IMdl_resolved_resource_element* impl)
      : m_impl( impl, mi::base::DUP_INTERFACE) { }
    mi::Size get_frame_number() const final { return m_impl->get_frame_number(); }
    mi::Size get_count() const final { return m_impl->get_count(); }
    const char* get_mdl_file_path( mi::Size i) const final { return m_impl->get_mdl_file_path( i); }
#ifdef RESOLVE_RESOURCES_WITHOUT_FILENAMES
    const char* get_filename( mi::Size i) const final { return nullptr; }
#else
    const char* get_filename( mi::Size i) const final { return m_impl->get_filename( i); }
#endif
    mi::neuraylib::IReader* create_reader( mi::Size i) const final
    { return m_impl->create_reader( i); }
    mi::base::Uuid get_resource_hash( mi::Size i) const final
    { return m_impl->get_resource_hash( i); }
    bool get_uvtile_uv( mi::Size i, mi::Sint32 & u, mi::Sint32 & v) const final
    { return m_impl->get_uvtile_uv( i, u, v); }

private:
    mi::base::Handle<const mi::neuraylib::IMdl_resolved_resource_element> m_impl;
};

// Wrapper for mi::neuraylib::IMdl_resolved_resource.
//
// Forwards all calls unless RESOLVE_RESOURCES_WITHOUT_FILENAMES is set. Used by My_resolver.
class My_resource : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource>
{
public:
    My_resource( mi::neuraylib::IMdl_resolved_resource* impl)
      : m_impl( impl, mi::base::DUP_INTERFACE) { }
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
        mi::base::Handle<const mi::neuraylib::IMdl_resolved_resource_element> elem(
            m_impl->get_element( i));
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
      : m_impl( impl, mi::base::DUP_INTERFACE)
        { }

    void enable_user_data_check( bool value) { m_enable_user_data_check = value; }

    void check_user_data( mi::neuraylib::IMdl_execution_context* context)
    {
        if( !m_enable_user_data_check)
            return;

        mi::Sint32 result = 0;
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
        mi::neuraylib::IMdl_execution_context* context) final
    {
        check_user_data( context);

#if 0
        std::cerr << "module " << module_name << " "
                  << SAFE_STR( owner_file_path) << " "
                  << SAFE_STR( owner_name) << std::endl;
#endif
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
        mi::neuraylib::IMdl_execution_context* context) final
    {
        check_user_data( context);

#if 0
        std::cerr << "resource " << file_path << " "
                  << SAFE_STR( owner_file_path) << " "
                  << SAFE_STR( owner_name) << std::endl;
#endif
        mi::base::Handle<mi::neuraylib::IMdl_resolved_resource> result( m_impl->resolve_resource(
            file_path, owner_file_path, owner_name, pos_line, pos_column, context));
        if( !result)
            return nullptr;
        return new My_resource( result.get());
    }

private:
    mi::base::Handle<IMdl_entity_resolver> m_impl;
    bool m_enable_user_data_check = false;
};

void install_external_resolver( mi::neuraylib::INeuray* neuray, bool enable_user_data_check = false)
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

    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    // Uninstall a pontentially already installed entity resolver to avoid wrapping it twice.
    mdl_configuration->set_entity_resolver( nullptr);
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> resolver(
        mdl_configuration->get_entity_resolver());
    // Wrap the resolver returned by the API, such that we can easily intercept all calls (and
    // e.g. modify them if RESOLVE_RESOURCES_WITHOUT_FILENAMES is defined).
    auto* my_resolver = new My_resolver( resolver.get());
    my_resolver->enable_user_data_check( enable_user_data_check);
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> modified_resolver( my_resolver);
    mdl_configuration->set_entity_resolver( modified_resolver.get());
#endif
}

void uninstall_external_resolver( mi::neuraylib::INeuray* neuray)
{
#ifdef EXTERNAL_ENTITY_RESOLVER
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mdl_configuration->set_entity_resolver( nullptr);
#endif
}

// === MDLE serialization ==========================================================================

// Demo implementation of IMdle_serialization_callback swapping upper and lower-case characters
// (except for the file extension). Adds "FOO" prefix to test handling of string length changes.
class Mdle_serialization_callback
  : public mi::base::Interface_implement<mi::neuraylib::IMdle_serialization_callback>
{
public:
    Mdle_serialization_callback( mi::neuraylib::IFactory* factory)
      : m_factory( factory, mi::base::DUP_INTERFACE) { }

    const mi::IString* get_serialized_filename( const char* filename) const final
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
        auto* result = m_factory->create<mi::IString>();
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

    const mi::IString* get_deserialized_filename( const char* filename) const final
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

        auto* result = m_factory->create<mi::IString>();
        result->set_c_str( s.c_str());
        return result;
    }

private:
    mi::base::Handle<mi::neuraylib::IFactory> m_factory;
};

// === Import ======================================================================================


// Imports the module in the given transaction.
void import_mdl_module(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    const char* module_name,
    mi::Sint32 expected_result)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Pass a custom interface as user data to the context in order to receive it in callbacks.
    mi::base::Handle<IMdl_execution_context_user_data> user_data(
        new Mdl_execution_context_user_data());
    context->set_option( "user_data", user_data.get());

    mi::Sint32 result = mdl_impexp_api->load_module( transaction, module_name, context.get());
    MI_CHECK_EQUAL( expected_result, result);
    if( expected_result >= 0)
        MI_CHECK_CTX( context);
}



// Loads the primary module "::test_mdl$" plus "test_mdl2" and "test_mdl3".
//
// Needs "prod/lib/neuray" and "io/scene" in the MDL search path.
void load_primary_modules( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    install_external_resolver( neuray, /*enable_user_data_check*/ true);

    // test_mdl.mdl, relative to search path, top-level, global scope
    import_mdl_module( transaction, neuray, "::" TEST_MDL, 0);
    MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::" TEST_MDL));

    import_mdl_module( transaction, neuray, "::test_mdl2", 0);
    MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::test_mdl2"));

    import_mdl_module( transaction, neuray, "::test_mdl3", 0);
    MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::test_mdl3"));

    // Disable the user-data check again. import_mdl_module() prepares the context correspondingly,
    // but other ad-hoc contexts used elsewhere do not.
    install_external_resolver( neuray, /*enable_user_data_check*/ false);
}


// Instantiates most materials and functions from the primary module using the defaults.
void create_function_calls(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

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
            "mdl::" TEST_MDL "::fc_parameter_references"
    };

    for( mi::Size i = 0; i < sizeof( functions) / sizeof( const char*); i+=2)
        do_create_function_call( transaction, functions[i], functions[i+1]);

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
        "mdl::" TEST_MDL "::md_folding()",
            "mdl::" TEST_MDL "::mi_folding",
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
        "mdl::" TEST_MDL "::md_texture(texture_2d)",
            "mdl::" TEST_MDL "::mi_texture",
        "mdl::" TEST_MDL "::md_light_profile(light_profile)",
            "mdl::" TEST_MDL "::mi_light_profile",
        "mdl::" TEST_MDL "::md_bsdf_measurement(bsdf_measurement)",
            "mdl::" TEST_MDL "::mi_bsdf_measurement",
        "mdl::" TEST_MDL "::md_baking()",
            "mdl::" TEST_MDL "::mi_baking",
        "mdl::" TEST_MDL "::md_class_baking(float,::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::mi_class_baking",
        "mdl::" TEST_MDL "::fd_bar_struct_to_int(::" TEST_MDL "::bar_struct)",
            "mdl::" TEST_MDL "::fc_bar_struct_to_int",
        "mdl::" TEST_MDL "::md_aov(float,float)",
            "mdl::" TEST_MDL "::mi_aov"
    };

    for( mi::Size i = 0; i < sizeof( materials) / sizeof( const char*); i+=2)
        do_create_function_call( transaction, materials[i], materials[i+1]);
}

void create_unicode_modules()
{
    std::string dirname     = std::string( DIR_PREFIX) + "/" + name_hoelzer;
    std::string dirname_mdl = dirname + "/" + name_mdl;
    std::string dirname_rus = std::string( DIR_PREFIX) + "/" + TEST_NO_1_RUS;

    std::string filename_eiche   = dirname + "/" + name_eiche + ".mdl";
    std::string filename_buche   = dirname + "/" + name_buche + ".mdl";
    std::string filename_foehre  = dirname + "/" + name_foehre + ".mdl";
    std::string filename_keyword = dirname_mdl + "/" + name_keyword + ".mdl";
    std::string filename_rus     = dirname_rus + "/1_module.mdl";

    fs::remove_all( fs::u8path( dirname));
    if( fs::create_directory( fs::u8path( dirname))) {
        std::ofstream( fs::u8path( filename_eiche)) << src_eiche << std::endl;
        std::ofstream( fs::u8path( filename_foehre)) << src_foehre << std::endl;
        std::ofstream( fs::u8path( filename_buche)) << src_buche << std::endl;
    }

    fs::remove_all( fs::u8path( dirname_mdl));
    if( fs::create_directory( fs::u8path( dirname_mdl))) {
        std::ofstream( fs::u8path( filename_keyword)) << src_keyword << std::endl;
    }

    fs::remove_all( fs::u8path( dirname_rus));
    if( fs::create_directory( fs::u8path( dirname_rus))) {
        std::ofstream( fs::u8path( filename_rus)) << src_rus << std::endl;
    }
}


// Loads various other modules uses by later tests. Also performs some checks for the import
// functionality.
//
// Needs  "prod/lib/neuray", "io/scene", "prod", and "prod/lib" in the MDL search path.
void load_secondary_modules(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    install_external_resolver( neuray, /*enable_user_data_check*/ true);

    {
        // base.mdl, relative to search path, top-level, global scope
        import_mdl_module( transaction, neuray, "::base", 1);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::base"));
    }
    {
        // test_mdl.mdl, relative to search path, sub-subdirectory, own scope
        mi::base::Handle<mi::neuraylib::IScope> local_scope( database->create_scope( nullptr, 1));
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            local_scope->create_transaction());
        import_mdl_module( transaction.get(), neuray, "::lib::neuray::" TEST_MDL, 0);
        MI_CHECK_EQUAL( 1, transaction->get_privacy_level( "mdl::lib::neuray::" TEST_MDL));
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::lib::neuray::" TEST_MDL));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL_CSTR( "::lib::neuray::" TEST_MDL, c_module->get_mdl_name());
        MI_CHECK_EQUAL_CSTR( "" TEST_MDL, c_module->get_mdl_simple_name());
        MI_CHECK_EQUAL( 2, c_module->get_mdl_package_component_count());
        MI_CHECK_EQUAL_CSTR( "lib", c_module->get_mdl_package_component_name( 0));
        MI_CHECK_EQUAL_CSTR( "neuray", c_module->get_mdl_package_component_name( 1));
        c_module.reset();
        transaction->commit();
    }
    {
        // test_mdl.mdl, relative to search path, subdirectory, own scope
        mi::base::Handle<mi::neuraylib::IScope> local_scope( database->create_scope( nullptr, 1));
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            local_scope->create_transaction());
        import_mdl_module( transaction.get(), neuray, "::neuray::" TEST_MDL, 0);
        MI_CHECK_EQUAL( 1, transaction->get_privacy_level( "mdl::neuray::" TEST_MDL));
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::neuray::" TEST_MDL));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL_CSTR( "::neuray::" TEST_MDL, c_module->get_mdl_name());
        MI_CHECK_EQUAL_CSTR( "" TEST_MDL, c_module->get_mdl_simple_name());
        MI_CHECK_EQUAL( 1, c_module->get_mdl_package_component_count());
        MI_CHECK_EQUAL_CSTR( "neuray", c_module->get_mdl_package_component_name( 0));
        c_module.reset();
        transaction->commit();
    }
    {
        // test_mdl.mdl, absolute URI, global scope (fails)
        std::string path = MI::TEST::mi_src_path( "prod/lib/neuray") + "/" TEST_MDL ".mdl";
        import_mdl_module( transaction, neuray, path.c_str(), -1);
    }
    {
        // state.mdl, relative to search path, top-level, own scope (standard library, file does not
        // exist(!))
        mi::base::Handle<mi::neuraylib::IScope> local_scope( database->create_scope( nullptr, 1));
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            local_scope->create_transaction());
        import_mdl_module( transaction.get(), neuray, "::state", 0);
        MI_CHECK_EQUAL( 1, transaction->get_privacy_level( "mdl::state"));
        transaction->commit();
    }
    {
        // test_archives.mdl, relative to search path, top-level, global scope
        import_mdl_module( transaction, neuray, "::test_archives", 0);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::test_archives"));
    }
    {
        // test_compatibility.mdl, relative to search path, top-level, global scope
        import_mdl_module( transaction, neuray, "::test_compatibility", 0);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::test_compatibility"));
    }
    {
        // test.mdle
        import_mdl_module( transaction, neuray, mdle_path.c_str(), 0);
        mi::base::Handle<const mi::IString> s( mdl_factory->get_db_module_name( mdle_path.c_str()));
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( s->get_c_str()));
    }
    {
        create_unicode_modules();

        // load module with unicode package/module names
        import_mdl_module( transaction, neuray, "::" TEST_NO_1_RUS "::1_module", 0);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::" TEST_NO_1_RUS "::1_module"));

        std::string mdl_name =    "::" + name_hoelzer + "::" + name_buche;
        std::string db_name  = "mdl::" + name_hoelzer + "::" + name_buche;
        import_mdl_module( transaction, neuray, mdl_name.c_str(), 0);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( db_name.c_str()));
    }
    {
        // prepare for mdl reload test
        fs::copy_file(
            fs::u8path( MI::TEST::mi_src_path( "prod/lib/neuray") + "/test_mdl_reload_orig.mdl"),
            fs::u8path( DIR_PREFIX "/test_mdl_reload.mdl"));

        import_mdl_module( transaction, neuray, "::test_mdl_reload", 0);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::test_mdl_reload"));

        import_mdl_module( transaction, neuray, "::test_mdl_reload_import", 0);
        MI_CHECK_EQUAL( 0, transaction->get_privacy_level( "mdl::test_mdl_reload_import"));

        const char* module_source = "mdl 1.0; export material some_material() = material();";
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        mi::Sint32 result = mdl_impexp_api->load_module_from_string(
            transaction, "::test_mdl_reload_from_string", module_source, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);
    }

    // Disable the user-data check again. import_mdl_module() prepares the context correspondingly,
    // but other ad-hoc contexts used elsewhere do not.
    install_external_resolver( neuray, /*enable_user_data_check*/ false);
}


// === Export and Re-import ========================================================================

// Data for check_mdl_export_reimport()
struct Exreimp_data
{
    // Module to export.
    std::string module_db_name;

    // MDL module name of the exported module.
    //
    // The suffix "_export" is used for added for file-based exports.
    // The suffix "_export_string" is used for for string-based exports.
    std::string export_mdl_name;

    mi::Sint32 error_number_file = 0;
    mi::Sint32 error_number_string = 0;
    bool modify_mdl_paths = false;
    bool bundle_resources = false;
    bool export_resources_with_module_prefix = false;
};


// Exports a module as file/string and imports it again.
void do_check_mdl_export_reimport(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray, const Exreimp_data& d)
{
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    // ::imports_from_string requires the module cache
    if( d.module_db_name == "mdl::imports_from_string")
        uninstall_external_resolver( neuray);

    std::string file_name = decode( d.export_mdl_name.substr( 2)) + "_export.mdl";

    std::string full_file_name
        = std::string( DIR_PREFIX) + dir_sep + "exported" + dir_sep + file_name;
    fs::create_directory( fs::u8path( DIR_PREFIX "/exported"));

    // exporter options
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    context->set_option( "bundle_resources", d.bundle_resources);
    context->set_option(
        "export_resources_with_module_prefix", d.export_resources_with_module_prefix);

    std::vector<std::string> mdl_paths;

    if( d.modify_mdl_paths) {
        mdl_paths = get_mdl_paths( neuray);
        mdl_configuration->clear_mdl_paths();
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( DIR_PREFIX));
        std::string path = MI::TEST::mi_src_path( "prod/lib/neuray");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
    }

    // file-based export
    mi::Sint32 result = mdl_impexp_api->export_module(
        transaction, d.module_db_name.c_str(), full_file_name.c_str(), context.get());
    if( d.error_number_file == 0)
        MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( -d.error_number_file, result);

    // re-import from file
    if( d.error_number_file == 0) {
        std::string mdl_module_name_from_file = "::exported::";
        mdl_module_name_from_file += encode( file_name);
        mdl_module_name_from_file.resize( mdl_module_name_from_file.size()-4); // remove ".mdl"
        result = mdl_impexp_api->load_module(
            transaction, mdl_module_name_from_file.c_str(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }

#ifndef RESOLVE_RESOURCES_FALSE
    // string-based export
    mi::base::Handle<mi::IString> exported_module(
        transaction->create<mi::IString>( "String"));
    result = mdl_impexp_api->export_module_to_string(
        transaction, d.module_db_name.c_str(), exported_module.get(), context.get());
    if( d.error_number_string == 0)
        MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( -d.error_number_string, result);

    // re-import from string
    if( d.error_number_string == 0) {
        std::string mdl_module_name_from_string = d.export_mdl_name + "_export_string";
        result = mdl_impexp_api->load_module_from_string( transaction,
            mdl_module_name_from_string.c_str(), exported_module->get_c_str(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
#endif // RESOLVE_RESOURCES_FALSE

    if( d.modify_mdl_paths)
        set_mdl_paths( neuray, mdl_paths);

    // ::imports_from_string requires the module cache
    if( d.module_db_name == "mdl::imports_from_string")
        install_external_resolver( neuray);
}

// Exports a canvas.
void export_canvas(
    mi::neuraylib::INeuray* neuray, const char* filename, const mi::neuraylib::ICanvas* canvas)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::Sint32 result = mdl_impexp_api->export_canvas( filename, canvas);
    MI_CHECK_EQUAL( 0, result);
}


#endif // PROD_LIB_NEURAY_TEST_SHARED_MDL_H

