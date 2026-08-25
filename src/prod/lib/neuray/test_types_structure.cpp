/******************************************************************************
 * Copyright (c) 2010-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
 ** \brief Test for mi::IStructure and mi::IStructure_decl
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>
#include <vector>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/iextension_api.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/istructure_decl.h>
#include <mi/neuraylib/itransaction.h>

#include <mi/neuraylib/iattribute_container.h>


#include "test_shared.h"

const char* decl_simple_type_names[] = {
    "Boolean", "Sint8", "Sint16", "Sint32", "Sint64",
    "Float32", "Float64", "String", "Uuid", "Void", "Pointer<Interface>", "Const_pointer<Interface>",
    "Ref", "Test_enum" };

const char* decl_simple_member_names[] = {
    "m_boolean", "m_sint8", "m_sint16", "m_sint32", "m_sint64",
    "m_float32", "m_float64", "m_string", "m_uuid", "m_void", "m_pointer", "m_const_pointer",
    "m_ref", "m_test_enum" };

const char* decl_simple_attr_type_names[] = {
    "Boolean", "Sint8", "Sint16", "Sint32", "Sint64",
    "Float32", "Float64", "String", "Ref", "Test_enum" };

const char* decl_simple_attr_member_names[] = {
    "m_boolean", "m_sint8", "m_sint16", "m_sint32", "m_sint64",
    "m_float32", "m_float64", "m_string", "m_ref", "m_test_enum" };

const char* decl_compound_type_names[] = {
    "Boolean<2>", "Boolean<3>", "Boolean<4>",
    "Sint32<2>",  "Sint32<3>",  "Sint32<4>",
    "Float32<2>", "Float32<3>", "Float32<4>",
    "Float64<2>", "Float64<3>", "Float64<4>",
    "Boolean<2,2>", "Boolean<2,3>", "Boolean<2,4>", "Boolean<3,2>", "Boolean<3,3>", "Boolean<3,4>", "Boolean<4,2>", "Boolean<4,3>", "Boolean<4,4>",
    "Sint32<2,2>",  "Sint32<2,3>",  "Sint32<2,4>",  "Sint32<3,2>",  "Sint32<3,3>",  "Sint32<3,4>",  "Sint32<4,2>",  "Sint32<4,3>",  "Sint32<4,4>",
    "Float32<2,2>", "Float32<2,3>", "Float32<2,4>", "Float32<3,2>", "Float32<3,3>", "Float32<3,4>", "Float32<4,2>", "Float32<4,3>", "Float32<4,4>",
    "Float64<2,2>", "Float64<2,3>", "Float64<2,4>", "Float64<3,2>", "Float64<3,3>", "Float64<3,4>", "Float64<4,2>", "Float64<4,3>", "Float64<4,4>",
    "Color", "Color3", "Spectrum", "Bbox3"};

const char* decl_compound_member_names[] = {
    "m_boolean_2", "m_boolean_3", "m_boolean_4",
    "m_sint32_2",  "m_sint32_3",  "m_sint32_4",
    "m_float32_2", "m_float32_3", "m_float32_4",
    "m_float64_2", "m_float64_3", "m_float64_4",
    "m_boolean_2_2", "m_boolean_2_3", "m_boolean_2_4", "m_boolean_3_2", "m_boolean_3_3", "m_boolean_3_4", "m_boolean_4_2", "m_boolean_4_3", "m_boolean_4_4",
    "m_sint32_2_2",  "m_sint32_2_3",  "m_sint32_2_4",  "m_sint32_3_2",  "m_sint32_3_3",  "m_sint32_3_4",  "m_sint32_4_2",  "m_sint32_4_3",  "m_sint32_4_4",
    "m_float32_2_2", "m_float32_2_3", "m_float32_2_4", "m_float32_3_2", "m_float32_3_3", "m_float32_3_4", "m_float32_4_2", "m_float32_4_3", "m_float32_4_4",
    "m_float64_2_2", "m_float64_2_3", "m_float64_2_4", "m_float64_3_2", "m_float64_3_3", "m_float64_3_4", "m_float64_4_2", "m_float64_4_3", "m_float64_4_4",
    "m_color", "m_color3", "m_spectrum", "m_bbox3"};

const char* decl_compound_attr_type_names[] = {
    "Boolean<2>", "Boolean<3>", "Boolean<4>",
    "Sint32<2>",  "Sint32<3>",  "Sint32<4>",
    "Float32<2>", "Float32<3>", "Float32<4>",
    "Float64<2>", "Float64<3>", "Float64<4>",
    "Float32<2,2>", "Float32<2,3>", "Float32<2,4>", "Float32<3,2>", "Float32<3,3>", "Float32<3,4>", "Float32<4,2>", "Float32<4,3>", "Float32<4,4>",
    "Float64<4,4>",
    "Color", "Color3", "Spectrum"};

const char* decl_compound_attr_member_names[] = {
    "m_boolean_2", "m_boolean_3", "m_boolean_4",
    "m_sint32_2",  "m_sint32_3",  "m_sint32_4",
    "m_float32_2", "m_float32_3", "m_float32_4",
    "m_float64_2", "m_float64_3", "m_float64_4",
    "m_float32_2_2", "m_float32_2_3", "m_float32_2_4", "m_float32_3_2", "m_float32_3_3", "m_float32_3_4", "m_float32_4_2", "m_float32_4_3", "m_float32_4_4",
    "m_float64_4_4",
    "m_color", "m_color3", "m_spectrum"};

const char* decl_collection_type_names[] = {
    "Sint32[3]", "Sint32[]", "Map<Sint32>", "Map<Interface>" };

const char* decl_collection_member_names[] = {
    "m_sint32_3", "m_sint32_0", "m_map_sint32", "m_map_interface" };

const char* decl_collection_attr_type_names[] = {
    "Sint32[3]", "Sint32[]" };

const char* decl_collection_attr_member_names[] = {
    "m_sint32_3", "m_sint32_0" };

const char* decl_nested_type_names[] = {
    "Simple", "Compound", "Collection"  };

const char* decl_nested_member_names[] = {
    "m_simple", "m_compound", "m_collection" };

const char* decl_nested_attr_type_names[] = {
    "Simple_attr", "Compound_attr", "Collection_attr",
    "Simple_attr[3]", "Compound_attr[3]", "Collection_attr[3]",
    "Simple_attr[]", "Compound_attr[]", "Collection_attr[]" };

const char* decl_nested_attr_member_names[] = {
    "m_simple", "m_compound", "m_collection",
    "m_simple_array", "m_compound_array", "m_collection_array",
    "m_simple_darray", "m_compound_darray", "m_collection_darray" };

const char* decl_approx_type_names[] = {
    "Float32", "Float32", "Sint8", "Sint8", "Float32" };

const char* decl_approx_member_names[] = {
    "const_u", "const_v", "method", "base_method", "quality" };

const char* decl_section_plane_type_names[] = {
    "Float32<3>", "Float32<3>", "Boolean", "Boolean" };

const char* decl_section_plane_member_names[] = {
    "origin", "normal", "clip_light", "disabled" };


const char** type_names[] = {
    decl_simple_type_names, decl_simple_attr_type_names,
    decl_compound_type_names, decl_compound_attr_type_names,
    decl_collection_type_names, decl_collection_attr_type_names,
    decl_nested_type_names, decl_nested_attr_type_names,
    decl_approx_type_names,
    decl_section_plane_type_names,
    };

const char** member_names[] = {
    decl_simple_member_names, decl_simple_attr_member_names,
    decl_compound_member_names, decl_compound_attr_member_names,
    decl_collection_member_names, decl_collection_attr_member_names,
    decl_nested_member_names, decl_nested_attr_member_names,
    decl_approx_member_names,
    decl_section_plane_member_names,
    };

mi::Size decl_lengths[] = {
    sizeof( decl_simple_type_names)                  / sizeof( const char*),
    sizeof( decl_simple_attr_type_names)             / sizeof( const char*),
    sizeof( decl_compound_type_names)                / sizeof( const char*),
    sizeof( decl_compound_attr_type_names)           / sizeof( const char*),
    sizeof( decl_collection_type_names)              / sizeof( const char*),
    sizeof( decl_collection_attr_type_names)         / sizeof( const char*),
    sizeof( decl_nested_type_names)                  / sizeof( const char*),
    sizeof( decl_nested_attr_type_names)             / sizeof( const char*),
    sizeof( decl_approx_type_names)                  / sizeof( const char*),
    sizeof( decl_section_plane_type_names)           / sizeof( const char*),
    };

const char* decl_names[] = {
    "Simple", "Simple_attr",
    "Compound", "Compound_attr",
    "Collection", "Collection_attr",
    "Nested", "Nested_attr",
    "Approx_like",
    "Section_plane_like",
    };

void set_simple( mi::neuraylib::ITransaction* transaction, mi::IData_simple* data, mi::Size i)
{
    mi::base::Handle<mi::INumber> number( data->get_interface<mi::INumber>());
    mi::base::Handle<mi::IString> string( data->get_interface<mi::IString>());
    mi::base::Handle<mi::IRef> ref( data->get_interface<mi::IRef>());
    mi::base::Handle<mi::IEnum> enum_( data->get_interface<mi::IEnum>());

    if( number)
        number->set_value( i);
    else if( string)
        string->set_c_str( "foo");
    else if( ref) {
        MI_CHECK_EQUAL( 0, ref->set_reference( i%2 ? "options1" : "options0"));
    } else if( enum_) {
        MI_CHECK_EQUAL( 0, enum_->set_value_by_name( i%2 ? "ONE" : "ZERO"));
    }
}

void check_simple( mi::neuraylib::ITransaction* transaction, mi::IData_simple* data, mi::Size i)
{
    mi::base::Handle<mi::INumber> number( data->get_interface<mi::INumber>());
    mi::base::Handle<mi::IString> string( data->get_interface<mi::IString>());
    mi::base::Handle<mi::IRef> ref( data->get_interface<mi::IRef>());
    mi::base::Handle<mi::IEnum> enum_( data->get_interface<mi::IEnum>());

    if( number)
        MI_CHECK_EQUAL( i % 256, number->get_value<mi::Size>() % 256);
    else if( string)
        MI_CHECK_EQUAL_CSTR( "foo", string->get_c_str());
    else if( ref) {
        const char* name = ref->get_reference_name();
        MI_CHECK( name);
        MI_CHECK_EQUAL_CSTR( name, i%2 ? "options1" : "options0");
    } else if( enum_) {
        const char* name = enum_->get_value_by_name();
        MI_CHECK( name);
        MI_CHECK_EQUAL_CSTR( name, i%2 ? "ONE" : "ZERO");
    }
}

void set_simple_structure( mi::neuraylib::ITransaction* transaction, mi::IStructure* structure, mi::Size factor = 1)
{
    mi::Size n = structure->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IData_simple> data( structure->get_value<mi::IData_simple>( i));
        set_simple( transaction, data.get(), factor * i);
    }
}

void check_simple_structure( mi::neuraylib::ITransaction* transaction, mi::IStructure* structure, mi::Size factor = 1)
{
    mi::Size n = structure->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IData_simple> data( structure->get_value<mi::IData_simple>( i));
        check_simple( transaction, data.get(), factor * i);
    }
}

void set_collection_structure( mi::neuraylib::ITransaction* transaction, mi::IStructure* structure, mi::Size factor = 1)
{
    mi::Size n = structure->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IData_collection> data_collection( structure->get_value<mi::IData_collection>( i));
        for( mi::Size j = 0; j < data_collection->get_length(); ++j) {
            mi::base::Handle<mi::INumber> number( data_collection->get_value<mi::INumber>( j));
            number->set_value( factor * j);
        }
    }
}

void check_collection_structure( mi::neuraylib::ITransaction* transaction, mi::IStructure* structure, mi::Size factor = 1)
{
    mi::Size n = structure->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IData_collection> data_collection( structure->get_value<mi::IData_collection>( i));
        for( mi::Size j = 0; j < data_collection->get_length(); ++j) {
            mi::base::Handle<mi::INumber> number( data_collection->get_value<mi::INumber>( j));
            mi::base::Handle<mi::IBoolean> boolean( number->get_interface<mi::IBoolean>());
            // skip test for collections with IBoolean members which can only represent 0 and 1
            if( boolean)
                continue;
            MI_CHECK_EQUAL( (factor * j) % 256, number->get_value<mi::Size>() % 256);
        }
    }
}

void set_nested_structure( mi::neuraylib::ITransaction* transaction, mi::IStructure* structure, mi::Size factor = 1)
{
    mi::base::Handle<mi::IStructure> simple( structure->get_value<mi::IStructure>( "m_simple"));
    set_simple_structure( transaction, simple.get());
    mi::base::Handle<mi::IStructure> compound( structure->get_value<mi::IStructure>( "m_compound"));
    set_collection_structure( transaction, compound.get());
    mi::base::Handle<mi::IArray> array( structure->get_value<mi::IArray>( "m_simple_array"));
    mi::Size n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IStructure> element( array->get_value<mi::IStructure>( i));
        set_simple_structure( transaction, element.get(), factor * i);
    }
    array = structure->get_value<mi::IArray>( "m_compound_array");
    n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IStructure> element( array->get_value<mi::IStructure>( i));
        set_collection_structure( transaction, element.get(), factor * i);
    }
    array = structure->get_value<mi::IArray>( "m_collection_array");
    n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IStructure> element( array->get_value<mi::IStructure>( i));
        set_collection_structure( transaction, element.get(), factor * i);
    }
}

void check_nested_structure( mi::neuraylib::ITransaction* transaction, mi::IStructure* structure, mi::Size factor = 1)
{
    mi::base::Handle<mi::IStructure> simple( structure->get_value<mi::IStructure>( "m_simple"));
    check_simple_structure( transaction, simple.get());
    mi::base::Handle<mi::IStructure> compound( structure->get_value<mi::IStructure>( "m_compound"));
    check_collection_structure( transaction, compound.get());
    mi::base::Handle<mi::IArray> array( structure->get_value<mi::IArray>( "m_simple_array"));
    mi::Size n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IStructure> element( array->get_value<mi::IStructure>( i));
        check_simple_structure( transaction, element.get(), factor * i);
    }
    array = structure->get_value<mi::IArray>( "m_compound_array");
    n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IStructure> element( array->get_value<mi::IStructure>( i));
        check_collection_structure( transaction, element.get(), factor * i);
    }
    array = structure->get_value<mi::IArray>( "m_collection_array");
    n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IStructure> element( array->get_value<mi::IStructure>( i));
        check_collection_structure( transaction, element.get(), factor * i);
    }
}

void test_decl( mi::neuraylib::IExtension_api* extension_api, mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::IStructure_decl> decl(
        transaction->create<mi::IStructure_decl>( "Structure_decl"));
    MI_CHECK( decl);

    // test IStructure_decl, other methods are tested later
    MI_CHECK_EQUAL( -1, decl->add_member( 0, "m_foo"));
    MI_CHECK_EQUAL( -1, decl->add_member( "Foo", 0));
    MI_CHECK_EQUAL(  0, decl->add_member( "Sint32", "m_foo"));
    MI_CHECK_EQUAL( -2, decl->add_member( "Sint32", "m_foo"));

    MI_CHECK_EQUAL( -1, decl->remove_member( 0));
    MI_CHECK_EQUAL(  0, decl->remove_member( "m_foo"));
    MI_CHECK_EQUAL( -2, decl->remove_member( "m_foo"));

    MI_CHECK_EQUAL(  nullptr, decl->get_structure_type_name());

    // test IExtension_api::register_structure_decl()
    MI_CHECK_EQUAL(  0, extension_api->register_structure_decl( "Test1", decl.get()));
    MI_CHECK_EQUAL( -1, extension_api->register_structure_decl( "Test1", decl.get()));
    MI_CHECK_EQUAL( -2, extension_api->register_structure_decl( 0, decl.get()));
    MI_CHECK_EQUAL( -2, extension_api->register_structure_decl( "Test1", 0));
    MI_CHECK_EQUAL( -4, extension_api->register_structure_decl( "Invalid_structure_name[]<>{};", decl.get()));

    // test IFactory::get_structure_decl()
    mi::base::Handle<const mi::IStructure_decl> copy( factory->get_structure_decl( "Test1"));
    MI_CHECK( copy);
    MI_CHECK_NOT_EQUAL( decl.get(), copy.get());
    MI_CHECK_EQUAL( nullptr, decl->get_structure_type_name());
    MI_CHECK_EQUAL_CSTR( copy->get_structure_type_name(), "Test1");

    MI_CHECK_EQUAL( nullptr, factory->get_structure_decl( "Non_existing"));

    mi::base::Handle<const mi::IStructure_decl> manifest_field( factory->get_structure_decl( "Manifest_field"));
    MI_CHECK( manifest_field);
    mi::base::Handle<const mi::IStructure_decl> material_data( factory->get_structure_decl( "Material_data"));
    MI_CHECK( material_data);
    mi::base::Handle<const mi::IStructure_decl> uvtile( factory->get_structure_decl( "Uvtile"));
    MI_CHECK( uvtile);
    mi::base::Handle<const mi::IStructure_decl> uvtile_reader( factory->get_structure_decl( "Uvtile_reader"));
    MI_CHECK( uvtile_reader);
    mi::base::Handle<const mi::IStructure_decl> mdle_data( factory->get_structure_decl( "Mdle_data"));
    MI_CHECK( mdle_data);
    mi::base::Handle<const mi::IStructure_decl> mdle_user_file( factory->get_structure_decl( "Mdle_user_file"));
    MI_CHECK( mdle_user_file);

    // test instance creation
    mi::base::Handle<mi::IStructure> structure( transaction->create<mi::IStructure>( "Test1"));
    MI_CHECK_EQUAL( 0, structure->get_length());

    // add another member, test that registered declaration for "Test1" is not modified
    MI_CHECK_EQUAL(  0, decl->add_member( "Sint32", "m_foo"));
    MI_CHECK_EQUAL(  0, extension_api->register_structure_decl( "Test2", decl.get()));
    structure = transaction->create<mi::IStructure>( "Test1");
    MI_CHECK_EQUAL( 0, structure->get_length());
    structure = transaction->create<mi::IStructure>( "Test2");
    MI_CHECK_EQUAL( 1, structure->get_length());

    // add another member, test that instantiation fails if a member has an invalid type name
    MI_CHECK_EQUAL(  0, decl->add_member( "Invalid_type", "m_bar"));
    MI_CHECK_EQUAL(  0, extension_api->register_structure_decl( "Test3", decl.get()));
    structure = transaction->create<mi::IStructure>( "Test3");
    MI_CHECK( !structure);

    // test IExtension_api::unregister_structure_decl()
    MI_CHECK_EQUAL(  0, extension_api->unregister_structure_decl( "Test1"));
    MI_CHECK_EQUAL(  0, extension_api->unregister_structure_decl( "Test2"));
    MI_CHECK_EQUAL(  0, extension_api->unregister_structure_decl( "Test3"));
    MI_CHECK_EQUAL( -1, extension_api->unregister_structure_decl( "Test1"));
    MI_CHECK_EQUAL( -1, extension_api->unregister_structure_decl( "Non_existing"));
    MI_CHECK_EQUAL( -2, extension_api->unregister_structure_decl( 0));
    MI_CHECK_EQUAL( -4, extension_api->unregister_structure_decl( "Invalid_structure_name[]<>{};"));
    MI_CHECK_EQUAL( -6, extension_api->unregister_structure_decl( "Manifest_field"));
    MI_CHECK_EQUAL( -6, extension_api->unregister_structure_decl( "Material_data"));
    MI_CHECK_EQUAL( -6, extension_api->unregister_structure_decl( "Uvtile"));
    MI_CHECK_EQUAL( -6, extension_api->unregister_structure_decl( "Uvtile_reader"));
    MI_CHECK_EQUAL( -6, extension_api->unregister_structure_decl( "Mdle_data"));
    MI_CHECK_EQUAL( -6, extension_api->unregister_structure_decl( "Mdle_user_file"));

    // register the real structure decls
    mi::Size N = sizeof( type_names) / sizeof( const char*);
    for( mi::Size i = 0; i < N; ++i) {

        mi::base::Handle<mi::IStructure_decl> decl(
            transaction->create<mi::IStructure_decl>( "Structure_decl"));
        MI_CHECK( decl);

        mi::Size n = decl_lengths[i];
        for( mi::Size j = 0; j < n; ++j) {
            MI_CHECK_EQUAL( 0, decl->add_member( type_names[i][j], member_names[i][j]));
        }

        MI_CHECK_EQUAL( 0, extension_api->register_structure_decl( decl_names[i], decl.get()));
    }

    // instantiate the real structure decls
    for( mi::Size i = 0; i < N; ++i) {

        mi::base::Handle<mi::IStructure> structure(
            transaction->create<mi::IStructure>( decl_names[i]));
        MI_CHECK( structure);

        mi::base::Handle<const mi::IStructure_decl> decl( structure->get_structure_decl());
        MI_CHECK( decl);

        mi::Size n = decl->get_length();
        for( mi::Size j = 0; j < n; ++j) {

            MI_CHECK_EQUAL_CSTR( decl->get_member_type_name( j), type_names[i][j]);
            MI_CHECK_EQUAL_CSTR( decl->get_member_name( j),  member_names[i][j]);
            MI_CHECK_EQUAL_CSTR( decl->get_member_type_name( member_names[i][j]), type_names[i][j]);
            MI_CHECK_EQUAL_CSTR( decl->get_member_name( j), structure->get_key( j));
        }
    }

    // test IExtension_api::register_structure_decl() (continued)

    // reject self-recursive declarations
    mi::base::Handle<mi::IStructure_decl> rec1(
        transaction->create<mi::IStructure_decl>( "Structure_decl"));
    MI_CHECK_EQUAL(  0, rec1->add_member( "Rec1", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec1", rec1.get()));

    // register a declaration as "Rec2" with a member of a yet-undefined type "Ref3"
    mi::base::Handle<mi::IStructure_decl> rec2(
        transaction->create<mi::IStructure_decl>( "Structure_decl"));
    MI_CHECK_EQUAL(  0, rec2->add_member( "Rec3", "m_foo"));
    MI_CHECK_EQUAL(  0, extension_api->register_structure_decl( "Rec2", rec2.get()));

    mi::base::Handle<mi::IStructure_decl> rec3(
        transaction->create<mi::IStructure_decl>( "Structure_decl"));

    // reject indirect recursive declarations
    MI_CHECK_EQUAL(  0, rec3->add_member( "Rec2", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec3", rec3.get()));
    MI_CHECK_EQUAL(  0, rec3->remove_member( "m_foo"));

    // reject indirect recursive declarations (via static array)
    MI_CHECK_EQUAL(  0, rec3->add_member( "Rec2[42]", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec3", rec3.get()));
    MI_CHECK_EQUAL(  0, rec3->remove_member( "m_foo"));

    // reject indirect recursive declarations (via dynamic array)
    MI_CHECK_EQUAL(  0, rec3->add_member( "Rec2[]", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec3", rec3.get()));
    MI_CHECK_EQUAL(  0, rec3->remove_member( "m_foo"));

    // reject indirect recursive declarations (via maps)
    MI_CHECK_EQUAL(  0, rec3->add_member( "Map<Rec2>", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec3", rec3.get()));
    MI_CHECK_EQUAL(  0, rec3->remove_member( "m_foo"));

    // reject indirect recursive declarations (via pointers)
    MI_CHECK_EQUAL(  0, rec3->add_member( "Pointer<Rec2>", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec3", rec3.get()));
    MI_CHECK_EQUAL(  0, rec3->remove_member( "m_foo"));

    // reject indirect recursive declarations (via const pointers)
    MI_CHECK_EQUAL(  0, rec3->add_member( "Const_pointer<Rec2>", "m_foo"));
    MI_CHECK_EQUAL( -5, extension_api->register_structure_decl( "Rec3", rec3.get()));
    MI_CHECK_EQUAL(  0, rec3->remove_member( "m_foo"));
}


void test_attribute_simple( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element;
    mi::base::Handle<mi::IStructure> structure;
    mi::base::Handle<const mi::IStructure_decl> decl;

    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Simple_attr
    structure = db_element->create_attribute<mi::IStructure>( "simple", "Simple_attr");
    decl = structure->get_structure_decl();
    MI_CHECK( decl);
    set_simple_structure( transaction, structure.get());
    decl.reset();
    structure.reset();

    // serialize and re-access the DB element
    db_element.reset();
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Simple_attr (continued)
    structure = db_element->edit_attribute<mi::IStructure>( "simple");
    MI_CHECK( db_element->is_attribute( "simple"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Simple_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "simple"), "Simple_attr");
    check_simple_structure( transaction, structure.get());
    structure.reset();

    MI_CHECK( db_element->destroy_attribute( "simple"));
}

void test_attribute_compound( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element;
    mi::base::Handle<mi::IStructure> structure;
    mi::base::Handle<const mi::IStructure_decl> decl;

    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Compound_attr
    structure = db_element->create_attribute<mi::IStructure>( "compound", "Compound_attr");
    decl = structure->get_structure_decl();
    MI_CHECK( decl);
    set_collection_structure( transaction, structure.get());
    decl.reset();
    structure.reset();

    // serialize and re-access the DB element
    db_element.reset();
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Compound_attr (continued)
    structure = db_element->edit_attribute<mi::IStructure>( "compound");
    MI_CHECK( db_element->is_attribute( "compound"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Compound_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "compound"), "Compound_attr");
    check_collection_structure( transaction, structure.get());
    structure.reset();

    MI_CHECK( db_element->destroy_attribute( "compound"));
}

void test_attribute_collection( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element;
    mi::base::Handle<mi::IStructure> structure;
    mi::base::Handle<mi::IDynamic_array> darray;
    mi::base::Handle<const mi::IStructure_decl> decl;

    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Collection_attr
    structure = db_element->create_attribute<mi::IStructure>( "collection", "Collection_attr");
    darray = structure->get_value<mi::IDynamic_array>( "m_sint32_0");
    darray->set_length( 7);
    darray.reset();
    decl = structure->get_structure_decl();
    MI_CHECK( decl);
    set_collection_structure( transaction, structure.get());
    decl.reset();
    structure.reset();

    // serialize and re-access the DB element
    db_element.reset();
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Collection_attr (continued)
    structure = db_element->edit_attribute<mi::IStructure>( "collection");
    MI_CHECK( db_element->is_attribute( "collection"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Collection_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "collection"), "Collection_attr");
    darray = structure->get_value<mi::IDynamic_array>( "m_sint32_0");
    MI_CHECK_EQUAL( 7, darray->get_length());
    darray.reset();
    check_collection_structure( transaction, structure.get());
    structure.reset();

    MI_CHECK( db_element->destroy_attribute( "collection"));
}

void test_attribute_nested( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element;
    mi::base::Handle<mi::IStructure> structure;
    mi::base::Handle<mi::IArray> array;
    mi::base::Handle<mi::IDynamic_array> darray;
    mi::base::Handle<mi::ISint32> sint32;
    mi::base::Handle<const mi::IStructure_decl> decl;

    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Nested_attr
    structure = db_element->create_attribute<mi::IStructure>( "nested", "Nested_attr");
    decl = structure->get_structure_decl();
    MI_CHECK( decl);
    set_nested_structure( transaction, structure.get());
    decl.reset();
    structure.reset();

    // serialize and re-access the DB element
    db_element.reset();
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Nested_attr (continued)
    structure = db_element->edit_attribute<mi::IStructure>( "nested");
    check_nested_structure( transaction, structure.get());
    MI_CHECK( db_element->is_attribute( "nested"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Nested_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "nested"), "Nested_attr");
    structure.reset();

    // test direct access to parts of an attribute (m_simple_array)
    array = db_element->edit_attribute<mi::IArray>( "nested.m_simple_array");
    MI_CHECK( array);
    MI_CHECK( db_element->is_attribute( "nested.m_simple_array"));
    MI_CHECK_EQUAL_CSTR( array->get_type_name(), "Simple_attr[3]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "nested.m_simple_array"), "Simple_attr[3]");
    structure = db_element->edit_attribute<mi::IStructure>( "nested.m_simple_array[2]");
    MI_CHECK( structure);
    MI_CHECK( db_element->is_attribute( "nested.m_simple_array[2]"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Simple_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "nested.m_simple_array[2]"), "Simple_attr");
    sint32 = db_element->edit_attribute<mi::ISint32>( "nested.m_simple_array[2].m_sint32");
    MI_CHECK( sint32);
    MI_CHECK( db_element->is_attribute( "nested.m_simple_array[2].m_sint32"));
    MI_CHECK_EQUAL_CSTR( sint32->get_type_name(), "Sint32");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "nested.m_simple_array[2].m_sint32"), "Sint32");
    sint32.reset();
    structure.reset();
    array.reset();

    // test direct access to parts of an attribute (m_simple_darray)
    darray = db_element->edit_attribute<mi::IDynamic_array>( "nested.m_simple_darray");
    MI_CHECK( darray);
    MI_CHECK( db_element->is_attribute( "nested.m_simple_darray"));
    MI_CHECK_EQUAL_CSTR( darray->get_type_name(), "Simple_attr[]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "nested.m_simple_darray"), "Simple_attr[]");
    MI_CHECK_EQUAL( 0, darray->get_length());
    darray.reset();

    MI_CHECK( db_element->destroy_attribute( "nested"));
}

void test_attribute_nested_array( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element;
    mi::base::Handle<mi::IStructure> structure;
    mi::base::Handle<mi::IArray> array;
    mi::base::Handle<mi::IDynamic_array> darray;
    mi::base::Handle<mi::ISint32> sint32;
    mi::base::Handle<const mi::IStructure_decl> decl;

    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Nested_attr[3] (array of arrays via a struct)
    array = db_element->create_attribute<mi::IArray>( "array", "Nested_attr[3]");
    mi::Size n = array->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        structure = array->get_element<mi::IStructure>( i);
        decl = structure->get_structure_decl();
        MI_CHECK( decl);
        set_nested_structure( transaction, structure.get(), i);
    }
    array.reset();

    // serialize and re-access the DB element
    db_element.reset();
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Nested_attr[3] (continued)
    array = db_element->edit_attribute<mi::IArray>( "array");
    MI_CHECK( db_element->is_attribute( "array"));
    MI_CHECK_EQUAL_CSTR( array->get_type_name(), "Nested_attr[3]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "array"), "Nested_attr[3]");
    n = array->get_length();
    MI_CHECK_EQUAL( 3, n);
    for( mi::Size i = 0; i < n; ++i) {
        structure = array->get_element<mi::IStructure>( i);
        check_nested_structure( transaction, structure.get(), i);
    }
    array.reset();

    // test direct access to parts of an attribute
    structure = db_element->edit_attribute<mi::IStructure>( "array[1]");
    MI_CHECK( structure);
    MI_CHECK( db_element->is_attribute( "array[1]"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Nested_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "array[1]"), "Nested_attr");
    array = db_element->edit_attribute<mi::IArray>( "array[1].m_simple_array");
    MI_CHECK( array);
    MI_CHECK( db_element->is_attribute( "array[1].m_simple_array"));
    MI_CHECK_EQUAL_CSTR( array->get_type_name(), "Simple_attr[3]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "array[1].m_simple_array"), "Simple_attr[3]");
    structure = db_element->edit_attribute<mi::IStructure>( "array[1].m_simple_array[2]");
    MI_CHECK( structure);
    MI_CHECK( db_element->is_attribute( "array[1].m_simple_array[2]"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Simple_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "array[1].m_simple_array[2]"), "Simple_attr");
    sint32 = db_element->edit_attribute<mi::ISint32>( "array[1].m_simple_array[2].m_sint32");
    MI_CHECK( sint32);
    MI_CHECK( db_element->is_attribute( "array[1].m_simple_array[2].m_sint32"));
    MI_CHECK_EQUAL_CSTR( sint32->get_type_name(), "Sint32");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "array[1].m_simple_array[2].m_sint32"), "Sint32");
    sint32.reset();
    array.reset();
    structure.reset();

    // test direct access to parts of an attribute (m_simple_darray)
    darray = db_element->edit_attribute<mi::IDynamic_array>( "array[1].m_simple_darray");
    MI_CHECK( darray);
    MI_CHECK( db_element->is_attribute( "array[1].m_simple_darray"));
    MI_CHECK_EQUAL_CSTR( darray->get_type_name(), "Simple_attr[]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "array[1].m_simple_darray"), "Simple_attr[]");
    MI_CHECK_EQUAL( 0, darray->get_length());
    darray.reset();

    MI_CHECK( db_element->destroy_attribute( "array"));
}

void test_attribute_nested_darray( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element;
    mi::base::Handle<mi::IStructure> structure;
    mi::base::Handle<mi::IArray> array;
    mi::base::Handle<mi::IDynamic_array> darray;
    mi::base::Handle<mi::ISint32> sint32;
    mi::base::Handle<const mi::IStructure_decl> decl;

    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Nested_attr[] (array of arrays via a struct)
    darray = db_element->create_attribute<mi::IDynamic_array>( "darray", "Nested_attr[]");
    darray->set_length( 3);
    mi::Size n = darray->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        structure = darray->get_element<mi::IStructure>( i);
        decl = structure->get_structure_decl();
        MI_CHECK( decl);
        set_nested_structure( transaction, structure.get(), i);
    }
    darray.reset();

    // serialize and re-access the DB element
    db_element.reset();
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "attribute_container");

    // test attribute of type Nested_attr[3] (continued)
    darray = db_element->edit_attribute<mi::IDynamic_array>( "darray");
    MI_CHECK( db_element->is_attribute( "darray"));
    MI_CHECK_EQUAL_CSTR( darray->get_type_name(), "Nested_attr[]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "darray"), "Nested_attr[]");
    n = darray->get_length();
    MI_CHECK_EQUAL( 3, n);
    for( mi::Size i = 0; i < n; ++i) {
        structure = darray->get_element<mi::IStructure>( i);
        check_nested_structure( transaction, structure.get(), i);
    }
    darray.reset();

    // test direct access to parts of an attribute
    structure = db_element->edit_attribute<mi::IStructure>( "darray[1]");
    MI_CHECK( structure);
    MI_CHECK( db_element->is_attribute( "darray[1]"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Nested_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "darray[1]"), "Nested_attr");
    array = db_element->edit_attribute<mi::IArray>( "darray[1].m_simple_array");
    MI_CHECK( array);
    MI_CHECK( db_element->is_attribute( "darray[1].m_simple_array"));
    MI_CHECK_EQUAL_CSTR( array->get_type_name(), "Simple_attr[3]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "darray[1].m_simple_array"), "Simple_attr[3]");
    structure = db_element->edit_attribute<mi::IStructure>( "darray[1].m_simple_array[2]");
    MI_CHECK( structure);
    MI_CHECK( db_element->is_attribute( "darray[1].m_simple_array[2]"));
    MI_CHECK_EQUAL_CSTR( structure->get_type_name(), "Simple_attr");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "darray[1].m_simple_array[2]"), "Simple_attr");
    sint32 = db_element->edit_attribute<mi::ISint32>( "darray[1].m_simple_array[2].m_sint32");
    MI_CHECK( sint32);
    MI_CHECK( db_element->is_attribute( "darray[1].m_simple_array[2].m_sint32"));
    MI_CHECK_EQUAL_CSTR( sint32->get_type_name(), "Sint32");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "darray[1].m_simple_array[2].m_sint32"), "Sint32");
    sint32.reset();
    array.reset();
    structure.reset();

    // test direct access to parts of an attribute (m_simple_darray)
    darray = db_element->edit_attribute<mi::IDynamic_array>( "darray[1].m_simple_darray");
    MI_CHECK( darray);
    MI_CHECK( db_element->is_attribute( "darray[1].m_simple_darray"));
    MI_CHECK_EQUAL_CSTR( darray->get_type_name(), "Simple_attr[]");
    MI_CHECK_EQUAL_CSTR( db_element->get_attribute_type_name( "darray[1].m_simple_darray"), "Simple_attr[]");
    MI_CHECK_EQUAL( 0, darray->get_length());
    darray.reset();

    MI_CHECK( db_element->destroy_attribute( "darray"));
}

void test_attribute( mi::neuraylib::ITransaction* transaction)
{
    // Dummy objects to test IRef.
    mi::base::Handle<mi::neuraylib::IAttribute_container> options0(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK_EQUAL( 0, transaction->store( options0.get(), "options0"));
    options0.reset();
    mi::base::Handle<mi::neuraylib::IAttribute_container> options1(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK_EQUAL( 0, transaction->store( options1.get(), "options1"));
    options1.reset();

    mi::base::Handle<mi::neuraylib::IAttribute_container> attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( attribute_container);
    MI_CHECK_EQUAL( 0, transaction->store( attribute_container.get(), "attribute_container"));

    test_attribute_simple( transaction);
    test_attribute_compound( transaction);
    test_attribute_collection( transaction);
    test_attribute_nested( transaction);
    test_attribute_nested_array( transaction);
    test_attribute_nested_darray( transaction);
}



void unregister_decls( mi::neuraylib::IExtension_api* extension_api)
{
    mi::Size N = sizeof( type_names) / sizeof( const char*);
    for( mi::Size i = 0; i < N; ++i)
        MI_CHECK_EQUAL( 0, extension_api->unregister_structure_decl( decl_names[i]));

    MI_CHECK_EQUAL(  0, extension_api->unregister_structure_decl( "Rec2"));
    MI_CHECK_EQUAL( 0, extension_api->unregister_enum_decl( "Test_enum"));
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK( factory);

        mi::base::Handle<mi::neuraylib::IExtension_api> extension_api(
            neuray->get_api_component<mi::neuraylib::IExtension_api>());
        MI_CHECK( extension_api);

        mi::base::Handle<mi::IEnum_decl> decl( transaction->create<mi::IEnum_decl>( "Enum_decl"));
        decl->add_enumerator( "ZERO", 0);
        decl->add_enumerator( "ONE", 1);
        MI_CHECK_EQUAL( 0, extension_api->register_enum_decl( "Test_enum", decl.get()));

        // run tests
        test_decl( extension_api.get(), factory.get(), transaction.get());

        // run attribute tests
        test_attribute( transaction.get());


        MI_CHECK_EQUAL( 0, transaction->commit());

        unregister_decls( extension_api.get());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_types_structure )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));


        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray.reset();
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

