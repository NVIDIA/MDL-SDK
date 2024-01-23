/***************************************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/scene/mdl_elements"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include "mdl_elements_value.h"
#include "mdl_elements_type.h"
#include "i_mdl_elements_expression.h"
#include "test_shared.h"

#include <boost/algorithm/string/replace.hpp>

#include <mi/neuraylib/istring.h>

#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serial_buffer_serializer.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>

using namespace MI;
using namespace MDL;

template<class T>
T* serialize_and_deserialize( const Value_factory& vf, const T* value)
{
    SERIAL::Buffer_serializer serializer;
    mi::base::Handle<T> cloned_value( vf.clone( value));
    vf.serialize( &serializer, cloned_value.get());
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    mi::base::Handle<T> deserialized_value( vf.deserialize<T>( &deserializer));
    T* cloned_and_deserialized_value = vf.clone( deserialized_value.get());
    return cloned_and_deserialized_value;
}

IValue_list* serialize_and_deserialize_list( const Value_factory& vf, const IValue_list* value_list)
{
    SERIAL::Buffer_serializer serializer;
    mi::base::Handle<IValue_list> cloned_value_list( vf.clone( value_list));
    vf.serialize_list( &serializer, cloned_value_list.get());
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    mi::base::Handle<IValue_list> deserialized_value_list( vf.deserialize_list( &deserializer));
    IValue_list* cloned_deserialized_value_list = vf.clone( deserialized_value_list.get());
    return cloned_deserialized_value_list;
};

void check_dump(
    DB::Transaction* transaction, Value_factory& vf, const IValue* value, const char* expected)
{
    mi::base::Handle<const mi::IString> dump( vf.dump( transaction, value, "foo"));
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), expected);

    mi::base::Handle<IValue_list> list( vf.create_value_list( 0));
    dump = vf.dump( transaction, list.get(), "bar");
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), "value_list bar = [ ]");

    MI_CHECK_EQUAL( 0, list->add_value( "foo", value));
    dump = vf.dump( transaction, list.get(), "bar");
    std::string list_expected = "value_list bar = [\n    ";
    list_expected += boost::replace_all_copy(
                         boost::replace_all_copy( std::string( expected), "\n", "\n    "),
                         "foo", "0: foo");
    list_expected += ";\n]";
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), list_expected.c_str());
}

void check_list( const Value_factory& vf, IValue* value)
{
    mi::base::Handle<IValue_list> vl( vf.create_value_list( 1));
    MI_CHECK_EQUAL( vl->add_value( "foo", value), 0);
    vl = serialize_and_deserialize_list( vf, vl.get());
    mi::base::Handle<const IValue> value2( vl->get_value( "foo"));
    MI_CHECK( value2);
    MI_CHECK_EQUAL( vl->get_size(), 1);
    MI_CHECK_EQUAL_CSTR( vl->get_name( 0), "foo");
    MI_CHECK( !vl->get_name( 1));
    MI_CHECK_EQUAL( vl->get_index( "foo"), 0);
    MI_CHECK_EQUAL( vl->get_index( "bar"), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( vl->add_value( "foo", value), -2);
    MI_CHECK_EQUAL( vl->set_value( "foo", value), 0);
}

void test_bool()
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_bool> bv;

    bv = vf.create_bool( true);
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK_EQUAL( bv->get_value(), true);
    bv->set_value( false);
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK_EQUAL( bv->get_value(), false);

    bv = vf.create<IValue_bool>( b.get());
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK_EQUAL( bv->get_value(), false);

    check_dump( /*transaction*/ nullptr, vf, bv.get(), "bool foo = false");
    check_list( vf, bv.get());
}

void test_int()
{
    Type_factory tf;
    mi::base::Handle<const IType_int> i( tf.create_int());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_int> iv;

    iv = vf.create_int( 42);
    iv = serialize_and_deserialize( vf, iv.get());
    MI_CHECK_EQUAL( iv->get_value(), 42);
    iv->set_value( 43);
    iv = serialize_and_deserialize( vf, iv.get());
    MI_CHECK_EQUAL( iv->get_value(), 43);

    iv = vf.create<IValue_int>( i.get());
    iv = serialize_and_deserialize( vf, iv.get());
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv->set_value( 44);
    iv = serialize_and_deserialize( vf, iv.get());
    MI_CHECK_EQUAL( iv->get_value(), 44);

    check_dump( /*transaction*/ nullptr, vf, iv.get(), "int foo = 44");
    check_list( vf, iv.get());
}

void test_enum()
{
    Type_factory tf;
    IType_enum::Values values;
    values.push_back( std::make_pair( "one", 1));
    values.push_back( std::make_pair( "two", 2));
    mi::base::Handle<const IAnnotation_block> annotations;
    IType_enum::Value_annotations value_annotations;
    mi::Sint32 errors;
    mi::base::Handle<const IType_enum> e( tf.create_enum(
        "::my_enum", IType_enum::EID_USER, values, annotations, value_annotations, &errors));
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( e);

    Value_factory vf( &tf);
    mi::base::Handle<IValue_enum> ev;

    ev = vf.create_enum( e.get(), 1);
    ev = serialize_and_deserialize( vf, ev.get());
    MI_CHECK_EQUAL( ev->get_index(), 1);
    MI_CHECK_EQUAL( ev->get_value(), 2);
    MI_CHECK_EQUAL_CSTR( ev->get_name(), "two");
    ev->set_index( 0);
    ev = serialize_and_deserialize( vf, ev.get());
    MI_CHECK_EQUAL( ev->get_index(), 0);
    MI_CHECK_EQUAL( ev->get_value(), 1);
    MI_CHECK_EQUAL_CSTR( ev->get_name(), "one");
    ev->set_value( 2);
    ev = serialize_and_deserialize( vf, ev.get());
    MI_CHECK_EQUAL( ev->get_index(), 1);
    MI_CHECK_EQUAL( ev->get_value(), 2);
    MI_CHECK_EQUAL_CSTR( ev->get_name(), "two");
    ev->set_name( "one");
    ev = serialize_and_deserialize( vf, ev.get());
    MI_CHECK_EQUAL( ev->get_index(), 0);
    MI_CHECK_EQUAL( ev->get_value(), 1);
    MI_CHECK_EQUAL_CSTR( ev->get_name(), "one");

    ev = vf.create<IValue_enum>( e.get());
    ev = serialize_and_deserialize( vf, ev.get());
    MI_CHECK_EQUAL( ev->get_index(), 0);
    MI_CHECK_EQUAL( ev->get_value(), 1);
    MI_CHECK_EQUAL_CSTR( ev->get_name(), "one");
    ev->set_value( 1);
    ev = serialize_and_deserialize( vf, ev.get());
    MI_CHECK_EQUAL( ev->get_index(), 0);
    MI_CHECK_EQUAL( ev->get_value(), 1);
    MI_CHECK_EQUAL_CSTR( ev->get_name(), "one");

    check_dump( /*transaction*/ nullptr, vf, ev.get(), "enum \"::my_enum\" foo = one(1)");
    check_list( vf, ev.get());
}

void test_float()
{
    Type_factory tf;
    mi::base::Handle<const IType_float> f( tf.create_float());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_float> fv;

    fv = vf.create_float( 42.0f);
    fv = serialize_and_deserialize( vf, fv.get());
    MI_CHECK_EQUAL( fv->get_value(), 42.0f);
    fv->set_value( 43.0f);
    fv = serialize_and_deserialize( vf, fv.get());
    MI_CHECK_EQUAL( fv->get_value(), 43.0f);

    fv = vf.create<IValue_float>( f.get());
    fv = serialize_and_deserialize( vf, fv.get());
    MI_CHECK_EQUAL( fv->get_value(), 0.0f);
    fv->set_value( 44.0f);
    fv = serialize_and_deserialize( vf, fv.get());
    MI_CHECK_EQUAL( fv->get_value(), 44.0f);

    check_dump( /*transaction*/ nullptr, vf, fv.get(), "float foo = 44");
    check_list( vf, fv.get());
}

void test_double()
{
    Type_factory tf;
    mi::base::Handle<const IType_double> d( tf.create_double());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_double> dv;

    dv = vf.create_double( 42.0);
    dv = serialize_and_deserialize( vf, dv.get());
    MI_CHECK_EQUAL( dv->get_value(), 42.0);
    dv->set_value( 43.0);
    dv = serialize_and_deserialize( vf, dv.get());
    MI_CHECK_EQUAL( dv->get_value(), 43.0);

    dv = vf.create<IValue_double>( d.get());
    dv = serialize_and_deserialize( vf, dv.get());
    MI_CHECK_EQUAL( dv->get_value(), 0.0);
    dv->set_value( 44.0);
    dv = serialize_and_deserialize( vf, dv.get());
    MI_CHECK_EQUAL( dv->get_value(), 44.0);

    check_dump( /*transaction*/ nullptr, vf, dv.get(), "double foo = 44");
    check_list( vf, dv.get());
}

void test_string()
{
    Type_factory tf;
    mi::base::Handle<const IType_string> s( tf.create_string());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_string> sv;

    sv = vf.create_string( "foo");
    sv = serialize_and_deserialize( vf, sv.get());
    MI_CHECK_EQUAL_CSTR( sv->get_value(), "foo");
    sv->set_value( "bar");
    sv = serialize_and_deserialize( vf, sv.get());
    MI_CHECK_EQUAL_CSTR( sv->get_value(), "bar");

    sv = vf.create<IValue_string>( s.get());
    sv = serialize_and_deserialize( vf, sv.get());
    MI_CHECK_EQUAL_CSTR( sv->get_value(), "");
    sv->set_value( "bar");
    sv = serialize_and_deserialize( vf, sv.get());
    MI_CHECK_EQUAL_CSTR( sv->get_value(), "bar");

    check_dump( /*transaction*/ nullptr, vf, sv.get(), "string foo = \"bar\"");
    check_list( vf, sv.get());
}

void test_vector()
{
    Type_factory tf;
    mi::base::Handle<const IType_int> i( tf.create_int());
    mi::base::Handle<const IType_vector> v( tf.create_vector( i.get(), 2));

    Value_factory vf( &tf);
    mi::base::Handle<IValue_vector> vv;
    mi::base::Handle<IValue_int> iv, iv2;

    vv = vf.create_vector( v.get());
    vv = serialize_and_deserialize( vf, vv.get());
    MI_CHECK_EQUAL( vv->get_size(), 2);
    iv = vv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv = vv->get_value<IValue_int>( 1);
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv = vv->get_value<IValue_int>( 2);
    MI_CHECK( !iv);

    vv = vf.create<IValue_vector>( v.get());
    vv = serialize_and_deserialize( vf, vv.get());
    MI_CHECK_EQUAL( vv->get_size(), 2);
    iv = vv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 0);

    iv2 = vf.create_int( 42);
    MI_CHECK_EQUAL( vv->set_value( 0, iv2.get()), 0);
    vv = serialize_and_deserialize( vf, vv.get());
    iv = vv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 42);
    MI_CHECK_EQUAL( vv->set_value( 0, nullptr), -1);
    MI_CHECK_EQUAL( vv->set_value( 2, iv2.get()), -2);

    check_dump( /*transaction*/ nullptr, vf, vv.get(), "int2 foo = (42, 0)");
    check_list( vf, vv.get());
}

void test_matrix()
{
    Type_factory tf;
    mi::base::Handle<const IType_float> f( tf.create_float());
    mi::base::Handle<const IType_vector> v( tf.create_vector( f.get(), 3));
    mi::base::Handle<const IType_matrix> m( tf.create_matrix( v.get(), 4));

    Value_factory vf( &tf);
    mi::base::Handle<IValue_matrix> mv;
    mi::base::Handle<IValue_vector> vv, vv2;
    mi::base::Handle<IValue_float> fv, fv2;

    mv = vf.create_matrix( m.get());
    mv = serialize_and_deserialize( vf, mv.get());
    MI_CHECK_EQUAL( mv->get_size(), 4);
    vv = mv->get_value( 0);
    fv = vv->get_value<IValue_float>( 0);
    MI_CHECK_EQUAL( fv->get_value(), 0);
    fv = vv->get_value<IValue_float>( 3);
    MI_CHECK( !fv);
    vv = mv->get_value( 4);
    MI_CHECK( !vv);

    mv = vf.create<IValue_matrix>( m.get());
    mv = serialize_and_deserialize( vf, mv.get());
    MI_CHECK_EQUAL( mv->get_size(), 4);
    vv = mv->get_value( 0);
    fv = vv->get_value<IValue_float>( 0);
    MI_CHECK_EQUAL( fv->get_value(), 0);

    fv2 = vf.create_float( 42);
    vv2 = vf.create_vector( v.get());
    MI_CHECK_EQUAL( vv2->set_value( 0, fv2.get()), 0);
    MI_CHECK_EQUAL( mv->set_value( 0, vv2.get()), 0);
    mv = serialize_and_deserialize( vf, mv.get());
    vv = mv->get_value( 0);
    fv = vv->get_value<IValue_float>( 0);
    MI_CHECK_EQUAL( fv->get_value(), 42);
    MI_CHECK_EQUAL( mv->set_value( 0, nullptr), -1);
    MI_CHECK_EQUAL( mv->set_value( 4, vv2.get()), -2);

    check_dump(
        /*transaction*/ nullptr, vf, mv.get(),
        "float4x3 foo = (42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)");
    check_list( vf, mv.get());
}

void test_color()
{
    Type_factory tf;
    mi::base::Handle<const IType_color> c( tf.create_color());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_color> cv;
    mi::base::Handle<IValue_float> fv, fv2;

    cv = vf.create_color( 1.0f, 2.0f, 3.0f);
    cv = serialize_and_deserialize( vf, cv.get());
    MI_CHECK_EQUAL( cv->get_size(), 3);
    fv = cv->get_value( 0);
    MI_CHECK_EQUAL( fv->get_value(), 1.0f);
    fv = cv->get_value( 1);
    MI_CHECK_EQUAL( fv->get_value(), 2.0f);
    fv = cv->get_value( 2);
    MI_CHECK_EQUAL( fv->get_value(), 3.0f);
    fv = cv->get_value( 3);
    MI_CHECK( !fv);

    cv = vf.create<IValue_color>( c.get());
    cv = serialize_and_deserialize( vf, cv.get());
    MI_CHECK_EQUAL( cv->get_size(), 3);
    fv = cv->get_value( 0);
    MI_CHECK_EQUAL( fv->get_value(), 0.0f);
    fv = cv->get_value( 1);
    MI_CHECK_EQUAL( fv->get_value(), 0.0f);
    fv = cv->get_value( 2);
    MI_CHECK_EQUAL( fv->get_value(), 0.0f);
    fv = cv->get_value( 3);
    MI_CHECK( !fv);

    fv2 = vf.create_float( 4.0f);
    MI_CHECK_EQUAL( cv->set_value( 0, fv2.get()), 0);
    cv = serialize_and_deserialize( vf, cv.get());
    fv = cv->get_value( 0);
    MI_CHECK_EQUAL( fv->get_value(), 4.0f);
    MI_CHECK_EQUAL( cv->set_value( 0, static_cast<IValue*>( nullptr)), -1);
    MI_CHECK_EQUAL( cv->set_value( 0, static_cast<IValue_float*>( nullptr)), -1);
    MI_CHECK_EQUAL( cv->set_value( 3, fv2.get()), -2);

    check_dump( /*transaction*/ nullptr, vf, cv.get(), "color foo = (4, 0, 0)");
    check_list( vf, cv.get());
}

void test_immediate_sized_array()
{
    Type_factory tf;
    mi::base::Handle<const IType_int> i( tf.create_int());
    mi::base::Handle<const IType_array> ar(
        tf.create_immediate_sized_array( i.get(), 2));

    Value_factory vf( &tf);
    mi::base::Handle<IValue_array> arv;
    mi::base::Handle<IValue_int> iv, iv2;

    arv = vf.create_array( ar.get());
    arv = serialize_and_deserialize( vf, arv.get());
    MI_CHECK_EQUAL( arv->get_size(), 2);
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv->set_value( 43);
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 43);
    iv2 = vf.create_int( 44);
    arv->set_value( 1, iv2.get());
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 0);
    iv = arv->get_value<IValue_int>( 1);
    MI_CHECK_EQUAL( iv->get_value(), 44);
    iv = arv->get_value<IValue_int>( 2);
    MI_CHECK( !iv);

    arv = vf.create<IValue_array>( ar.get());
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( arv->get_size(), 2);
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv->set_value( 43);
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 43);
    iv2 = vf.create_int( 44);
    arv->set_value( 1, iv2.get());
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 1);
    MI_CHECK_EQUAL( iv->get_value(), 44);
    iv = arv->get_value<IValue_int>( 2);
    MI_CHECK( !iv);

    check_dump( /*transaction*/ nullptr, vf, arv.get(),
        "int[2] foo = [\n"
        "    int 0 = 43;\n"
        "    int 1 = 44;\n"
        "]");
    check_list( vf, arv.get());
}

void test_deferred_sized_array()
{
    Type_factory tf;
    mi::base::Handle<const IType_int> i( tf.create_int());
    mi::base::Handle<const IType_array> ar(
        tf.create_deferred_sized_array( i.get(), "N"));

    Value_factory vf( &tf);
    mi::base::Handle<IValue_array> arv;
    mi::base::Handle<IValue_int> iv, iv2;

    arv = vf.create_array( ar.get());
    arv = serialize_and_deserialize( vf, arv.get());
    MI_CHECK_EQUAL( arv->get_size(), 0);
    MI_CHECK_EQUAL( 0, arv->set_size( 2));
    arv = serialize_and_deserialize( vf, arv.get());
    MI_CHECK_EQUAL( arv->get_size(), 2);
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv->set_value( 43);
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 43);
    iv2 = vf.create_int( 44);
    arv->set_value( 1, iv2.get());
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 1);
    MI_CHECK_EQUAL( iv->get_value(), 44);
    iv = arv->get_value<IValue_int>( 2);
    MI_CHECK( !iv);

    arv = vf.create<IValue_array>( ar.get());
    arv = serialize_and_deserialize( vf, arv.get());
    MI_CHECK_EQUAL( arv->get_size(), 0);
    MI_CHECK_EQUAL( 0, arv->set_size( 2));
    arv = serialize_and_deserialize( vf, arv.get());
    MI_CHECK_EQUAL( arv->get_size(), 2);
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv->set_value( 43);
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 0);
    MI_CHECK_EQUAL( iv->get_value(), 43);
    iv2 = vf.create_int( 44);
    arv->set_value( 1, iv2.get());
    arv = serialize_and_deserialize( vf, arv.get());
    iv = arv->get_value<IValue_int>( 1);
    MI_CHECK_EQUAL( iv->get_value(), 44);
    iv = arv->get_value<IValue_int>( 2);
    MI_CHECK( !iv);

    check_dump( /*transaction*/ nullptr, vf, arv.get(),
        "int[N] foo = [\n"
        "    int 0 = 43;\n"
        "    int 1 = 44;\n"
        "]");
    check_list( vf, arv.get());

    MI_CHECK_EQUAL( 0, arv->set_size( 0));

    check_dump( /*transaction*/ nullptr, vf, arv.get(),
        "int[N] foo = [ ]");
    check_list( vf, arv.get());
}

void test_struct()
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());
    mi::base::Handle<const IType_int> i( tf.create_int());
    mi::base::Handle<const IType_float> f( tf.create_float());
    mi::base::Handle<const IType_double> d( tf.create_double());
    mi::base::Handle<const IType_string> s( tf.create_string());

    IType_struct::Fields fields;
    fields.push_back( std::make_pair( b, "m_bool"));
    fields.push_back( std::make_pair( i, "m_int"));
    fields.push_back( std::make_pair( f, "m_float"));
    fields.push_back( std::make_pair( d, "m_double"));
    fields.push_back( std::make_pair( s, "m_string"));
    mi::base::Handle<const IAnnotation_block> annotations;
    IType_struct::Field_annotations field_annotations;
    mi::Sint32 errors = 0;
    mi::base::Handle<const IType_struct> st( tf.create_struct(
        "::my_struct",  IType_struct::SID_USER, fields, annotations, field_annotations, &errors));
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st);


    Value_factory vf( &tf);
    mi::base::Handle<IValue_struct> stv;
    mi::base::Handle<IValue_bool> bv;
    mi::base::Handle<IValue_int> iv;
    mi::base::Handle<IValue_float> fv;
    mi::base::Handle<IValue_double> dv;
    mi::base::Handle<IValue_string> sv;
    mi::base::Handle<IValue> vv;

    stv = vf.create_struct( st.get());
    stv = serialize_and_deserialize( vf, stv.get());
    bv = stv->get_field<IValue_bool>( "m_bool");
    MI_CHECK_EQUAL( bv->get_value(), false);
    bv->set_value( true);
    iv = stv->get_field<IValue_int>( "m_int");
    MI_CHECK_EQUAL( iv->get_value(), 0);
    iv->set_value( 42);
    fv = stv->get_field<IValue_float>( "m_float");
    MI_CHECK_EQUAL( fv->get_value(), 0.0f);
    fv->set_value( 43.0f);
    dv = stv->get_field<IValue_double>( "m_double");
    MI_CHECK_EQUAL( dv->get_value(), 0.0);
    dv->set_value( 44.0);
    sv = stv->get_field<IValue_string>( "m_string");
    MI_CHECK_EQUAL_CSTR( sv->get_value(), "");
    sv->set_value( "bar");
    vv = stv->get_field( "non_existing");
    MI_CHECK( !vv);

    stv = serialize_and_deserialize( vf, stv.get());
    bv = stv->get_value<IValue_bool>( 0);
    MI_CHECK_EQUAL( bv->get_value(), true);
    iv = stv->get_value<IValue_int>( 1);
    MI_CHECK_EQUAL( iv->get_value(), 42);
    fv = stv->get_value<IValue_float>( 2);
    MI_CHECK_EQUAL( fv->get_value(), 43.0f);
    dv = stv->get_value<IValue_double>( 3);
    MI_CHECK_EQUAL( dv->get_value(), 44.0);
    sv = stv->get_value<IValue_string>( 4);
    MI_CHECK_EQUAL_CSTR( sv->get_value(), "bar");
    vv = stv->get_value( 5);
    MI_CHECK( !vv);

    check_dump( /*transaction*/ nullptr, vf, stv.get(),
        "struct \"::my_struct\" foo = {\n"
        "    bool m_bool = true;\n"
        "    int m_int = 42;\n"
        "    float m_float = 43;\n"
        "    double m_double = 44;\n"
        "    string m_string = \"bar\";\n"
        "}");
    check_list( vf, stv.get());
}

void test_texture( DB::Transaction* transaction)
{
    TEXTURE::Texture* texture = new TEXTURE::Texture;
    DB::Tag texture_tag = transaction->store( texture, "texture");
    MI_CHECK( texture_tag);

    TEXTURE::Texture* texture2 = new TEXTURE::Texture;
    DB::Tag texture2_tag = transaction->store( texture2, "texture2");
    MI_CHECK( texture2_tag);

    Type_factory tf;
    mi::base::Handle<const IType_texture> t(
        tf.create_texture( IType_texture::TS_2D));

    Value_factory vf( &tf);
    mi::base::Handle<IValue_texture> tv, tv2;

    // valid tag and absolute file path
    tv = vf.create_texture( t.get(), texture_tag);
    tv = serialize_and_deserialize( vf, tv.get());
    MI_CHECK_EQUAL( tv->get_value(), texture_tag);

    // change tag, file path is cleared
    tv->set_value( texture2_tag);
    MI_CHECK_EQUAL( tv->get_value(), texture2_tag);

    // clear tag
    tv->set_value( DB::Tag());
    tv = serialize_and_deserialize( vf, tv.get());
    MI_CHECK( !tv->get_value());

    // default construction
    tv = vf.create<IValue_texture>( t.get());
    tv = serialize_and_deserialize( vf, tv.get());
    MI_CHECK( !tv->get_value());

    // change tag
    tv->set_value( texture_tag);
    tv = serialize_and_deserialize( vf, tv.get());
    MI_CHECK_EQUAL( tv->get_value(), texture_tag);

    // construction with unresolved mdl file path, owner module, gamma and selector
    tv2 = vf.create_texture(t.get(), texture_tag, "./test.png", "::test_module", 2.2f, "R");
    tv2 = serialize_and_deserialize(vf, tv2.get());
    MI_CHECK_EQUAL(tv2->get_value(), texture_tag);
    MI_CHECK_EQUAL_CSTR(tv2->get_unresolved_file_path(), "./test.png");
    MI_CHECK_EQUAL_CSTR(tv2->get_owner_module(), "::test_module");
    MI_CHECK_EQUAL(tv2->get_gamma(), 2.2f);
    MI_CHECK_EQUAL_CSTR(tv2->get_selector(), "R");

    check_dump( transaction, vf, tv.get(), "texture_2d foo = \"texture\"");
    tv->set_value( DB::Tag());
    check_dump( transaction, vf, tv.get(),
        "texture_2d foo = (unset, owner module \"\", unresolved MDL file path \"\")");
    tv = vf.create_texture( t.get(), texture_tag);
    check_dump( transaction, vf, tv.get(), "texture_2d foo = \"texture\"");
    check_list( vf, tv.get());
}

void test_light_profile( DB::Transaction* transaction)
{
    LIGHTPROFILE::Lightprofile* light_profile = new LIGHTPROFILE::Lightprofile;
    DB::Tag light_profile_tag = transaction->store( light_profile, "light_profile");
    MI_CHECK( light_profile_tag);

    LIGHTPROFILE::Lightprofile* light_profile2 = new LIGHTPROFILE::Lightprofile;
    DB::Tag light_profile2_tag = transaction->store( light_profile2, "light_profile2");
    MI_CHECK( light_profile2_tag);

    Type_factory tf;
    mi::base::Handle<const IType_light_profile> l( tf.create_light_profile());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_light_profile> lv, lv2;

    // valid tag and absolute file path
    lv = vf.create_light_profile( light_profile_tag);
    lv = serialize_and_deserialize( vf, lv.get());
    MI_CHECK_EQUAL( lv->get_value(), light_profile_tag);

    // change tag, file path is cleared
    lv->set_value( light_profile2_tag);
    MI_CHECK_EQUAL( lv->get_value(), light_profile2_tag);

    // clear tag
    lv->set_value( DB::Tag());
    lv = serialize_and_deserialize( vf, lv.get());
    MI_CHECK( !lv->get_value());

    // default construction
    lv = vf.create<IValue_light_profile>( l.get());
    lv = serialize_and_deserialize( vf, lv.get());
    MI_CHECK( !lv->get_value());

    // change tag
    lv->set_value( light_profile_tag);
    lv = serialize_and_deserialize( vf, lv.get());
    MI_CHECK_EQUAL( lv->get_value(), light_profile_tag);

    // construction with unresolved mdl file path and owner module
    lv2 = vf.create_light_profile(light_profile_tag, "./test.ies", "::test_module");
    lv2 = serialize_and_deserialize(vf, lv2.get());
    MI_CHECK_EQUAL(lv2->get_value(), light_profile_tag);
    MI_CHECK_EQUAL_CSTR(lv2->get_unresolved_file_path(), "./test.ies");
    MI_CHECK_EQUAL_CSTR(lv2->get_owner_module(), "::test_module");

    check_dump( transaction, vf, lv.get(), "light_profile foo = \"light_profile\"");
    lv->set_value( DB::Tag());
    check_dump( transaction, vf, lv.get(),
        "light_profile foo = (unset, owner module \"\", unresolved MDL file path \"\")");
    lv = vf.create_light_profile( light_profile_tag);
    check_dump( transaction, vf, lv.get(), "light_profile foo = \"light_profile\"");
    check_list( vf, lv.get());
}

void test_bsdf_measurement( DB::Transaction* transaction)
{
    BSDFM::Bsdf_measurement* bsdf_measurement = new BSDFM::Bsdf_measurement;
    DB::Tag bsdf_measurement_tag = transaction->store( bsdf_measurement, "bsdf_measurement");
    MI_CHECK( bsdf_measurement_tag);

    BSDFM::Bsdf_measurement* bsdf_measurement2 = new BSDFM::Bsdf_measurement;
    DB::Tag bsdf_measurement2_tag = transaction->store( bsdf_measurement2, "bsdf_measurement2");
    MI_CHECK( bsdf_measurement2_tag);

    Type_factory tf;
    mi::base::Handle<const IType_bsdf_measurement> b( tf.create_bsdf_measurement());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_bsdf_measurement> bv, bv2;

    // valid tag and absolute file path
    bv = vf.create_bsdf_measurement( bsdf_measurement_tag);
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK_EQUAL( bv->get_value(), bsdf_measurement_tag);

    // change tag, file path is cleared
    bv->set_value( bsdf_measurement2_tag);
    MI_CHECK_EQUAL( bv->get_value(), bsdf_measurement2_tag);

    // clear tag
    bv->set_value( DB::Tag());
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK( !bv->get_value());

    // default construction
    bv = vf.create<IValue_bsdf_measurement>( b.get());
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK( !bv->get_value());

    // change tag
    bv->set_value( bsdf_measurement_tag);
    bv = serialize_and_deserialize( vf, bv.get());
    MI_CHECK_EQUAL( bv->get_value(), bsdf_measurement_tag);

    // construction with unresolved mdl file path and owner module
    bv2 = vf.create_bsdf_measurement(bsdf_measurement_tag, "./test.mbsdf", "::test_module");
    bv2 = serialize_and_deserialize(vf, bv2.get());
    MI_CHECK_EQUAL(bv2->get_value(), bsdf_measurement_tag);
    MI_CHECK_EQUAL_CSTR(bv2->get_unresolved_file_path(), "./test.mbsdf");
    MI_CHECK_EQUAL_CSTR(bv2->get_owner_module(), "::test_module");

    check_dump( transaction, vf, bv.get(), "bsdf_measurement foo = \"bsdf_measurement\"");
    bv->set_value( DB::Tag());
    check_dump( transaction, vf, bv.get(),
        "bsdf_measurement foo = (unset, owner module \"\", unresolved MDL file path \"\")");
    bv = vf.create_bsdf_measurement( bsdf_measurement_tag);
    check_dump(
        transaction, vf, bv.get(), "bsdf_measurement foo = \"bsdf_measurement\"");
    check_list( vf, bv.get());
}

void test_invalid_df()
{
    Type_factory tf;
    mi::base::Handle<const IType_bsdf> bsdf( tf.create_bsdf());
    mi::base::Handle<const IType_edf> edf( tf.create_edf());
    mi::base::Handle<const IType_vdf> vdf( tf.create_vdf());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_invalid_df> rv;

    rv = vf.create_invalid_df( bsdf.get());
    rv = serialize_and_deserialize( vf, rv.get());
    MI_CHECK( rv);
    rv = vf.create<IValue_invalid_df>( bsdf.get());
    rv = serialize_and_deserialize( vf, rv.get());
    MI_CHECK( rv);

    check_dump( /*transaction*/ nullptr, vf, rv.get(), "bsdf foo = (invalid reference)");

    rv = vf.create_invalid_df( edf.get());
    rv = serialize_and_deserialize( vf, rv.get());
    MI_CHECK( rv);
    rv = vf.create<IValue_invalid_df>( edf.get());
    rv = serialize_and_deserialize( vf, rv.get());
    MI_CHECK( rv);

    check_dump( /*transaction*/ nullptr, vf, rv.get(), "edf foo = (invalid reference)");

    rv = vf.create_invalid_df( vdf.get());
    rv = serialize_and_deserialize( vf, rv.get());
    MI_CHECK( rv);
    rv = vf.create<IValue_invalid_df>( vdf.get());
    rv = serialize_and_deserialize( vf, rv.get());
    MI_CHECK( rv);

    check_dump( /*transaction*/ nullptr, vf, rv.get(), "vdf foo = (invalid reference)");
    check_list( vf, rv.get());

    // deprecated

    mi::base::Handle<const IType_texture> t(
        tf.create_texture( IType_texture::TS_2D));
    mi::base::Handle<const IType_light_profile> l( tf.create_light_profile());
    mi::base::Handle<const IType_bsdf_measurement> b( tf.create_bsdf_measurement());

    rv = vf.create_invalid_df( t.get());
    MI_CHECK( !rv);
    rv = vf.create_invalid_df( l.get());
    MI_CHECK( !rv);
    rv = vf.create_invalid_df( b.get());
    MI_CHECK( !rv);
}

void test_compare( DB::Transaction* transaction)
{
    Type_factory tf;
    Value_factory vf( &tf);
    IValue_factory* ivf = &vf; // use only the general overload

    // compare ints
    mi::base::Handle<IValue_int> iv1, iv2, iv3;
    iv1 = vf.create_int( 0);
    iv2 = vf.create_int( 1);
    iv3 = vf.create_int( 1);
    MI_CHECK_EQUAL( ivf->compare( nullptr, iv1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( iv1.get(), nullptr), +1);
    MI_CHECK_EQUAL( ivf->compare( iv1.get(), iv2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( iv2.get(), iv3.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( iv3.get(), iv1.get()), +1);

    // compare floats
    mi::base::Handle<IValue_float> fv1, fv2, fv3;
    fv1 = vf.create_float( 0.0f);
    fv2 = vf.create_float( 1.0f);
    fv3 = vf.create_float( 1.0f);
    MI_CHECK_EQUAL( ivf->compare( nullptr, fv1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( fv1.get(), nullptr), +1);
    MI_CHECK_EQUAL( ivf->compare( fv1.get(), fv2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( fv2.get(), fv3.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( fv3.get(), fv1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( fv1.get(), fv2.get(),  0.9), -1);
    MI_CHECK_EQUAL( ivf->compare( fv1.get(), fv2.get(),  1.0),  0);
    MI_CHECK_EQUAL( ivf->compare( fv2.get(), fv3.get(), -0.5),  0);
    MI_CHECK_EQUAL( ivf->compare( fv2.get(), fv3.get(),  0.0),  0);
    MI_CHECK_EQUAL( ivf->compare( fv2.get(), fv3.get(),  1.0),  0);
    MI_CHECK_EQUAL( ivf->compare( fv3.get(), fv1.get(),  0.9), +1);
    MI_CHECK_EQUAL( ivf->compare( fv3.get(), fv1.get(),  1.0),  0);

    // compare doubles
    mi::base::Handle<IValue_double> dv1, dv2, dv3;
    dv1 = vf.create_double( 0.0);
    dv2 = vf.create_double( 1.0);
    dv3 = vf.create_double( 1.0);
    MI_CHECK_EQUAL( ivf->compare( nullptr, dv1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( dv1.get(), nullptr), +1);
    MI_CHECK_EQUAL( ivf->compare( dv1.get(), dv2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( dv2.get(), dv3.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( dv3.get(), dv1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( dv1.get(), dv2.get(),  0.9), -1);
    MI_CHECK_EQUAL( ivf->compare( dv1.get(), dv2.get(),  1.0),  0);
    MI_CHECK_EQUAL( ivf->compare( dv2.get(), dv3.get(), -0.5),  0);
    MI_CHECK_EQUAL( ivf->compare( dv2.get(), dv3.get(),  0.0),  0);
    MI_CHECK_EQUAL( ivf->compare( dv2.get(), dv3.get(),  1.0),  0);
    MI_CHECK_EQUAL( ivf->compare( dv3.get(), dv1.get(),  0.9), +1);
    MI_CHECK_EQUAL( ivf->compare( dv3.get(), dv1.get(),  1.0),  0);

    // compare strings
    mi::base::Handle<IValue_string> sv1, sv2, sv3, sv4;
    sv1 = vf.create_string( nullptr);
    sv2 = vf.create_string( "");
    sv3 = vf.create_string( "foo");
    sv4 = vf.create_string( "bar");
    MI_CHECK_EQUAL( ivf->compare( nullptr, sv1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( sv1.get(), nullptr), +1);
    MI_CHECK_EQUAL( ivf->compare( sv1.get(), sv2.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( sv2.get(), sv3.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( sv3.get(), sv4.get()), +1);

    // compare colors (compounds in general)
    mi::base::Handle<IValue_color> cv1, cv2, cv3, cv4;
    cv1 = vf.create_color( 0.0f, 2.0f, 3.0f);
    cv2 = vf.create_color( 1.0f, 2.0f, 3.0f);
    cv3 = vf.create_color( 1.0f, 2.0f, 3.0f);
    cv4 = vf.create_color( 1.0f, 2.0f, 4.0f);
    MI_CHECK_EQUAL( ivf->compare( cv1.get(), cv2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( cv2.get(), cv3.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( cv3.get(), cv4.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( cv4.get(), cv1.get()), +1);

    // compare deferred-sized arrays
    mi::base::Handle<const IType_int> i( tf.create_int());
    mi::base::Handle<const IType_array> isa1(
        tf.create_immediate_sized_array( i.get(), 1));
    mi::base::Handle<const IType_array> isa2(
        tf.create_immediate_sized_array( i.get(), 2));
    mi::base::Handle<const IType_array> dsa(
        tf.create_deferred_sized_array( i.get(), "N"));
    mi::base::Handle<IValue_array> isav1, isav2, isav3, dsav1, dsav2, dsav3;
    isav1 = vf.create_array( isa1.get());
    isav2 = vf.create_array( isa2.get());
    isav3 = vf.create_array( isa2.get());
    dsav1 = vf.create_array( dsa.get());
    dsav2 = vf.create_array( dsa.get());
    dsav3 = vf.create_array( dsa.get());
    MI_CHECK_EQUAL( 0, dsav1->set_size( 1));
    MI_CHECK_EQUAL( 0, dsav2->set_size( 2));
    MI_CHECK_EQUAL( 0, dsav3->set_size( 2));
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), isav2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( isav2.get(), isav3.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( isav3.get(), isav1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), dsav2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( dsav2.get(), dsav3.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( dsav3.get(), dsav1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), dsav1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), isav1.get()), +1);

    // compare textures (resources in general)
    TEXTURE::Texture* texture1 = new TEXTURE::Texture;
    DB::Tag texture_tag1 = transaction->store( texture1, "texture1");
    MI_CHECK( texture_tag1);
    TEXTURE::Texture* texture2 = new TEXTURE::Texture;
    DB::Tag texture_tag2 = transaction->store( texture2, "texture2");
    MI_CHECK( texture_tag2);

    mi::base::Handle<const IType_texture> t(
        tf.create_texture( IType_texture::TS_2D));
    mi::base::Handle<IValue> tv0, tv1, tv2;
    tv0 = vf.create_texture( t.get(), DB::Tag());
    tv1 = vf.create_texture( t.get(), texture_tag1);
    tv2 = vf.create_texture( t.get(), texture_tag2);
    MI_CHECK_EQUAL( ivf->compare( tv0.get(), tv0.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( tv0.get(), tv1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( tv1.get(), tv1.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( tv1.get(), tv2.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( tv2.get(), tv0.get()), +1);

    // compare different kinds
    MI_CHECK_EQUAL( ivf->compare( iv1.get()  , iv1.get())  ,  0);
    MI_CHECK_EQUAL( ivf->compare( iv1.get()  , cv1.get())  , -1);
    MI_CHECK_EQUAL( ivf->compare( iv1.get()  , isav1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( iv1.get()  , dsav1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( iv1.get()  , tv1.get())  , -1);
    MI_CHECK_EQUAL( ivf->compare( cv1.get()  , iv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( cv1.get()  , cv1.get())  ,  0);
    MI_CHECK_EQUAL( ivf->compare( cv1.get()  , isav1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( cv1.get()  , dsav1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( cv1.get()  , tv1.get())  , -1);
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), iv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), cv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), isav1.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), dsav1.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( isav1.get(), tv1.get())  , -1);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), iv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), cv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), isav1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), dsav1.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( dsav1.get(), tv1.get())  , -1);
    MI_CHECK_EQUAL( ivf->compare( tv1.get()  , iv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( tv1.get()  , cv1.get())  , +1);
    MI_CHECK_EQUAL( ivf->compare( tv1.get()  , isav1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( tv1.get()  , dsav1.get()), +1);
    MI_CHECK_EQUAL( ivf->compare( tv1.get()  , tv1.get())  ,  0);
}

void test_compare_list()
{
    Type_factory tf;
    Value_factory vf( &tf);
    IValue_factory* ivf = &vf; // use only the general overload

    mi::base::Handle<IValue_list> m1( vf.create_value_list( 1));
    mi::base::Handle<IValue_list> m2( vf.create_value_list( 1));
    mi::base::Handle<IValue_list> m3( vf.create_value_list( 1));

    mi::base::Handle<IValue_int> iv1( vf.create_int( 0));
    mi::base::Handle<IValue_int> iv2( vf.create_int( 1));

    // compare list sizes
    MI_CHECK_EQUAL( m3->add_value( "foo", iv1.get()), 0);
    MI_CHECK_EQUAL( ivf->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( m3.get(), m1.get()), +1);

    // compare element names
    MI_CHECK_EQUAL( m1->add_value( "bar", iv1.get()), 0);
    MI_CHECK_EQUAL( m2->add_value( "bar", iv1.get()), 0);
    MI_CHECK_EQUAL( ivf->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( m3.get(), m1.get()), +1);

    // compare element values
    m3 = vf.create_value_list( 1);
    MI_CHECK_EQUAL( m3->add_value( "bar", iv2.get()), 0);
    MI_CHECK_EQUAL( ivf->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( ivf->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( ivf->compare( m3.get(), m1.get()), +1);
}

MI_TEST_AUTO_FUNCTION( test )
{
    Unified_database_access db_access;

    DB::Database* database = db_access.get_database();
    DB::Scope* scope = database->get_global_scope();
    DB::Transaction* transaction = scope->start_transaction();

    test_bool();
    test_int();
    test_enum();
    test_float();
    test_double();
    test_string();

    test_vector();
    test_matrix();
    test_color();

    test_immediate_sized_array();
    test_deferred_sized_array();
    test_struct();

    test_texture( transaction);
    test_light_profile( transaction);
    test_bsdf_measurement( transaction);

    test_invalid_df();

    test_compare( transaction);
    test_compare_list();

    transaction->commit();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
