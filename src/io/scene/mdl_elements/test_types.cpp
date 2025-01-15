/***************************************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_elements_type.h"
#include "i_mdl_elements_expression.h"

#include <boost/algorithm/string/replace.hpp>

#include <mi/neuraylib/istring.h>

#include <base/system/main/access_module.h>
#include <base/system/main/access_module.h>
#include <base/hal/thread/i_thread_thread.h>
#include <base/hal/thread/i_thread_condition.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/data/serial/i_serial_buffer_serializer.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

using namespace MI;
using namespace MDL;

#define MI_CHECK_EQUAL_TYPES( lhs, rhs) MI_CHECK_EQUAL( tf.compare( lhs, rhs), 0)

const IStruct_category* serialize_and_deserialize_struct_category(
    Type_factory& tf, const IStruct_category* struct_category)
{
    SERIAL::Buffer_serializer serializer;
    tf.serialize_struct_category( &serializer, struct_category);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return tf.deserialize_struct_category( &deserializer);
}

IStruct_category_list* serialize_and_deserialize_list(
    Type_factory& tf, const IStruct_category_list* list)
{
    SERIAL::Buffer_serializer serializer;
    mi::base::Handle<IStruct_category_list> cloned_list( tf.clone( list));
    tf.serialize_struct_category_list( &serializer, cloned_list.get());
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    mi::base::Handle<IStruct_category_list> deserialized_list(
        tf.deserialize_struct_category_list( &deserializer));
    IStruct_category_list* cloned_deserialized_list = tf.clone( deserialized_list.get());
    return cloned_deserialized_list;
}

void check_dump( Type_factory& tf, const IStruct_category* sc, const char* expected)
{
    mi::base::Handle<const mi::IString> dump( tf.dump( sc, 0));
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), expected);

    mi::base::Handle<IStruct_category_list> list( tf.create_struct_category_list( 0));
    dump = tf.dump( list.get(), 0);
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), "struct_category_list [ ]");

    MI_CHECK_EQUAL( 0, list->add_struct_category( "bar", sc));
    dump = tf.dump( list.get(), 0);
    std::string list_expected = "struct_category_list [\n    0: bar = ";
    list_expected += boost::replace_all_copy( std::string( expected), "\n", "\n    ");
    list_expected += "\n]";
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), list_expected.c_str());
}

void check_list( Type_factory& tf, const IStruct_category* sc)
{
    mi::base::Handle<IStruct_category_list> scl( tf.create_struct_category_list( 1));
    scl->add_struct_category( "foo", sc);
    scl = serialize_and_deserialize_list( tf, scl.get());
    mi::base::Handle<const IStruct_category> struct_category2( scl->get_struct_category( "foo"));
    MI_CHECK_EQUAL_TYPES( struct_category2.get(), sc);
    MI_CHECK_EQUAL( scl->get_size(), 1);
    MI_CHECK_EQUAL_CSTR( scl->get_name( 0), "foo");
    MI_CHECK( !scl->get_name( 1));
    MI_CHECK_EQUAL( scl->get_index( "foo"), 0);
    MI_CHECK_EQUAL( scl->get_index( "bar"), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( scl->add_struct_category( "foo", sc), -2);
    MI_CHECK_EQUAL( scl->set_struct_category( "foo", sc), 0);
}

template<class T>
const T* serialize_and_deserialize( Type_factory& tf, const T* type)
{
    SERIAL::Buffer_serializer serializer;
    tf.serialize( &serializer, type);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return tf.deserialize<T>( &deserializer);
}

IType_list* serialize_and_deserialize_list( Type_factory& tf, const IType_list* list)
{
    SERIAL::Buffer_serializer serializer;
    mi::base::Handle<IType_list> cloned_list( tf.clone( list));
    tf.serialize_list( &serializer, cloned_list.get());
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    mi::base::Handle<IType_list> deserialized_list( tf.deserialize_list( &deserializer));
    IType_list* cloned_deserialized_list = tf.clone( deserialized_list.get());
    return cloned_deserialized_list;
}

template<class T>
void serialize_and_deserialize_name( Type_factory& tf, const T* type)
{
    std::string s = tf.get_mdl_type_name( type);
    mi::base::Handle<const IType> type2( tf.create_from_mdl_type_name( s.c_str()));
    MI_CHECK( type2);
    MI_CHECK_EQUAL_TYPES( type, type2.get());
}

void check_dump( Type_factory& tf, const IType* type, const char* expected)
{
    mi::base::Handle<const mi::IString> dump( tf.dump( type, 0));
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), expected);

    mi::base::Handle<IType_list> list( tf.create_type_list( 0));
    dump = tf.dump( list.get(), 0);
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), "type_list [ ]");

    MI_CHECK_EQUAL( 0, list->add_type( "bar", type));
    dump = tf.dump( list.get(), 0);
    std::string list_expected = "type_list [\n    0: bar = ";
    list_expected += boost::replace_all_copy( std::string( expected), "\n", "\n    ");
    list_expected += "\n]";
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), list_expected.c_str());
}

void check_list( Type_factory& tf, const IType* type)
{
    mi::base::Handle<IType_list> tl( tf.create_type_list( 1));
    tl->add_type( "foo", type);
    tl = serialize_and_deserialize_list( tf, tl.get());
    mi::base::Handle<const IType> type2( tl->get_type( "foo"));
    MI_CHECK_EQUAL_TYPES( type2.get(), type);
    MI_CHECK_EQUAL( tl->get_size(), 1);
    MI_CHECK_EQUAL_CSTR( tl->get_name( 0), "foo");
    MI_CHECK( !tl->get_name( 1));
    MI_CHECK_EQUAL( tl->get_index( "foo"), 0);
    MI_CHECK_EQUAL( tl->get_index( "bar"), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( tl->add_type( "foo", type), -2);
    MI_CHECK_EQUAL( tl->set_type( "foo", type), 0);
}

void test_type_factory()
{
    Type_factory tf;
    mi::Sint32 errors = 0;
    mi::base::Handle<const IType> element;
    mi::base::Handle<const IType> base;

    // check bool type
    mi::base::Handle<const IType_bool> b( tf.create_bool());
    b = serialize_and_deserialize( tf, b.get());
    MI_CHECK_EQUAL( b->get_kind(), IType::TK_BOOL);
    MI_CHECK_EQUAL( b->get_all_type_modifiers(), 0);
    base = b->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), b.get());
    check_dump( tf, b.get(), "bool");
    check_list( tf, b.get());
    serialize_and_deserialize_name( tf, b.get());

    // check alias type (uniform, name)
    mi::base::Handle<const IType_alias> a1(
        tf.create_alias( b.get(), IType::MK_UNIFORM, "::foo::uniform_alias"));
    a1 = serialize_and_deserialize( tf, a1.get());
    MI_CHECK_EQUAL( a1->get_kind(), IType::TK_ALIAS);
    MI_CHECK_EQUAL( a1->get_all_type_modifiers(), IType::MK_UNIFORM);
    MI_CHECK_EQUAL_CSTR( a1->get_symbol(), "::foo::uniform_alias");
    base = a1->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), b.get());
    check_dump( tf, a1.get(), "alias \"::foo::uniform_alias\" uniform\n"
        "    bool");
    check_list( tf, a1.get());
    // TODO support named aliases
    std::string str = tf.get_mdl_type_name( a1.get());
    MI_CHECK_EQUAL( str, "uniform bool");
    mi::base::Handle<const IType_alias> a1_recreated(
        static_cast<MDL::IType_factory*>( &tf)->create_from_mdl_type_name<IType_alias>(
            "uniform bool"));
    MI_CHECK( a1_recreated);
    // MI_CHECK_EQUAL_TYPES( a1_recreated.get(), a1.get());
    // or shorter:
    // serialize_and_deserialize_name( tf, a1.get());

    // check alias type (varying, no name)
    mi::base::Handle<const IType_alias> a2(
        tf.create_alias( a1.get(), IType::MK_VARYING, nullptr));
    a2 = serialize_and_deserialize( tf, a2.get());
    MI_CHECK_EQUAL( a2->get_kind(), IType::TK_ALIAS);
    MI_CHECK_EQUAL( a2->get_all_type_modifiers(),
        IType::MK_UNIFORM | IType::MK_VARYING);
    MI_CHECK_EQUAL( a2->get_symbol(), nullptr);
    base = a2->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), b.get());
    check_dump( tf, a2.get(), "alias varying\n"
        "    alias \"::foo::uniform_alias\" uniform\n"
        "        bool");
    check_list( tf, a2.get());
    // TODO support named aliases
    str = tf.get_mdl_type_name( a2.get());
    MI_CHECK_EQUAL( str, "varying uniform bool");
    mi::base::Handle<const IType_alias> a2_recreated(
        static_cast<MDL::IType_factory*>( &tf)->create_from_mdl_type_name<IType_alias>(
            "varying uniform bool"));
    MI_CHECK( a2_recreated);
    // MI_CHECK_EQUAL_TYPES( a2_recreated.get(), a2.get());
    // or shorter:
    // serialize_and_deserialize_name( tf, a2.get());

    // check int type
    mi::base::Handle<const IType_int> i( tf.create_int());
    i = serialize_and_deserialize( tf, i.get());
    MI_CHECK_EQUAL( i->get_kind(), IType::TK_INT);
    MI_CHECK_EQUAL( i->get_all_type_modifiers(), 0);
    base = i->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), i.get());
    check_dump( tf, i.get(), "int");
    check_list( tf, i.get());
    serialize_and_deserialize_name( tf, i.get());

    // register enum type
    IType_enum::Values values;
    values.push_back( std::make_pair( "one", 1));
    values.push_back( std::make_pair( "two", 2));
    mi::base::Handle<const IAnnotation_block> annotations;
    IType_enum::Value_annotations value_annotations;
    mi::base::Handle<const IType_enum> e;

    e = tf.create_enum(
        "::my_enum", IType_enum::EID_USER, values, annotations, value_annotations, &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( e);
    // keep the enum alive
    mi::base::Handle<const IType_enum> e_backup = e;

    e = tf.create_enum(
        "::my_enum", IType_enum::EID_USER, values, annotations, value_annotations, &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( e);
    serialize_and_deserialize_name( tf, e.get());

    e = tf.create_enum(
        nullptr, IType_enum::EID_USER, values, annotations, value_annotations, &errors);
    MI_CHECK_EQUAL( -1, errors);
    MI_CHECK( !e);

    e = tf.create_enum(
        "my_enum", IType_enum::EID_USER, values, annotations, value_annotations, &errors);
    MI_CHECK_EQUAL( -2, errors);
    MI_CHECK( !e);
    values.push_back( std::make_pair(
        "three", 3));

    e = tf.create_enum(
        "::my_enum", IType_enum::EID_USER, values, annotations, value_annotations, &errors);
    MI_CHECK_EQUAL( -4, errors);
    MI_CHECK( !e);

    e = e_backup;
    e_backup.reset();

    // check enum type
    e = tf.create_enum( "::my_enum");
    e = serialize_and_deserialize( tf, e.get());
    MI_CHECK_EQUAL_CSTR( e->get_symbol(), "::my_enum");
    MI_CHECK_EQUAL( e->get_size(), 2);
    MI_CHECK_EQUAL_CSTR( e->get_value_name( 0), "one");
    MI_CHECK_EQUAL_CSTR( e->get_value_name( 1), "two");
    MI_CHECK_EQUAL( e->get_value_name( 2), nullptr);
    MI_CHECK_EQUAL( e->get_value_code( 0, &errors), 1);
    MI_CHECK_EQUAL( errors, 0);
    MI_CHECK_EQUAL( e->get_value_code( 1, &errors), 2);
    MI_CHECK_EQUAL( errors, 0);
    MI_CHECK_EQUAL( e->get_value_code( 2, &errors), 0);
    MI_CHECK_EQUAL( errors, -1);
    MI_CHECK_EQUAL( e->find_value( "one"), 0);
    MI_CHECK_EQUAL( e->find_value( "two"), 1);
    MI_CHECK_EQUAL( e->find_value( 0), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( e->find_value( "foo"), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( e->find_value( 1), 0);
    MI_CHECK_EQUAL( e->find_value( 2), 1);
    MI_CHECK_EQUAL( e->find_value( 0), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( e->find_value( 3), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( e->get_predefined_id(), IType_enum::EID_USER);
    check_dump( tf, e.get(), "enum \"::my_enum\" {\n"
        "    one = 1,\n"
        "    two = 2,\n"
        "}");
    check_list( tf, e.get());

    // check float type
    mi::base::Handle<const IType_float> f( tf.create_float());
    f = serialize_and_deserialize( tf, f.get());
    MI_CHECK_EQUAL( f->get_kind(), IType::TK_FLOAT);
    MI_CHECK_EQUAL( f->get_all_type_modifiers(), 0);
    base = f->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), f.get());
    check_dump( tf, f.get(), "float");
    check_list( tf, f.get());
    serialize_and_deserialize_name( tf, f.get());

    // check double type
    mi::base::Handle<const IType_double> d( tf.create_double());
    d = serialize_and_deserialize( tf, d.get());
    MI_CHECK_EQUAL( d->get_kind(), IType::TK_DOUBLE);
    MI_CHECK_EQUAL( d->get_all_type_modifiers(), 0);
    base = d->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), d.get());
    check_dump( tf, d.get(), "double");
    check_list( tf, d.get());
    serialize_and_deserialize_name( tf, d.get());

    // check string type
    mi::base::Handle<const IType_string> s( tf.create_string());
    s = serialize_and_deserialize( tf, s.get());
    MI_CHECK_EQUAL( s->get_kind(), IType::TK_STRING);
    MI_CHECK_EQUAL( s->get_all_type_modifiers(), 0);
    base = s->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), s.get());
    check_dump( tf, s.get(), "string");
    check_list( tf, s.get());
    serialize_and_deserialize_name( tf, s.get());

    // check vector type
    mi::base::Handle<const IType_vector> v( tf.create_vector( f.get(), 3));
    v = serialize_and_deserialize( tf, v.get());
    MI_CHECK_EQUAL( v->get_kind(), IType::TK_VECTOR);
    MI_CHECK_EQUAL( v->get_all_type_modifiers(), 0);
    base = v->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), v.get());

    MI_CHECK_EQUAL( v->get_size(), 3);
    MI_CHECK_EQUAL( v->get_component_type( 3), nullptr);
    element = v->get_component_type( 0);
    MI_CHECK_EQUAL_TYPES( element.get(), f.get());

    MI_CHECK_EQUAL( v->get_size(), 3);
    element = v->get_element_type();
    MI_CHECK_EQUAL_TYPES( element.get(), f.get());

    MI_CHECK( !tf.create_vector( e.get(), 3));
    MI_CHECK( !tf.create_vector( s.get(), 3));

    check_dump( tf, v.get(), "float3");
    check_list( tf, v.get());
    serialize_and_deserialize_name( tf, v.get());

    // check matrix type
    mi::base::Handle<const IType_matrix> m( tf.create_matrix( v.get(), 4));
    m = serialize_and_deserialize( tf, m.get());
    MI_CHECK_EQUAL( m->get_kind(), IType::TK_MATRIX);
    MI_CHECK_EQUAL( m->get_all_type_modifiers(), 0);
    base = m->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), m.get());

    MI_CHECK_EQUAL( m->get_size(), 4);
    MI_CHECK_EQUAL( m->get_component_type( 4), nullptr);
    element = m->get_component_type( 0);
    MI_CHECK_EQUAL_TYPES( element.get(), v.get());

    element = m->get_element_type();
    MI_CHECK_EQUAL_TYPES( element.get(), v.get());

    check_dump( tf, m.get(), "float4x3");

    mi::base::Handle<const IType_vector> iv( tf.create_vector( i.get(), 3));
    MI_CHECK( !tf.create_matrix( iv.get(), 3));
    check_list( tf, m.get());
    serialize_and_deserialize_name( tf, m.get());

    // check color type
    mi::base::Handle<const IType_color> c( tf.create_color());
    c = serialize_and_deserialize( tf, c.get());
    MI_CHECK_EQUAL( c->get_kind(), IType::TK_COLOR);
    MI_CHECK_EQUAL( c->get_all_type_modifiers(), 0);
    base = c->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), c.get());

    MI_CHECK_EQUAL( c->get_component_type( 3), nullptr);
    element = c->get_component_type( 0);
    MI_CHECK_EQUAL( element->get_kind(), IType::TK_FLOAT);
    MI_CHECK_EQUAL( c->get_size(), 3);

    check_dump( tf, c.get(), "color");
    check_list( tf, c.get());
    serialize_and_deserialize_name( tf, c.get());

    // check immediate-sized array type
    mi::base::Handle<const IType_array> ar(
        tf.create_immediate_sized_array( b.get(), 42));
    ar = serialize_and_deserialize( tf, ar.get());
    MI_CHECK_EQUAL( ar->get_kind(), IType::TK_ARRAY);
    MI_CHECK_EQUAL( ar->get_all_type_modifiers(), 0);
    base = ar->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), ar.get());

    MI_CHECK_EQUAL( ar->get_component_type( 42), nullptr);
    element = ar->get_component_type( 0);
    MI_CHECK_EQUAL( element->get_kind(), IType::TK_BOOL);
    MI_CHECK_EQUAL( ar->get_size(), 42);
    MI_CHECK_EQUAL( ar->get_deferred_size(), nullptr);

    check_dump( tf, ar.get(), "bool[42]");
    check_list( tf, ar.get());
    serialize_and_deserialize_name( tf, ar.get());

    // check deferred-sized array type
    ar = tf.create_deferred_sized_array( b.get(), "N");
    ar = serialize_and_deserialize( tf, ar.get());
    MI_CHECK_EQUAL( ar->get_kind(), IType::TK_ARRAY);
    MI_CHECK_EQUAL( ar->get_all_type_modifiers(), 0);
    base = ar->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), ar.get());

    element = ar->get_component_type( 42);
    MI_CHECK_EQUAL( element->get_kind(), IType::TK_BOOL);
    element = ar->get_component_type( 0);
    MI_CHECK_EQUAL( element->get_kind(), IType::TK_BOOL);
    MI_CHECK_EQUAL( ar->get_size(), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL_CSTR( ar->get_deferred_size(), "N");

    check_dump( tf, ar.get(), "bool[N]");
    check_list( tf, ar.get());
    serialize_and_deserialize_name( tf, ar.get());

    ar = tf.create_deferred_sized_array( b.get(), "123");
    MI_CHECK( !ar);
    ar = tf.create_deferred_sized_array( b.get(), "material");
    MI_CHECK( !ar);
    ar = tf.create_deferred_sized_array( b.get(), "::foo::bar::N");
    MI_CHECK( !ar);

    // register struct category
    mi::base::Handle<const IStruct_category> sc;
    sc = tf.create_struct_category(
        "::my_struct_category",
        IStruct_category::CID_USER,
        annotations,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( sc);
    // keep the struct category alive
    mi::base::Handle<const IStruct_category> sc_backup = sc;

    sc = tf.create_struct_category(
        "::my_struct_category",
        IStruct_category::CID_USER,
        annotations,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( sc);
    sc = serialize_and_deserialize_struct_category( tf, sc.get());

    sc = tf.create_struct_category(
        nullptr,
        IStruct_category::CID_USER,
        annotations,
        &errors);
    MI_CHECK_EQUAL( -1, errors);
    MI_CHECK( !sc);

    sc = tf.create_struct_category(
        "my_struct_category",
        IStruct_category::CID_USER,
        annotations,
        &errors);
    MI_CHECK_EQUAL( -2, errors);
    MI_CHECK( !sc);

    sc = tf.create_struct_category(
        "::my_struct_category",
        IStruct_category::CID_MATERIAL_CATEGORY,
        annotations,
        &errors);
    MI_CHECK_EQUAL( -4, errors);
    MI_CHECK( !sc);

    sc = sc_backup;
    sc_backup.reset();

    // check struct category
    MI_CHECK_EQUAL_CSTR( sc->get_symbol(), "::my_struct_category");
    MI_CHECK_EQUAL( sc->get_predefined_id(), IStruct_category::CID_USER);
    check_dump( tf, sc.get(), "\"::my_struct_category\"");
    check_list( tf, sc.get());

    // create another struct category for later tests
    mi::base::Handle<const IStruct_category> sc2( tf.create_struct_category(
        "::my_struct_category2",
        IStruct_category::CID_USER,
        annotations,
        &errors));

    // register struct type
    IType_struct::Fields fields;
    fields.push_back( std::make_pair( i, "m_int"));
    fields.push_back( std::make_pair( f, "m_float"));
    fields.push_back( std::make_pair( d, "m_double"));
    fields.push_back( std::make_pair( s, "m_string"));
    IType_struct::Field_annotations field_annotations;
    mi::base::Handle<const IType_struct> st;
    mi::base::Handle<const IType_struct> st2;
    mi::base::Handle<const IType_struct> st3;
    mi::base::Handle<const IType_struct> st4;
    mi::base::Handle<const IType_struct> st5;
    mi::base::Handle<const IType_struct> st6;

    st = tf.create_struct(
        "::my_struct",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_cateogry*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st);
    // keep the struct type alive
    mi::base::Handle<const IType_struct> st_backup = st;

    st = tf.create_struct(
        "::my_struct",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_cateogry*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st);
    serialize_and_deserialize_name( tf, st.get());

    st = tf.create_struct(
        nullptr,
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_cateogry*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( -1, errors);
    MI_CHECK( !st);

    st = tf.create_struct(
        "::my_struct",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        sc.get(),
        &errors);
    MI_CHECK_EQUAL( -1, errors);
    MI_CHECK( !st);

    st = tf.create_struct(
        "my_struct",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_cateogry*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( -2, errors);
    MI_CHECK( !st);

    fields.push_back( std::make_pair( s, "m_foo"));
    st = tf.create_struct(
        "::my_struct",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_cateogry*/ nullptr,
        &errors);
    fields.pop_back();
    MI_CHECK_EQUAL( -4, errors);
    MI_CHECK( !st);

    st = st_backup;
    st_backup.reset();

    st2 = tf.create_struct(
        "::my_decl_struct",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ true,
        /*struct_cateogry*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st2);
    serialize_and_deserialize_name( tf, st2.get());

    st3 = tf.create_struct(
        "::my_struct_with_category",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ true,
        sc.get(),
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st3);
    serialize_and_deserialize_name( tf, st3.get());

    st4 = tf.create_struct(
        "::my_struct2_with_category",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ true,
        sc.get(),
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st4);
    serialize_and_deserialize_name( tf, st4.get());

    st5 = tf.create_struct(
        "::my_struct_with_category2",
        IType_struct::SID_USER,
        fields,
        annotations,
        field_annotations,
        /*is_declarative*/ true,
        sc2.get(),
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st5);
    serialize_and_deserialize_name( tf, st5.get());

    st6 = tf.create_struct(
        "::my_empty_struct",
        IType_struct::SID_USER,
        {},
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_category*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( st6);
    serialize_and_deserialize_name( tf, st6.get());

    // check struct type
    st = tf.create_struct( "::my_struct");
    st = serialize_and_deserialize( tf, st.get());
    MI_CHECK_EQUAL_CSTR( st->get_symbol(), "::my_struct");
    MI_CHECK_EQUAL( st->get_size(), 4);
    MI_CHECK_EQUAL_CSTR( st->get_field_name( 0), "m_int");
    MI_CHECK_EQUAL_CSTR( st->get_field_name( 1), "m_float");
    MI_CHECK_EQUAL_CSTR( st->get_field_name( 2), "m_double");
    MI_CHECK_EQUAL_CSTR( st->get_field_name( 3), "m_string");
    MI_CHECK_EQUAL( st->get_field_name( 4), nullptr);
    element = st->get_field_type( 0);
    MI_CHECK_EQUAL( element.get(), i.get());
    element = st->get_field_type( 1);
    MI_CHECK_EQUAL( element.get(), f.get());
    element = st->get_component_type( 2);
    MI_CHECK_EQUAL( element.get(), d.get());
    element = st->get_component_type( 3);
    MI_CHECK_EQUAL( element.get(), s.get());
    MI_CHECK( !st->get_field_type( 4));
    MI_CHECK_EQUAL( st->find_field( "m_int"), 0);
    MI_CHECK_EQUAL( st->find_field( "m_float"), 1);
    MI_CHECK_EQUAL( st->find_field( "m_double"), 2);
    MI_CHECK_EQUAL( st->find_field( "m_string"), 3);
    MI_CHECK_EQUAL( st->find_field( "foo"), mi::Size( -1));
    MI_CHECK_EQUAL( st->get_predefined_id(), IType_struct::SID_USER);
    MI_CHECK_EQUAL( st->is_declarative(), false);
    MI_CHECK( !st->get_struct_category());

    st2 = tf.create_struct( "::my_decl_struct");
    st2 = serialize_and_deserialize( tf, st2.get());
    MI_CHECK_EQUAL( st2->is_declarative(), true);
    MI_CHECK( !st2->get_struct_category());

    st3 = tf.create_struct( "::my_struct_with_category");
    st3 = serialize_and_deserialize( tf, st3.get());
    MI_CHECK_EQUAL( st3->is_declarative(), true);
    mi::base::Handle<const IStruct_category> st3_category( st3->get_struct_category());
    MI_CHECK_EQUAL_CSTR( st3_category->get_symbol(), "::my_struct_category");

    st4 = tf.create_struct( "::my_struct2_with_category");
    st4 = serialize_and_deserialize( tf, st4.get());
    MI_CHECK_EQUAL( st4->is_declarative(), true);
    mi::base::Handle<const IStruct_category> st4_category( st4->get_struct_category());
    MI_CHECK_EQUAL_CSTR( st4_category->get_symbol(), "::my_struct_category");

    st5 = tf.create_struct( "::my_struct_with_category2");
    st5 = serialize_and_deserialize( tf, st5.get());
    MI_CHECK_EQUAL( st5->is_declarative(), true);
    mi::base::Handle<const IStruct_category> st5_category( st5->get_struct_category());
    MI_CHECK_EQUAL_CSTR( st5_category->get_symbol(), "::my_struct_category2");

    st6 = tf.create_struct( "::my_empty_struct");
    st6 = serialize_and_deserialize( tf, st6.get());
    MI_CHECK_EQUAL( st6->is_declarative(), false);
    MI_CHECK( !st6->get_struct_category());

    check_dump( tf, st.get(),
        "struct \"::my_struct\" {\n"
        "    int m_int;\n"
        "    float m_float;\n"
        "    double m_double;\n"
        "    string m_string;\n"
        "}");
    check_dump( tf, st2.get(),
        "declarative struct \"::my_decl_struct\" {\n"
        "    int m_int;\n"
        "    float m_float;\n"
        "    double m_double;\n"
        "    string m_string;\n"
        "}");
    check_dump( tf, st3.get(),
        "declarative struct \"::my_struct_with_category\" in \"::my_struct_category\" {\n"
        "    int m_int;\n"
        "    float m_float;\n"
        "    double m_double;\n"
        "    string m_string;\n"
        "}");
    check_dump( tf, st4.get(),
        "declarative struct \"::my_struct2_with_category\" in \"::my_struct_category\" {\n"
        "    int m_int;\n"
        "    float m_float;\n"
        "    double m_double;\n"
        "    string m_string;\n"
        "}");
    check_dump( tf, st5.get(),
        "declarative struct \"::my_struct_with_category2\" in \"::my_struct_category2\" {\n"
        "    int m_int;\n"
        "    float m_float;\n"
        "    double m_double;\n"
        "    string m_string;\n"
        "}");
    check_dump( tf, st6.get(),
        "struct \"::my_empty_struct\" {\n"
        "}");
    check_list( tf, st.get());

    // check from_same_struct_category()
    MI_CHECK_EQUAL(  0, tf.from_same_struct_category( st3.get(), st4.get()));
    MI_CHECK_EQUAL( +1, tf.from_same_struct_category( st3.get(), st3.get()));
    MI_CHECK_EQUAL( -1, tf.from_same_struct_category( st3.get(), st5.get()));
    MI_CHECK_EQUAL( -1, tf.from_same_struct_category( st3.get(), st2.get()));
    MI_CHECK_EQUAL( -1, tf.from_same_struct_category( st3.get(), f.get()));
    MI_CHECK_EQUAL( -1, tf.from_same_struct_category( st2.get(), st3.get()));
    MI_CHECK_EQUAL( -1, tf.from_same_struct_category( f.get(), st2.get()));

    // check array of struct type
    mi::base::Handle<const IType_array> ar2(
        tf.create_immediate_sized_array( st2.get(), 42));
    ar2 = serialize_and_deserialize( tf, ar2.get());
    MI_CHECK_EQUAL( ar2->is_declarative(), true);
    serialize_and_deserialize_name( tf, ar2.get());

    // register struct of (struct, enum, ...) type
    IType_struct::Fields fields2;
    fields2.push_back( std::make_pair( st, "m_my_struct"));
    fields2.push_back( std::make_pair( e,  "m_my_enum"));
    fields2.push_back( std::make_pair( i,  "m_int"));
    mi::base::Handle<const IType_struct> sta;
    sta = tf.create_struct(
        "::my_struct3",
        IType_struct::SID_USER,
        fields2,
        annotations,
        field_annotations,
        /*is_declarative*/ false,
        /*struct_cateogry*/ nullptr,
        &errors);
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( sta);

    // check struct of (struct, enum, ...) type
    sta = serialize_and_deserialize( tf, sta.get());
    MI_CHECK_EQUAL_CSTR( sta->get_symbol(), "::my_struct3");
    MI_CHECK_EQUAL( sta->get_size(), 3);
    MI_CHECK_EQUAL_CSTR( sta->get_field_name( 0), "m_my_struct");
    MI_CHECK_EQUAL_CSTR( sta->get_field_name( 1), "m_my_enum");
    MI_CHECK_EQUAL_CSTR( sta->get_field_name( 2), "m_int");
    MI_CHECK_EQUAL( sta->get_field_name( 3), nullptr);
    element = sta->get_field_type( 0);
    MI_CHECK_EQUAL( element.get(), st.get());
    element = sta->get_field_type( 1);
    MI_CHECK_EQUAL( element.get(), e.get());
    element = sta->get_component_type( 2);
    MI_CHECK_EQUAL( element.get(), i.get());
    MI_CHECK( !sta->get_field_type( 3));
    MI_CHECK_EQUAL( sta->find_field( "m_my_struct"), 0);
    MI_CHECK_EQUAL( sta->find_field( "m_my_enum"), 1);
    MI_CHECK_EQUAL( sta->find_field( "m_int"), 2);
    MI_CHECK_EQUAL( sta->find_field( "foo"), mi::Size( -1));
    MI_CHECK_EQUAL( sta->get_predefined_id(), IType_struct::SID_USER);

    // check texture type
    mi::base::Handle<const IType_texture> t(
        tf.create_texture( IType_texture::TS_3D));
    t = serialize_and_deserialize( tf, t.get());
    MI_CHECK_EQUAL( t->get_kind(), IType::TK_TEXTURE);
    MI_CHECK_EQUAL( t->get_all_type_modifiers(), 0);
    base = t->skip_all_type_aliases();
    MI_CHECK_EQUAL_TYPES( base.get(), t.get());
    MI_CHECK_EQUAL( IType_texture::TS_3D, t->get_shape());

    check_dump( tf, t.get(), "texture_3d");
    check_list( tf, t.get());
    serialize_and_deserialize_name( tf, t.get());

    // no tests for trivial resources like light profile and BSDF measurement

    // no tests for trivial DFs like BSDF, EDF, VDF
}

void test_compare()
{
    Type_factory tf;
    IType_factory* itf = &tf; // use only the general overload

    mi::base::Handle<const IType_float> f1( itf->create_float());
    mi::base::Handle<const IType_float> f2( itf->create_float());
    mi::base::Handle<const IType_double> d( itf->create_double());

    mi::base::Handle<const IType_alias> a1(
        itf->create_alias( d.get(), IType::MK_UNIFORM, nullptr));
    mi::base::Handle<const IType_alias> a2(
        itf->create_alias( d.get(), IType::MK_VARYING, nullptr));
    mi::base::Handle<const IType_alias> a3(
        itf->create_alias( f1.get(), IType::MK_VARYING, nullptr));
    mi::base::Handle<const IType_alias> a4(
        itf->create_alias( f1.get(), IType::MK_VARYING, "::foo::varying_alias"));
    mi::base::Handle<const IType_alias> a5(
        itf->create_alias( f1.get(), IType::MK_VARYING, "::foo::varying_alias2"));

    mi::base::Handle<const IType_vector> v1( itf->create_vector( d.get(), 2));
    mi::base::Handle<const IType_vector> v2( itf->create_vector( d.get(), 3));
    mi::base::Handle<const IType_vector> v3( itf->create_vector( f1.get(), 3));

    mi::base::Handle<const IType_matrix> m1( itf->create_matrix( v2.get(), 2));
    mi::base::Handle<const IType_matrix> m2( itf->create_matrix( v2.get(), 3));
    mi::base::Handle<const IType_matrix> m3( itf->create_matrix( v3.get(), 3));

    mi::base::Handle<const IType_texture> t1(
        itf->create_texture( IType_texture::TS_2D));
    mi::base::Handle<const IType_texture> t2(
        itf->create_texture( IType_texture::TS_3D));

    MI_CHECK_EQUAL( itf->compare( nullptr, f1.get()), -1);
    MI_CHECK_EQUAL( itf->compare( f1.get(), nullptr), +1);

    MI_CHECK_EQUAL( itf->compare( f1.get(), f1.get()), 0);
    MI_CHECK_EQUAL( itf->compare( f1.get(), d.get()), -1);
    MI_CHECK_EQUAL( itf->compare( d.get(), f1.get()), +1);

    MI_CHECK_EQUAL( itf->compare( a1.get(), a1.get()), 0);
    MI_CHECK_EQUAL( itf->compare( a1.get(), a2.get()), -1);
    MI_CHECK_EQUAL( itf->compare( a2.get(), a1.get()), +1);

    MI_CHECK_EQUAL( itf->compare( a2.get(), a3.get()), +1);
    MI_CHECK_EQUAL( itf->compare( a3.get(), a2.get()), -1);

    MI_CHECK_EQUAL( itf->compare( a3.get(), a4.get()), -1);
    MI_CHECK_EQUAL( itf->compare( a4.get(), a5.get()), -1);
    MI_CHECK_EQUAL( itf->compare( a5.get(), a3.get()), +1);

    MI_CHECK_EQUAL( itf->compare( v1.get(), v1.get()), 0);
    MI_CHECK_EQUAL( itf->compare( v1.get(), v2.get()), -1);
    MI_CHECK_EQUAL( itf->compare( v2.get(), v1.get()), +1);

    MI_CHECK_EQUAL( itf->compare( v2.get(), v3.get()), +1);
    MI_CHECK_EQUAL( itf->compare( v3.get(), v2.get()), -1);

    MI_CHECK_EQUAL( itf->compare( m1.get(), m1.get()), 0);
    MI_CHECK_EQUAL( itf->compare( m1.get(), m2.get()), -1);
    MI_CHECK_EQUAL( itf->compare( m2.get(), m1.get()), +1);

    MI_CHECK_EQUAL( itf->compare( m2.get(), m3.get()), +1);
    MI_CHECK_EQUAL( itf->compare( m3.get(), m2.get()), -1);

    MI_CHECK_EQUAL( itf->compare( t1.get(), t2.get()), -1);
    MI_CHECK_EQUAL( itf->compare( t2.get(), t1.get()), +1);
}

void test_compare_list()
{
    Type_factory tf;
    IType_factory* itf = &tf; // use only the general overload

    mi::base::Handle<IType_list> m1( itf->create_type_list( 1));
    mi::base::Handle<IType_list> m2( itf->create_type_list( 1));
    mi::base::Handle<IType_list> m3( itf->create_type_list( 1));

    mi::base::Handle<const IType> t1( itf->create_int());
    mi::base::Handle<const IType> t2( itf->create_float());

    // compare list sizes
    MI_CHECK_EQUAL( m3->add_type( "foo", t1.get()), 0);
    MI_CHECK_EQUAL( itf->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( itf->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( itf->compare( m3.get(), m1.get()), +1);

    // compare element names
    MI_CHECK_EQUAL( m1->add_type( "bar", t1.get()), 0);
    MI_CHECK_EQUAL( m2->add_type( "bar", t1.get()), 0);
    MI_CHECK_EQUAL( itf->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( itf->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( itf->compare( m3.get(), m1.get()), +1);

    // compare element types
    m3 = itf->create_type_list( 1);
    MI_CHECK_EQUAL( m3->add_type( "bar", t2.get()), 0);
    MI_CHECK_EQUAL( itf->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( itf->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( itf->compare( m3.get(), m1.get()), +1);
}

void test_is_compatible()
{
    Type_factory tf;
    IType_factory* itf = &tf; // use only the general overload

    mi::base::Handle<const IType> b( itf->create_bool());
    mi::base::Handle<const IType> i1( itf->create_int());
    mi::base::Handle<const IType> i2( itf->create_int());
    mi::base::Handle<const IType> f( itf->create_float());

    MI_CHECK_EQUAL( itf->is_compatible( b.get(), i1.get()), -1);
    MI_CHECK_EQUAL( itf->is_compatible( i1.get(), i1.get()), 1);
    MI_CHECK_EQUAL( itf->is_compatible( i1.get(), i2.get()), 1);

    mi::Sint32 errors = 0;
    mi::base::Handle<const IAnnotation_block> annotations;
    mi::base::Handle<const IType_struct> ar_st1, ar_st2;

    {
        IType_struct::Field_annotations field_annotations;

        IType_struct::Fields fields1;
        fields1.push_back( std::make_pair( i1, "m_int1"));
        fields1.push_back( std::make_pair( i2, "m_int2"));
        fields1.push_back( std::make_pair( b,  "m_bool"));

        mi::base::Handle<const IType_struct> st1a( itf->create_struct(
            "::my_struct1",
            IType_struct::SID_USER,
            fields1,
            annotations,
            field_annotations,
            /*is_declarative*/ false,
            /*struct_cateogry*/ nullptr,
            &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( st1a);
        ar_st1 = st1a;

        mi::base::Handle<const IType_struct> st1b( itf->create_struct(
            "::my_struct1",
            IType_struct::SID_USER,
            fields1,
            annotations,
            field_annotations,
            /*is_declarative*/ false,
            /*struct_cateogry*/ nullptr,
            &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( st1b);

        // st1a and st1b are identical types
        MI_CHECK_EQUAL( itf->is_compatible( st1a.get(), st1b.get()), 1);

        IType_struct::Fields fields2;
        fields2.push_back( std::make_pair( i1, "m_other_int1"));
        fields2.push_back( std::make_pair( i2, "m_other_int2"));
        fields2.push_back( std::make_pair( b, "m_other_bool"));
        mi::base::Handle<const IType_struct> st2( itf->create_struct(
            "::my_struct2",
            IType_struct::SID_USER,
            fields2,
            annotations,
            field_annotations,
            /*is_declarative*/ false,
            /*struct_cateogry*/ nullptr,
            &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( st2);
        ar_st2 = st2;

        // st1a and st2 are compatible types (but not identical)
        MI_CHECK_EQUAL( itf->is_compatible( st1a.get(), st2.get()), 0);

        fields2.push_back( std::make_pair( b, "m_yet_another_bool"));
        mi::base::Handle<const IType_struct> st3( itf->create_struct(
            "::my_struct3",
            IType_struct::SID_USER,
            fields2,
            annotations,
            field_annotations,
            /*is_declarative*/ false,
            /*struct_cateogry*/ nullptr,
            &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( st3);

        // st1a and st3 have a different number of fields
        MI_CHECK_EQUAL( itf->is_compatible( st1a.get(), st3.get()), -1);

        // m_yet_another_bool is now actually an int despite its name
        fields2.pop_back();
        fields2.push_back( std::make_pair( i1, "m_yet_another_bool"));
        mi::base::Handle<const IType_struct> st4( itf->create_struct(
            "::my_struct4",
            IType_struct::SID_USER,
            fields2,
            annotations,
            field_annotations,
            /*is_declarative*/ false,
            /*struct_cateogry*/ nullptr,
            &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( st4);

        // st3 and st4 have incompatible fields
        MI_CHECK_EQUAL( itf->is_compatible( st3.get(), st4.get()), -1);

        mi::base::Handle<const IType_struct> st5( itf->create_struct(
            "::my_struct5",
            IType_struct::SID_USER,
            fields2,
            annotations,
            field_annotations,
            /*is_declarative*/ true,
            /*struct_cateogry*/ nullptr,
            &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( st5);

        // st5 is declarative
        MI_CHECK_EQUAL( itf->is_compatible( st1a.get(), st5.get()), -1);
    }
    {
        IType_enum::Value_annotations value_annotations;

        IType_enum::Values values1;
        values1.push_back( std::make_pair( "one", 1));
        values1.push_back( std::make_pair( "two", 2));
        mi::base::Handle<const IType_enum> e1a( itf->create_enum(
            "::my_enum1", IType_enum::EID_USER, values1, annotations, value_annotations, &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( e1a);

        mi::base::Handle<const IType_enum> e1b( itf->create_enum(
            "::my_enum1", IType_enum::EID_USER, values1, annotations, value_annotations, &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( e1b);

        // e1a and e1b are identical types
        MI_CHECK_EQUAL( itf->is_compatible( e1a.get(), e1b.get()), 1);

        IType_enum::Values values2;
        values2.push_back( std::make_pair( "zwei", 2));
        values2.push_back( std::make_pair( "eins", 1));
        mi::base::Handle<const IType_enum> e2( itf->create_enum(
            "::my_enum2", IType_enum::EID_USER, values2, annotations, value_annotations, &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( e2);

        // 1a and e1b are compatible types (but not identical)
        MI_CHECK_EQUAL( itf->is_compatible( e1a.get(), e2.get()), 0);

        IType_enum::Values values3;
        values3.push_back( std::make_pair( "un", 1));
        values3.push_back( std::make_pair( "deux", 2));
        values3.push_back( std::make_pair( "trois", 3));
        mi::base::Handle<const IType_enum> e3( itf->create_enum(
            "::my_enum3", IType_enum::EID_USER, values3, annotations, value_annotations, &errors));
        MI_CHECK_EQUAL( 0, errors);
        MI_CHECK( e3);

        MI_CHECK_EQUAL( itf->is_compatible( e1a.get(), e3.get()), -1);
        MI_CHECK_EQUAL( itf->is_compatible( e3.get(), e1a.get()), -1);
    }
    {
        mi::base::Handle<const IType_array> ar1( itf->create_immediate_sized_array( b.get(), 42));
        MI_CHECK( ar1);
        mi::base::Handle<const IType_array> ar2( itf->create_immediate_sized_array( b.get(), 43));
        MI_CHECK( ar2);

        // ar1 and ar2 have different sizes
        MI_CHECK_EQUAL( itf->is_compatible( ar1.get(), ar2.get()), -1);

        mi::base::Handle<const IType_array> ar3(
            itf->create_immediate_sized_array( ar_st1.get(), 42));
        mi::base::Handle<const IType_array> ar4(
            itf->create_immediate_sized_array( ar_st2.get(), 42));

        // ar1 and ar3 have different element types
        MI_CHECK_EQUAL( itf->is_compatible( ar1.get(), ar3.get()), -1);
        // ar3 and ar4 have compatible element types
        MI_CHECK_EQUAL( itf->is_compatible( ar3.get(), ar4.get()),  0);
    }
}

// Thread that repeatedly creates, queries, and destroys enum types.
class Enum_types_thread : public THREAD::Thread
{
public:
    void initialize( IType_factory* itf, mi::Size iteration_count)
    {
        m_itf = itf;
        m_iteration_count = iteration_count;
    }

    void run() final
    {
        mi::base::Handle<const IAnnotation_block> annotations;
        IType_enum::Value_annotations value_annotations;
        mi::Sint32 errors = 0;

        for( mi::Size i = 0; i < m_iteration_count; ++i) {

            mi::base::Handle<const IType_enum> e1( m_itf->create_enum(
                "::my_enum4",
                IType_enum::EID_USER,
                {{"my_enumerator", 42}},
                annotations,
                value_annotations,
                &errors));
            MI_CHECK_EQUAL( errors, 0);
            MI_CHECK( e1);

            mi::base::Handle<const IType_enum> e2( m_itf->create_enum( "::my_enum4"));
            MI_CHECK( e2);
            e2.reset();

            e1.reset();

            mi::base::Handle<const IType_enum> e3( m_itf->create_enum( "::my_enum4"));
            // Can succeed (other threads kept the enum registered) or fail.
            e3.reset();
        }
    }

private:
    IType_factory* m_itf;
    mi::Size m_iteration_count;
};

void test_enum_type_multithreaded()
{
    Type_factory tf;
    mi::Size thread_count = 10;
    mi::Size iteration_count = 100;

    auto* threads = new Enum_types_thread[thread_count];
    for( mi::Size i = 0; i < thread_count; ++i) {
        threads[i].initialize( &tf, iteration_count);
        threads[i].start();
    }
    for( mi::Size i = 0; i < thread_count; ++i) {
        threads[i].join();
    }

    delete[] threads;
}

// Thread that repeatedly creates, queries, and destroys struct types.
class Struct_types_thread : public THREAD::Thread
{
public:
    void initialize( IType_factory* itf, mi::Size iteration_count)
    {
        m_itf = itf;
        m_iteration_count = iteration_count;
    }

    void run() final
    {
        mi::base::Handle<const IType> int_type( m_itf->create_int());
        mi::base::Handle<const IAnnotation_block> annotations;
        IType_struct::Field_annotations field_annotations;
        mi::Sint32 errors = 0;

        for( mi::Size i = 0; i < m_iteration_count; ++i) {

            mi::base::Handle<const IType_struct> s1( m_itf->create_struct(
                "::my_struct4",
                IType_struct::SID_USER,
                {{int_type, "my_field"}},
                annotations,
                field_annotations,
                /*is_declarative*/ false,
                /*struct_category*/ nullptr,
                &errors));
            MI_CHECK_EQUAL( errors, 0);
            MI_CHECK( s1);

            mi::base::Handle<const IType_struct> s2( m_itf->create_struct( "::my_struct4"));
            MI_CHECK( s2);
            s2.reset();

            s1.reset();

            mi::base::Handle<const IType_struct> s3( m_itf->create_struct( "::my_struct4"));
            // Can succeed (other threads kept the struct registered) or fail.
            s3.reset();
        }
    }

private:
    IType_factory* m_itf;
    mi::Size m_iteration_count;
};

void test_struct_type_multithreaded()
{
    Type_factory tf;
    mi::Size thread_count = 10;
    mi::Size iteration_count = 100;

    auto* threads = new Struct_types_thread[thread_count];
    for( mi::Size i = 0; i < thread_count; ++i) {
        threads[i].initialize( &tf, iteration_count);
        threads[i].start();
    }
    for( mi::Size i = 0; i < thread_count; ++i) {
        threads[i].join();
    }

    delete[] threads;
}

MI_TEST_AUTO_FUNCTION( test )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    // For IMDL::is_valid_mdl_identifier() in IType_factory::create_deferred_sized_array().
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);

    test_type_factory();
    test_compare();
    test_compare_list();
    test_is_compatible();

    // TODO MDL-957 IType_enum::release() and IType_struct::release() are not thread-safe.
    // test_enum_type_multithreaded();
    // test_struct_type_multithreaded();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
