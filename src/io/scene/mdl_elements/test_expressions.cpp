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

#include "mdl_elements_expression.h"
#include "mdl_elements_type.h"
#include "mdl_elements_value.h"
#include "test_shared.h"

#include <boost/algorithm/string/replace.hpp>

#include <mi/neuraylib/istring.h>

#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serial_buffer_serializer.h>

using namespace MI;
using namespace MDL;

#define MI_CHECK_EQUAL_VALUES( lhs, rhs) \
    MI_CHECK_EQUAL( vf.compare( lhs, rhs), 0)

template<class T>
T* serialize_and_deserialize( const Expression_factory& ef, T* expr)
{
    SERIAL::Buffer_serializer serializer;
    mi::base::Handle<T> cloned_expr(
        ef.clone( expr, /*transaction*/ nullptr, /*copy_immutable_calls*/ false));
    ef.serialize( &serializer, cloned_expr.get());
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    mi::base::Handle<T> deserialized_expr(ef.deserialize<T>( &deserializer));
    T* cloned_and_deserialized_expr = ef.clone(
        deserialized_expr.get(), /*transaction*/ nullptr, /*copy_immutable_calls*/ false);
    return cloned_and_deserialized_expr;
}

IExpression_list* serialize_and_deserialize_list(
    const Expression_factory& ef, const IExpression_list* expression_list)
{
    SERIAL::Buffer_serializer serializer;
    mi::base::Handle<IExpression_list> cloned_expression_list( ef.clone(
        expression_list, /*transaction*/ nullptr, /*copy_immutable_calls*/ false));
    ef.serialize_list( &serializer, cloned_expression_list.get());
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    mi::base::Handle<IExpression_list> deserialized_expression_list(
        ef.deserialize_list( &deserializer));
    IExpression_list* cloned_deserialized_expression_list
        = ef.clone(deserialized_expression_list.get(),
            /*transaction*/ nullptr, /*copy_immutable_calls*/ false);
    return cloned_deserialized_expression_list;
}

IAnnotation_block* serialize_and_deserialize_annotation_block(
    const Expression_factory& ef, const IAnnotation_block* annotation_block)
{
    SERIAL::Buffer_serializer serializer;
    ef.serialize_annotation_block( &serializer, annotation_block);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return ef.deserialize_annotation_block( &deserializer);
}

IAnnotation_list* serialize_and_deserialize_annotation_list(
    const Expression_factory& ef, const IAnnotation_list* annotation_list)
{
    SERIAL::Buffer_serializer serializer;
    ef.serialize_annotation_list( &serializer, annotation_list);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return ef.deserialize_annotation_list( &deserializer);
}

void check_dump(
    DB::Transaction* transaction,
    Expression_factory& ef,
    const IExpression* expr,
    const char* expected)
{
    mi::base::Handle<const mi::IString> dump( ef.dump( transaction, expr, "foo"));
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), expected);

    mi::base::Handle<IExpression_list> list( ef.create_expression_list( 0));
    dump = ef.dump( transaction, list.get(), "bar");
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), "expression_list bar = [ ]");

    MI_CHECK_EQUAL( 0, list->add_expression( "foo", expr));
    dump = ef.dump( transaction, list.get(), "bar");
    std::string list_expected = "expression_list bar = [\n    ";
    list_expected += boost::replace_all_copy(
                         boost::replace_all_copy( std::string( expected), "\n", "\n    "),
                         "foo", "0: foo");
    list_expected += ";\n]";
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(), list_expected.c_str());
}

void check_list( const Expression_factory& ef, IExpression* expression)
{
    mi::base::Handle<IExpression_list> el( ef.create_expression_list( 1));
    MI_CHECK_EQUAL( el->add_expression( "foo", expression), 0);
    el = serialize_and_deserialize_list( ef, el.get());
    mi::base::Handle<const IExpression> expression2( el->get_expression( "foo"));
    MI_CHECK( expression2);
    MI_CHECK_EQUAL( el->get_size(), 1);
    MI_CHECK_EQUAL_CSTR( el->get_name( 0), "foo");
    MI_CHECK( !el->get_name( 1));
    MI_CHECK_EQUAL( el->get_index( "foo"), 0);
    MI_CHECK_EQUAL( el->get_index( "bar"), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( el->add_expression( "foo", expression), -2);
    MI_CHECK_EQUAL( el->set_expression( "foo", expression), 0);
}

void check_list(
    DB::Transaction* transaction, const Expression_factory& ef, IAnnotation_block* annotation_block)
{
    mi::base::Handle<IAnnotation_list> al( ef.create_annotation_list( 1));
    MI_CHECK_EQUAL( al->add_annotation_block( "foo", annotation_block), 0);
    al = serialize_and_deserialize_annotation_list( ef, al.get());
    mi::base::Handle<const IAnnotation_block> annotation_block2(
        al->get_annotation_block( "foo"));
    MI_CHECK( annotation_block2);
    MI_CHECK_EQUAL( al->get_size(), 1);
    MI_CHECK_EQUAL_CSTR( al->get_name( 0), "foo");
    MI_CHECK( !al->get_name( 1));
    MI_CHECK_EQUAL( al->get_index( "foo"), 0);
    MI_CHECK_EQUAL( al->get_index( "bar"), static_cast<mi::Size>( -1));
    MI_CHECK_EQUAL( al->add_annotation_block( "foo", annotation_block), -2);
    MI_CHECK_EQUAL( al->set_annotation_block( "foo", annotation_block), 0);

    mi::base::Handle<const mi::IString> dump;
    dump = ef.dump( transaction, al.get(), "bar");
    MI_CHECK_EQUAL_CSTR( dump->get_c_str(),
        "annotation_list bar = [\n"
        "    annotation_block 0: foo = [\n"
        "        annotation 0 = \"my_anno\" (\n"
        "            constant bool foo = true\n"
        "        );\n"
        "    ];\n"
        "]");
}

void test_constant( DB::Transaction* transaction)
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());

    Value_factory vf( &tf);
    mi::base::Handle<IValue_bool> bv, bv2;
    bv = vf.create_bool( true);
    bv2 = vf.create_bool( false);

    Expression_factory ef( &vf);
    mi::base::Handle<IExpression_constant> e( ef.create_constant( bv.get()));
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK( e);

    MI_CHECK_EQUAL( e->get_kind(), IExpression::EK_CONSTANT);
    mi::base::Handle<const IType> t( e->get_type());
    MI_CHECK_EQUAL( tf.compare( t.get(), b.get()), 0);

    mi::base::Handle<IValue_bool> bv3( e->get_value<IValue_bool>());
    MI_CHECK_EQUAL_VALUES( bv3.get(), bv.get());

    e->set_value( bv2.get());
    e = serialize_and_deserialize( ef, e.get());
    bv3 = e->get_value<IValue_bool>();
    MI_CHECK_EQUAL_VALUES( bv3.get(), bv2.get());

    check_dump( /*transaction*/ nullptr, ef, e.get(), "constant bool foo = false");
    check_list( ef, e.get());
}

void test_call( DB::Transaction* transaction)
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());

    Value_factory vf( &tf);

    // Unfortunately, no real DB elements for function calls exist in this simple test.
    // Use fake tags 42 and 43.

    Expression_factory ef( &vf);
    mi::base::Handle<IExpression_call> e( ef.create_call( b.get(), DB::Tag( 42)));
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK( e);

    MI_CHECK_EQUAL( e->get_kind(), IExpression::EK_CALL);
    MI_CHECK_EQUAL( DB::Tag( 42), e->get_call());

    check_dump( /*transaction*/ nullptr, ef, e.get(), "call foo = tag 42");
    check_list( ef, e.get());
}

void test_parameter()
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());

    Value_factory vf( &tf);

    Expression_factory ef( &vf);
    mi::base::Handle<IExpression_parameter> e( ef.create_parameter( b.get(), 42));
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK( e);

    MI_CHECK_EQUAL( e->get_kind(), IExpression::EK_PARAMETER);
    MI_CHECK_EQUAL( e->get_index(), 42);
    e->set_index( 43);
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK_EQUAL( e->get_index(), 43);

    check_dump( /*transaction*/ nullptr, ef, e.get(), "parameter bool foo = index 43");
    check_list( ef, e.get());
}

void test_direct_call( DB::Transaction* transaction)
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());

    Value_factory vf( &tf);

    // Unfortunately, no real DB elements for function calls exist in this simple test.
    // Use fake tag 42.

    Expression_factory ef( &vf);
    mi::base::Handle<IExpression_list> args( ef.create_expression_list( 0));
    mi::base::Handle<IExpression_direct_call> e( ef.create_direct_call(
        b.get(), DB::Tag( 43), Mdl_tag_ident( 42, 1), "test", args.get()));
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK( e);

    MI_CHECK_EQUAL( e->get_kind(), IExpression::EK_DIRECT_CALL);
    MI_CHECK_EQUAL( DB::Tag( 42), e->get_definition( /*transaction*/ nullptr));
    MI_CHECK_EQUAL( DB::Tag( 43), e->get_module());
    check_dump( /*transaction*/ nullptr, ef, e.get(), "direct call foo = tag 42 ()");
    check_list( ef, e.get());
}

void test_temporary()
{
    Type_factory tf;
    mi::base::Handle<const IType_bool> b( tf.create_bool());

    Value_factory vf( &tf);

    Expression_factory ef( &vf);
    mi::base::Handle<IExpression_temporary> e( ef.create_temporary( b.get(), 42));
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK( e);

    MI_CHECK_EQUAL( e->get_kind(), IExpression::EK_TEMPORARY);
    MI_CHECK_EQUAL( e->get_index(), 42);
    e->set_index( 43);
    e = serialize_and_deserialize( ef, e.get());
    MI_CHECK_EQUAL( e->get_index(), 43);

    check_dump( /*transaction*/ nullptr, ef, e.get(), "temporary foo = index 43");
    check_list( ef, e.get());
}

void test_compare( DB::Transaction* transaction)
{
    Type_factory tf;
    Value_factory vf( &tf);
    Expression_factory ef( &vf);
    IExpression_factory* ief = &ef; // use only the general overload

    mi::base::Handle<const IType_bool> b( tf.create_bool());
    mi::base::Handle<const IType_int> i( tf.create_int());

    // compare constants
    mi::base::Handle<IValue_bool> bv1, bv2, bv3;
    bv1 = vf.create_bool( false);
    bv2 = vf.create_bool( true);
    mi::base::Handle<IExpression_constant> ec1, ec2, ec3;
    ec1 = ef.create_constant( bv1.get());
    ec2 = ef.create_constant( bv2.get());
    ec3 = ef.create_constant( bv2.get());
    MI_CHECK_EQUAL( ief->compare( nullptr, ec1.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ec1.get(), nullptr), +1);
    MI_CHECK_EQUAL( ief->compare( ec1.get(), ec2.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ec2.get(), ec3.get()),  0);
    MI_CHECK_EQUAL( ief->compare( ec3.get(), ec1.get()), +1);

    // compare calls
    //
    // Unfortunately, no real DB elements for function calls exist in this simple test.
    // Use fake tags 42 and 43. See test_misc.cpp for a more elaborate test.
    mi::base::Handle<IExpression_call> eca1, eca2, eca3, eca4;
    eca1 = ef.create_call( b.get(), DB::Tag( 42));
    eca2 = ef.create_call( i.get(), DB::Tag( 42));
    eca3 = ef.create_call( i.get(), DB::Tag( 42));
    eca4 = ef.create_call( i.get(), DB::Tag( 43));
    MI_CHECK_EQUAL( ief->compare( eca1.get(), eca2.get()), -1);
    MI_CHECK_EQUAL( ief->compare( eca2.get(), eca3.get()),  0);
    MI_CHECK_EQUAL( ief->compare( eca3.get(), eca4.get()), -1);
    MI_CHECK_EQUAL( ief->compare( eca4.get(), eca1.get()), +1);

    // compare parameters
    mi::base::Handle<IExpression_parameter> ep1, ep2, ep3, ep4;
    ep1 = ef.create_parameter( b.get(), 0);
    ep2 = ef.create_parameter( i.get(), 0);
    ep3 = ef.create_parameter( i.get(), 42);
    ep4 = ef.create_parameter( i.get(), 42);
    MI_CHECK_EQUAL( ief->compare( ep1.get(), ep1.get()),  0);
    MI_CHECK_EQUAL( ief->compare( ep1.get(), ep2.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ep1.get(), ep3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ep1.get(), ep4.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ep2.get(), ep1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( ep2.get(), ep2.get()),  0);
    MI_CHECK_EQUAL( ief->compare( ep2.get(), ep3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ep2.get(), ep4.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ep3.get(), ep1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( ep3.get(), ep2.get()), +1);
    MI_CHECK_EQUAL( ief->compare( ep3.get(), ep3.get()),  0);
    MI_CHECK_EQUAL( ief->compare( ep3.get(), ep4.get()),  0);
    MI_CHECK_EQUAL( ief->compare( ep4.get(), ep1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( ep4.get(), ep2.get()), +1);
    MI_CHECK_EQUAL( ief->compare( ep4.get(), ep3.get()),  0);
    MI_CHECK_EQUAL( ief->compare( ep4.get(), ep4.get()),  0);

    // compare direct calls
    //
    // Unfortunately, no real DB elements for function calls exist in this simple test.
    // Use fake tags 42 and 43.
    mi::base::Handle<IExpression_list> args1, args2;
    args1 = ef.create_expression_list( 0);
    args2 = ef.create_expression_list( 1);
    args2->add_expression( "foo", ec1.get());
    MI_CHECK_EQUAL( ief->compare( args1.get(), args2.get()), -1);
    mi::base::Handle<IExpression_direct_call> edc1, edc2, edc3, edc4, edc5;
    edc1 = ef.create_direct_call( b.get(), DB::Tag( 42), Mdl_tag_ident(43, 1), "test", args1.get());
    edc2 = ef.create_direct_call( b.get(), DB::Tag( 42), Mdl_tag_ident(43, 1), "test", args1.get());
    edc3 = ef.create_direct_call( i.get(), DB::Tag( 42), Mdl_tag_ident(43, 1), "test", args1.get());
    edc4 = ef.create_direct_call( i.get(), DB::Tag( 43), Mdl_tag_ident(44, 1), "test", args1.get());
    edc5 = ef.create_direct_call( i.get(), DB::Tag( 43), Mdl_tag_ident(44, 1), "test", args2.get());
    MI_CHECK_EQUAL( ief->compare( edc1.get(), edc2.get()),  0);
    MI_CHECK_EQUAL( ief->compare( edc2.get(), edc3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( edc3.get(), edc4.get()), -1);
    MI_CHECK_EQUAL( ief->compare( edc4.get(), edc5.get()), -1);
    MI_CHECK_EQUAL( ief->compare( edc5.get(), edc1.get()), +1);

    // compare temporaries
    mi::base::Handle<IExpression_temporary> et1, et2, et3, et4;
    et1 = ef.create_temporary( b.get(), 0);
    et2 = ef.create_temporary( i.get(), 0);
    et3 = ef.create_temporary( i.get(), 42);
    et4 = ef.create_temporary( i.get(), 42);
    MI_CHECK_EQUAL( ief->compare( et1.get(), et1.get()),  0);
    MI_CHECK_EQUAL( ief->compare( et1.get(), et2.get()), -1);
    MI_CHECK_EQUAL( ief->compare( et1.get(), et3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( et1.get(), et4.get()), -1);
    MI_CHECK_EQUAL( ief->compare( et2.get(), et1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et2.get(), et2.get()),  0);
    MI_CHECK_EQUAL( ief->compare( et2.get(), et3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( et2.get(), et4.get()), -1);
    MI_CHECK_EQUAL( ief->compare( et3.get(), et1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et3.get(), et2.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et3.get(), et3.get()),  0);
    MI_CHECK_EQUAL( ief->compare( et3.get(), et4.get()),  0);
    MI_CHECK_EQUAL( ief->compare( et4.get(), et1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et4.get(), et2.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et4.get(), et3.get()),  0);
    MI_CHECK_EQUAL( ief->compare( et4.get(), et4.get()),  0);

    // compare different kinds
    MI_CHECK_EQUAL( ief->compare( ec1.get() , ec1.get()) ,  0);
    MI_CHECK_EQUAL( ief->compare( ec1.get() , eca1.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ec1.get() , ep1.get()),  -1);
    MI_CHECK_EQUAL( ief->compare( ec1.get() , edc1.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ec1.get() , et1.get()),  -1);
    MI_CHECK_EQUAL( ief->compare( eca1.get(), ec1.get()) , +1);
    MI_CHECK_EQUAL( ief->compare( eca1.get(), eca1.get()),  0);
    MI_CHECK_EQUAL( ief->compare( eca1.get(), ep1.get()),  -1);
    MI_CHECK_EQUAL( ief->compare( eca1.get(), edc1.get()), -1);
    MI_CHECK_EQUAL( ief->compare( eca1.get(), et1.get()),  -1);
    MI_CHECK_EQUAL( ief->compare( ep1.get(),  ec1.get()) , +1);
    MI_CHECK_EQUAL( ief->compare( ep1.get(),  eca1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( ep1.get(),  ep1.get()),   0);
    MI_CHECK_EQUAL( ief->compare( ep1.get(),  edc1.get()), -1);
    MI_CHECK_EQUAL( ief->compare( ep1.get(),  et1.get()),  -1);
    MI_CHECK_EQUAL( ief->compare( edc1.get(), ec1.get()) , +1);
    MI_CHECK_EQUAL( ief->compare( edc1.get(), eca1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( edc1.get(), ep1.get()),  +1);
    MI_CHECK_EQUAL( ief->compare( edc1.get(), edc1.get()),  0);
    MI_CHECK_EQUAL( ief->compare( edc1.get(), et1.get()),  -1);
    MI_CHECK_EQUAL( ief->compare( et1.get(),  ec1.get()) , +1);
    MI_CHECK_EQUAL( ief->compare( et1.get(),  eca1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et1.get(),  ep1.get()),  +1);
    MI_CHECK_EQUAL( ief->compare( et1.get(),  edc1.get()), +1);
    MI_CHECK_EQUAL( ief->compare( et1.get(),  et1.get()),   0);
}

void test_compare_list()
{
    Type_factory tf;
    Value_factory vf( &tf);
    Expression_factory ef( &vf);
    IExpression_factory* ief = &ef; // use only the general overload

    mi::base::Handle<IExpression_list> m1( ef.create_expression_list( 1));
    mi::base::Handle<IExpression_list> m2( ef.create_expression_list( 1));
    mi::base::Handle<IExpression_list> m3( ef.create_expression_list( 1));

    mi::base::Handle<IValue_int> iv1( vf.create_int( 0));
    mi::base::Handle<IValue_int> iv2( vf.create_int( 1));

    mi::base::Handle<IExpression_constant> ec1( ef.create_constant( iv1.get()));
    mi::base::Handle<IExpression_constant> ec2( ef.create_constant( iv2.get()));

    // compare list sizes
    MI_CHECK_EQUAL( m3->add_expression( "foo", ec1.get()), 0);
    MI_CHECK_EQUAL( ief->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( ief->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( m3.get(), m1.get()), +1);

    // compare element names
    MI_CHECK_EQUAL( m1->add_expression( "bar", ec1.get()), 0);
    MI_CHECK_EQUAL( m2->add_expression( "bar", ec1.get()), 0);
    MI_CHECK_EQUAL( ief->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( ief->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( m3.get(), m1.get()), +1);

    // compare element expressions
    m3 = ef.create_expression_list( 1);
    MI_CHECK_EQUAL( m3->add_expression( "bar", ec2.get()), 0);
    MI_CHECK_EQUAL( ief->compare( m1.get(), m2.get()),  0);
    MI_CHECK_EQUAL( ief->compare( m2.get(), m3.get()), -1);
    MI_CHECK_EQUAL( ief->compare( m3.get(), m1.get()), +1);
}

MI_TEST_AUTO_FUNCTION( test )
{
    Unified_database_access db_access;

    DB::Database* database = db_access.get_database();
    DB::Scope* scope = database->get_global_scope();
    DB::Transaction* transaction = scope->start_transaction();

    test_constant( transaction);
    test_call( transaction);
    test_parameter();
    test_direct_call( transaction);
    test_temporary();

    test_compare( transaction);
    test_compare_list();

    transaction->commit();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
