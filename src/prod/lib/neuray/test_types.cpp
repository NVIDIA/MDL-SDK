/******************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/ipointer.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/iuuid.h>

#include <mi/neuraylib/iattribute_container.h>

#include "test_shared.h"

#define GET_REFCOUNT(X) ((X) ? (X)->retain(), (X)->release() : 999)

template <typename I, typename T>
void test_value(
    mi::neuraylib::ITransaction* transaction, const char* type_name,
    T old_value, T new_value)
{
    // test for the standard variant

    // create instance
    mi::base::Handle<I> value( transaction->create<I>( type_name));
    MI_CHECK( value);

    // check the type name
    MI_CHECK_EQUAL_CSTR( value->get_type_name(), type_name);

    T tmp;

    // check initial value
    tmp = value->template get_value<T>();
    MI_CHECK_EQUAL( T(0), tmp);

    // modify value
    value->set_value( new_value);
    tmp = value->template get_value<T>();
    MI_CHECK_EQUAL( new_value, tmp);
    value->get_value( tmp);
    MI_CHECK_EQUAL( new_value, tmp);

    // destroy instance
    value = nullptr;
}

template <typename I, typename T>
void test_value_attribute(
    mi::neuraylib::ITransaction* transaction, const char* type_name,
    T old_value, T new_value)
{
    // create a dummy owner
    mi::base::Handle<mi::neuraylib::IAttribute_container> owner(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( owner);

    // create the attribute
    mi::base::Handle<I> value( owner->create_attribute<I>( "attribute", type_name));
    MI_CHECK( value);

    // check the type name
    MI_CHECK_EQUAL_CSTR( value->get_type_name(), type_name);

    T tmp;

    // check initial value
    value->get_value( tmp);
    MI_CHECK_EQUAL( T(0), tmp);

    // modify value
    value->set_value( new_value);
    value->get_value( tmp);
    MI_CHECK_EQUAL( new_value, tmp);

    // test creation without explicit type name parameter
    value = owner->create_attribute<I>( "attribute2");
    MI_CHECK( value);
    MI_CHECK_EQUAL_CSTR( value->get_type_name(), type_name);

    // destroy instance
    value = nullptr;

    MI_CHECK_EQUAL( 1, GET_REFCOUNT( owner));
    owner = nullptr;
}

void test_string(
    mi::neuraylib::ITransaction* transaction, const char* type_name,
    const char* old_value, const char* new_value)
{
    // test for the standard variant

    // create instance
    mi::base::Handle<mi::IString> str( transaction->create<mi::IString>( type_name));
    MI_CHECK( str);

    // check the type name
    MI_CHECK_EQUAL_CSTR( str->get_type_name(), type_name);

    const char* tmp;

    // check initial value
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( "", tmp);

    // modify value
    str->set_c_str( old_value);
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( old_value, tmp);

    // test 0
    str->set_c_str( nullptr);
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( "", tmp);

    // test ""
    str->set_c_str( "");
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( "", tmp);

    // modify value again (test destructor)
    str->set_c_str( new_value);
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( new_value, tmp);

    // destroy instance
    str = nullptr;

    // test for the proxy variant

    // create a dummy owner
    mi::base::Handle<mi::neuraylib::IAttribute_container> owner(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( owner);

    // create the attribute
    str = owner->create_attribute<mi::IString>( "attribute", "String");
    MI_CHECK( str);

    // check the type name
    MI_CHECK_EQUAL_CSTR( str->get_type_name(), type_name);

    // check initial value
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( "", tmp);

    // modify value
    str->set_c_str( new_value);
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( new_value, tmp);

    // test 0
    str->set_c_str( nullptr);
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( "", tmp);

    // test ""
    str->set_c_str( "");
    tmp = str->get_c_str();
    MI_CHECK_EQUAL_CSTR( "", tmp);

    // test creation without explicit type name parameter
    str = owner->create_attribute<mi::IString>( "attribute2");
    MI_CHECK( str);
    MI_CHECK_EQUAL_CSTR( str->get_type_name(), "String");

    // destroy instance
    str = nullptr;
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( owner));
    owner = nullptr;
}


void test_ref( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    std::string type_name_str = type_name;

    std::string dummy_type_name;
    if( type_name_str == "Ref")
        dummy_type_name = "Attribute_container";
    else if( type_name_str.substr( 0, 11) == "Ref<Texture")
        dummy_type_name = "Texture";
    else if( type_name_str == "Ref<Lightprofile>")
        dummy_type_name = "Lightprofile";
    else if( type_name_str == "Ref<Bsdf_measurement>")
        dummy_type_name = "Bsdf_measurement";
    else
        MI_CHECK( false);

    std::string dummy_name  = "the_" + dummy_type_name;

    mi::base::Handle<mi::base::IInterface> m_dummy;
    mi::base::Handle<const mi::base::IInterface> c_dummy;
    mi::base::Handle<mi::base::IInterface> iinterface;

    // create dummy that only work for untyped references
    m_dummy = transaction->create( "Attribute_container");
    MI_CHECK_EQUAL( 0, transaction->store( m_dummy.get(), "the_Attribute_container"));

    // create dummy that should always work
    m_dummy = transaction->create( dummy_type_name.c_str());
    MI_CHECK_EQUAL( 0, transaction->store( m_dummy.get(), dummy_name.c_str()));

    // test for the standard variant

    // create instance
    mi::base::Handle<mi::IRef> ref( transaction->create<mi::IRef>( type_name));
    MI_CHECK( ref);

    // check the type name
    MI_CHECK_EQUAL_CSTR( ref->get_type_name(), type_name);

    // check initial value
    iinterface = ref->get_reference();
    MI_CHECK( !iinterface);

    // modify value via pointer (Attribute_container)
    c_dummy = transaction->access( "the_Attribute_container");
    MI_CHECK( c_dummy);
    ref->set_reference( c_dummy.get());
    c_dummy = nullptr;

    // check value (works only for untyped references), reset
    iinterface = ref->get_reference();
    if( type_name_str == "Ref") {
        MI_CHECK(iinterface);
        ref->set_reference( static_cast<mi::base::IInterface*>( nullptr));
        iinterface = ref->get_reference();
        MI_CHECK( !iinterface);
    } else {
        MI_CHECK( !iinterface);
    }

    // modify value via pointer (correct type)
    c_dummy = transaction->access( dummy_name.c_str());
    MI_CHECK( c_dummy);
    ref->set_reference( c_dummy.get());
    c_dummy = nullptr;

    // check value (always works), reset
    iinterface = ref->get_reference();
    MI_CHECK(iinterface);
    ref->set_reference( static_cast<mi::base::IInterface*>( nullptr));
    iinterface = ref->get_reference();
    MI_CHECK( !iinterface);

    // modify value via string (Attribute_container)
    ref->set_reference( "the_Attribute_container");

    // check value (works only for untyped references), reset
    if( type_name_str == "Ref") {
        MI_CHECK_EQUAL_CSTR( ref->get_reference_name(), "the_Attribute_container");
        ref->set_reference( zero_string);
        MI_CHECK_EQUAL( ref->get_reference_name(), nullptr);
    } else {
        MI_CHECK_EQUAL( ref->get_reference_name(), nullptr);
    }

    // modify value via string (correct type)
    ref->set_reference( dummy_name.c_str());

    // check value (always works), reset
    MI_CHECK_EQUAL_CSTR( ref->get_reference_name(), dummy_name.c_str());
    ref->set_reference( zero_string);
    MI_CHECK_EQUAL( ref->get_reference_name(), nullptr);

    // destroy instance
    ref = nullptr;

    // test for the proxy variant

    // create a dummy owner
    mi::base::Handle<mi::neuraylib::IAttribute_container> owner(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( owner);

    // create the attribute
    ref = owner->create_attribute<mi::IRef>( "attribute", type_name);
    MI_CHECK( ref);

    // check the type name
    MI_CHECK_EQUAL_CSTR( ref->get_type_name(), type_name);

    // check initial value
    iinterface = ref->get_reference();
    MI_CHECK( !iinterface);

    // modify value via pointer (Attribute_container)
    c_dummy = transaction->access( "the_Attribute_container");
    MI_CHECK( c_dummy);
    ref->set_reference( c_dummy.get());
    c_dummy = nullptr;

    // check value (works only for untyped references), reset
    iinterface = ref->get_reference();
    if( type_name_str == "Ref") {
        MI_CHECK(iinterface);
        ref->set_reference( static_cast<mi::base::IInterface*>( nullptr));
        iinterface = ref->get_reference();
        MI_CHECK( !iinterface);
    } else {
        MI_CHECK( !iinterface);
    }

    // modify value via pointer (correct type)
    c_dummy = transaction->access( dummy_name.c_str());
    MI_CHECK( c_dummy);
    ref->set_reference( c_dummy.get());
    c_dummy = nullptr;

    // check value (always works), reset
    iinterface = ref->get_reference();
    MI_CHECK(iinterface);
    ref->set_reference( static_cast<mi::base::IInterface*>( nullptr));
    iinterface = ref->get_reference();
    MI_CHECK( !iinterface);

    // modify value via string (Attribute_container)
    ref->set_reference( "the_Attribute_container");

    // check value (works only for untyped references), reset
    if( type_name_str == "Ref") {
        MI_CHECK_EQUAL_CSTR( ref->get_reference_name(), "the_Attribute_container");
        ref->set_reference( zero_string);
        MI_CHECK_EQUAL( ref->get_reference_name(), nullptr);
    } else {
        MI_CHECK_EQUAL( ref->get_reference_name(), nullptr);
    }

    // modify value via string (correct type)
    ref->set_reference( dummy_name.c_str());

    // check value (always works), reset
    MI_CHECK_EQUAL_CSTR( ref->get_reference_name(), dummy_name.c_str());
    ref->set_reference( zero_string);
    MI_CHECK_EQUAL( ref->get_reference_name(), nullptr);

    // test creation without explicit type name temporary
    ref = owner->create_attribute<mi::IRef>( "attribute2");
    MI_CHECK( ref);
    if( type_name_str == "Ref")
        MI_CHECK_EQUAL_CSTR( ref->get_type_name(), type_name);
    else
        MI_CHECK_EQUAL_CSTR( ref->get_type_name(), "Ref");

    // destroy instance
    ref = nullptr;
}

void test_ref_array( mi::neuraylib::ITransaction* transaction)
{
    // create dummy elements
    mi::base::Handle<mi::neuraylib::IAttribute_container> m_attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( m_attribute_container);
    MI_CHECK_EQUAL( 0, transaction->store( m_attribute_container.get(), "ref_array_attribute_container_1"));

    m_attribute_container = transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container");
    MI_CHECK( m_attribute_container);
    MI_CHECK_EQUAL( 0, transaction->store( m_attribute_container.get(), "ref_array_attribute_container_2"));
    m_attribute_container = nullptr;

    // create a dummy owner
    mi::base::Handle<mi::neuraylib::IAttribute_container> owner(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( owner);

    // create array attribute
    const mi::Size N = 10;
    std::ostringstream str;
    str << "Ref[" << N << "]";
    mi::base::Handle<mi::IArray> array( owner->create_attribute<mi::IArray>( "ref_array", str.str().c_str()));
    MI_CHECK( array);
    MI_CHECK( array->get_length() == N);
    for( mi::Size i = 0; i < N; ++i) {
        mi::base::Handle<mi::IRef> ref( array->get_element<mi::IRef>( i));
        MI_CHECK( ref);
        mi::base::Handle<mi::base::IInterface> iinterface( ref->get_reference());
        MI_CHECK( !iinterface);
    }

    // modify value via pointer
    mi::base::Handle<const mi::neuraylib::IAttribute_container> c_attribute_container(
        transaction->access<mi::neuraylib::IAttribute_container>( "ref_array_attribute_container_1"));
    MI_CHECK( c_attribute_container);
    mi::base::Handle<mi::IRef> ref( array->get_element<mi::IRef>( 1));
    MI_CHECK( ref);
    ref->set_reference( c_attribute_container.get());
    c_attribute_container = nullptr;

    MI_CHECK_EQUAL_CSTR( "ref_array_attribute_container_1", ref->get_reference_name());

    // modify value via string
    ref = array->get_element<mi::IRef>( 2);
    MI_CHECK( ref);
    ref->set_reference( "ref_array_attribute_container_2");

    MI_CHECK_EQUAL_CSTR( "ref_array_attribute_container_2", ref->get_reference_name());

    // access array attribute and check values
    mi::base::Handle<const mi::IArray> c_array;
    mi::base::Handle<const mi::IRef> c_ref;
    mi::base::Handle<const mi::base::IInterface> c_interface;
    c_array = owner->access_attribute<mi::IArray>( "ref_array");
    MI_CHECK( c_array);
    MI_CHECK( c_array->get_length() == N);
    for( mi::Size i = 0; i < N; ++i) {
        c_ref = array->get_element<mi::IRef>( i);
        MI_CHECK( c_ref);
        c_interface = c_ref->get_reference();
        MI_CHECK( (i == 1) || (i == 2) || !c_interface);
        MI_CHECK( (i == 0) || (i >= 3) ||  c_interface);
    }
    c_ref = c_array->get_element<mi::IRef>( 1);
    MI_CHECK( c_ref);
    MI_CHECK_EQUAL_CSTR( "ref_array_attribute_container_1", c_ref->get_reference_name());
    c_ref = c_array->get_element<mi::IRef>( 2);
    MI_CHECK( c_ref);
    MI_CHECK_EQUAL_CSTR( "ref_array_attribute_container_2", c_ref->get_reference_name());
    c_ref = nullptr;
    c_array = nullptr;

    // edit array attribute and check values
    mi::base::Handle<mi::base::IInterface> iinterface;
    array = owner->edit_attribute<mi::IArray>( "ref_array");
    MI_CHECK( array);
    MI_CHECK( array->get_length() == N);
    for( mi::Size i = 0; i < N; ++i) {
        ref = array->get_element<mi::IRef>( i);
        MI_CHECK( ref);
        iinterface = ref->get_reference();
        MI_CHECK( (i == 1) || (i == 2) || !iinterface);
        MI_CHECK( (i == 0) || (i >= 3) || iinterface);
    }
    ref = array->get_element<mi::IRef>( 1);
    MI_CHECK( ref);
    MI_CHECK_EQUAL_CSTR( "ref_array_attribute_container_1", ref->get_reference_name());
    ref = array->get_element<mi::IRef>( 2);
    MI_CHECK( ref);
    MI_CHECK_EQUAL_CSTR( "ref_array_attribute_container_2", ref->get_reference_name());
    ref = nullptr;
    array = nullptr;
}


void test_void( mi::neuraylib::ITransaction* transaction)
{
    // create instance
    mi::base::Handle<mi::IVoid> voidd( transaction->create<mi::IVoid>( "Void"));
    MI_CHECK( voidd);

    // check the type name
    MI_CHECK_EQUAL_CSTR( voidd->get_type_name(), "Void");

    // destroy instance
    voidd = nullptr;
}

void test_uuid( mi::neuraylib::ITransaction* transaction)
{
    // create instance
    mi::base::Handle<mi::IUuid> uuid( transaction->create<mi::IUuid>( "Uuid"));
    MI_CHECK( uuid);

    // check the type name
    MI_CHECK_EQUAL_CSTR( uuid->get_type_name(), "Uuid");

    mi::base::Uuid uuid1;
    uuid1.m_id1 = 0;
    uuid1.m_id2 = 0;
    uuid1.m_id3 = 0;
    uuid1.m_id4 = 0;
    mi::base::Uuid uuid2;
    uuid2.m_id1 = 0x12345678;
    uuid2.m_id2 = 0x87654321;
    uuid2.m_id3 = 0x00000001;
    uuid2.m_id4 = 0x10000000;

    mi::base::Uuid uuid_tmp = uuid->get_uuid();
    MI_CHECK( uuid_tmp == uuid1);

    uuid->set_uuid( uuid2);
    uuid_tmp = uuid->get_uuid();
    MI_CHECK( uuid_tmp == uuid2);

    // destroy instance
    uuid = nullptr;
}

void test_pointer( mi::neuraylib::ITransaction* transaction)
{
    // dummies
    mi::base::Handle<mi::ISint32> m_sint32( transaction->create<mi::ISint32>( "Sint32"));
    mi::base::Handle<const mi::ISint32> c_sint32( transaction->create<mi::ISint32>( "Sint32"));
    mi::base::Handle<mi::neuraylib::IAttribute_container> m_attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    mi::base::Handle<const mi::neuraylib::IAttribute_container> c_attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));

    // typed IPointer

    // create instance
    mi::base::Handle<mi::IPointer> p_sint32( transaction->create<mi::IPointer>( "Pointer<Sint32>"));
    MI_CHECK( p_sint32);

    // check the type name
    MI_CHECK_EQUAL_CSTR( p_sint32->get_type_name(), "Pointer<Sint32>");

    // check initial value
    MI_CHECK_EQUAL( nullptr, p_sint32->get_pointer());

    // set value
    MI_CHECK_EQUAL( 0, p_sint32->set_pointer( nullptr));
    MI_CHECK_EQUAL( 0, p_sint32->set_pointer( m_sint32.get()));
    MI_CHECK_EQUAL( -1, p_sint32->set_pointer( m_attribute_container.get()));

    // get value
    mi::base::Handle<mi::ISint32> m_sint32_tmp( p_sint32->get_pointer<mi::ISint32>());
    MI_CHECK_EQUAL( m_sint32.get(), m_sint32_tmp.get());

    // untyped IPointer

    // create instance
    mi::base::Handle<mi::IPointer> p_untyped( transaction->create<mi::IPointer>( "Pointer<Interface>"));
    MI_CHECK( p_untyped);

    // check the type name
    MI_CHECK_EQUAL_CSTR( p_untyped->get_type_name(), "Pointer<Interface>");

    // check initial value
    MI_CHECK_EQUAL( nullptr, p_untyped->get_pointer());

    // set value
    MI_CHECK_EQUAL( 0, p_untyped->set_pointer( nullptr));
    MI_CHECK_EQUAL( 0, p_untyped->set_pointer( m_sint32.get()));
    MI_CHECK_EQUAL( 0, p_untyped->set_pointer( m_attribute_container.get()));

    // get value
    mi::base::Handle<mi::neuraylib::IAttribute_container> m_attribute_container_tmp(
        p_untyped->get_pointer<mi::neuraylib::IAttribute_container>());
    MI_CHECK_EQUAL( m_attribute_container.get(), m_attribute_container_tmp.get());

    // typed IConst_pointer

    // create instance
    mi::base::Handle<mi::IConst_pointer> cp_sint32( transaction->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    MI_CHECK( cp_sint32);

    // check the type name
    MI_CHECK_EQUAL_CSTR( cp_sint32->get_type_name(), "Const_pointer<Sint32>");

    // check initial value
    MI_CHECK_EQUAL( nullptr, cp_sint32->get_pointer());

    // set value
    MI_CHECK_EQUAL( 0, cp_sint32->set_pointer( nullptr));
    MI_CHECK_EQUAL( 0, cp_sint32->set_pointer( c_sint32.get()));
    MI_CHECK_EQUAL( -1, cp_sint32->set_pointer( c_attribute_container.get()));

    // get value
    mi::base::Handle<const mi::ISint32> c_sint32_tmp( cp_sint32->get_pointer<mi::ISint32>());
    MI_CHECK_EQUAL( c_sint32.get(), c_sint32_tmp.get());

    // untyped IConst_pointer

    // create instance
    mi::base::Handle<mi::IConst_pointer> cp_untyped( transaction->create<mi::IConst_pointer>( "Const_pointer<Interface>"));
    MI_CHECK( cp_untyped);

    // check the type name
    MI_CHECK_EQUAL_CSTR( cp_untyped->get_type_name(), "Const_pointer<Interface>");

    // check initial value
    MI_CHECK_EQUAL( nullptr, cp_untyped->get_pointer());

    // set value
    MI_CHECK_EQUAL( 0, cp_untyped->set_pointer( nullptr));
    MI_CHECK_EQUAL( 0, cp_untyped->set_pointer( c_sint32.get()));
    MI_CHECK_EQUAL( 0, cp_untyped->set_pointer( c_attribute_container.get()));

    // get value
    mi::base::Handle<const mi::neuraylib::IAttribute_container> c_attribute_container_tmp(
        cp_untyped->get_pointer<mi::neuraylib::IAttribute_container>());
    MI_CHECK_EQUAL( c_attribute_container.get(), c_attribute_container_tmp.get());

    // intended use cases

    // (a) set/get const pointers in mutable collections
    mi::base::Handle<mi::IArray> m_array( transaction->create<mi::IArray>( "Const_pointer<Sint32>[42]"));
    mi::base::Handle<mi::IConst_pointer> cp( transaction->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    MI_CHECK_EQUAL( 0, cp->set_pointer( c_sint32.get()));
    MI_CHECK_EQUAL( 0, m_array->set_element( 0, cp.get()));
    mi::base::Handle<mi::IConst_pointer> cp_tmp( m_array->get_element<mi::IConst_pointer>( 0));
    c_sint32_tmp = cp->get_pointer<mi::ISint32>();
    MI_CHECK( c_sint32_tmp);

    // (b) get mutable pointers from const collections
    mi::base::Handle<const mi::IArray> c_array( transaction->create<mi::IArray>( "Pointer<Sint32>[42]"));
    mi::base::Handle<const mi::IPointer> p( c_array->get_element<mi::IPointer>( 0));
    m_sint32_tmp = p->get_pointer<mi::ISint32>();
    MI_CHECK( !m_sint32_tmp);
}

template <typename I>
void test_nested( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    mi::base::Handle<I> nested( transaction->create<I>( type_name));
    MI_CHECK( nested);

    mi::base::Handle<mi::IData> data( nested->template get_interface<mi::IData>());
    MI_CHECK( data);
    MI_CHECK_EQUAL_CSTR( data->get_type_name(), type_name);

    mi::base::Handle<mi::IData_collection> data_collection(
        nested->template get_interface<mi::IData_collection>());
    MI_CHECK( data_collection);

    mi::base::Handle<mi::IData> element(
        data_collection->get_value<mi::IData>( zero_size));
    if( element) { // only static, typed arrays are non-empty by default
        MI_CHECK( element->get_type_name());
    }
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope(
            database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        test_value<mi::IBoolean,    bool>(           transaction.get(), "Boolean", true, false);
        test_value<mi::ISint8,      mi::Sint8>(      transaction.get(), "Sint8", -1, -42);
        test_value<mi::ISint16,     mi::Sint16>(     transaction.get(), "Sint16", -1, -42);
        test_value<mi::ISint32,     mi::Sint32>(     transaction.get(), "Sint32", -1, -42);
        test_value<mi::ISint64,     mi::Sint64>(     transaction.get(), "Sint64", -1, -42);
        test_value<mi::IUint8,      mi::Uint8>(      transaction.get(), "Uint8", 1, 42);
        test_value<mi::IUint16,     mi::Uint16>(     transaction.get(), "Uint16", 1, 42);
        test_value<mi::IUint32,     mi::Uint32>(     transaction.get(), "Uint32", 1, 42);
        test_value<mi::IUint64,     mi::Uint64>(     transaction.get(), "Uint64", 1, 42);
        test_value<mi::IFloat32,    mi::Float32>(    transaction.get(), "Float32", 1.0f, 42.0f);
        test_value<mi::IFloat64,    mi::Float64>(    transaction.get(), "Float64", 1.0, 42.0);
        test_value<mi::ISize,       mi::Size>(       transaction.get(), "Size", 1, 42);
        test_value<mi::IDifference, mi::Difference>( transaction.get(), "Difference", -1, -42);

        test_value_attribute<mi::IBoolean,    bool>(        transaction.get(), "Boolean", true, false);
        test_value_attribute<mi::ISint8,      mi::Sint8>(   transaction.get(), "Sint8", -1, -42);
        test_value_attribute<mi::ISint16,     mi::Sint16>(  transaction.get(), "Sint16", -1, -42);
        test_value_attribute<mi::ISint32,     mi::Sint32>(  transaction.get(), "Sint32", -1, -42);
        test_value_attribute<mi::ISint64,     mi::Sint64>(  transaction.get(), "Sint64", -1, -42);
        test_value_attribute<mi::IFloat32,    mi::Float32>( transaction.get(), "Float32", 1.0f, 42.0f);
        test_value_attribute<mi::IFloat64,    mi::Float64>( transaction.get(), "Float64", 1.0, 42.0);

        test_string( transaction.get(), "String", "foo", "bar");

        test_ref( transaction.get(), "Ref");
        test_ref_array( transaction.get());

        test_uuid( transaction.get());

        test_void( transaction.get());

        test_pointer( transaction.get());

        // nested types, typed
        test_nested<mi::IArray>(         transaction.get(), "Sint32[5][7]");
        test_nested<mi::IArray>(         transaction.get(), "Sint32[][7]");
        test_nested<mi::IDynamic_array>( transaction.get(), "Sint32[5][]");
        test_nested<mi::IDynamic_array>( transaction.get(), "Sint32[][]");
        test_nested<mi::IMap>(           transaction.get(), "Map<Map<Sint32>>");
        test_nested<mi::IArray>(         transaction.get(), "Map<Sint32>[7]");
        test_nested<mi::IDynamic_array>( transaction.get(), "Map<Sint32>[]");
        test_nested<mi::IMap>(           transaction.get(), "Map<Sint32[7]>");
        test_nested<mi::IMap>(           transaction.get(), "Map<Sint32[]>");

        // nested types, untyped
        test_nested<mi::IArray>(         transaction.get(), "Interface[5][7]");
        test_nested<mi::IArray>(         transaction.get(), "Interface[][7]");
        test_nested<mi::IDynamic_array>( transaction.get(), "Interface[5][]");
        test_nested<mi::IDynamic_array>( transaction.get(), "Interface[][]");
        test_nested<mi::IMap>(           transaction.get(), "Map<Map<Interface>>");
        test_nested<mi::IArray>(         transaction.get(), "Map<Interface>[7]");
        test_nested<mi::IDynamic_array>( transaction.get(), "Map<Interface>[]");
        test_nested<mi::IMap>(           transaction.get(), "Map<Interface[7]>");
        test_nested<mi::IMap>(           transaction.get(), "Map<Interface[]>");

        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_types )
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

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

